"""Generic durable consumer registration and canonical mutation requeue."""

import sqlite3

import pytest

from mr_mem import MemoryCore, MemoryLifecycle
from mr_mem.memory.projection import ProjectionWorker
from tests.test_semantic_admission import Sources, service
from tests.test_semantic_validator import delta, edge


def admit(core, sources, name, payload=None, activated=()):
    return service(core, sources).admit_semantic_delta(
        payload or delta(), binding=sources.bind(name), activated_memory_ids=activated,
    ).memory_ids[0]


def test_registration_backfill_incremental_restart_and_embedding_compatibility(tmp_path):
    path = tmp_path / "memory.sqlite"
    sources = Sources()
    with MemoryCore(path) as core:
        old = admit(core, sources, "old")
        queue = core.canonical.projection_queue()
        assert queue.register_projection_target("embedding-v1") == 1
        assert queue.register_projection_target("lce-semantic-v1") == 1
        for intent in queue.pending(100):
            queue.succeed(intent.intent_id, intent.memory_id)
        assert queue.register_projection_target("lce-semantic-v1") == 0
        assert queue.pending(10) == ()
    with MemoryCore(path) as core:
        new = admit(core, sources, "new")
        queue = core.canonical.projection_queue()
        intents = queue.pending(10)
        assert {i.target for i in intents} == {"unassigned", "embedding-v1", "lce-semantic-v1"}
        assert {i.memory_id for i in intents} == {new}
        assert core.get(old).lifecycle is MemoryLifecycle.ACTIVE
        assert queue.rebuild(target="lce-semantic-v1", reset=True) == 2
        assert {i.memory_id for i in queue.pending(10, target="lce-semantic-v1")} == {old, new}


@pytest.mark.parametrize("effect", ["supersede", "none"])
def test_relation_and_lifecycle_mutation_requeues_existing_consumer(tmp_path, effect):
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        sources = Sources()
        old = admit(core, sources, "old")
        queue = core.canonical.projection_queue()
        queue.register_projection_target("lce-semantic-v1")
        original = queue.pending(10, target="lce-semantic-v1")[0]
        queue.succeed(original.intent_id, "projection-ref")
        payload = delta()
        payload["dependencies"] = [edge(target_id=old, lifecycle_effect=effect)]
        new = admit(core, sources, "new", payload, (old,))
        pending = queue.pending(10, target="lce-semantic-v1")
        assert {i.memory_id for i in pending} == {old, new}
        assert next(i for i in pending if i.memory_id == old).intent_id == original.intent_id
        assert core.get(old).lifecycle is (
            MemoryLifecycle.SUPERSEDED if effect == "supersede" else MemoryLifecycle.ACTIVE
        )


def test_registration_rolls_back_as_one_unit(tmp_path):
    path = tmp_path / "memory.sqlite"
    with MemoryCore(path) as core:
        admit(core, Sources(), "old")
        with sqlite3.connect(path) as db:
            db.execute("CREATE TRIGGER fail_target BEFORE INSERT ON projection_intents "
                       "WHEN NEW.target='broken' BEGIN SELECT RAISE(ABORT,'injected'); END")
        with pytest.raises(sqlite3.IntegrityError):
            core.canonical.projection_queue().register_projection_target("broken")
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT * FROM projection_targets").fetchall() == []


def test_worker_ack_cannot_erase_mutation_during_delivery(tmp_path):
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        mid = admit(core, Sources(), "old")
        queue = core.canonical.projection_queue()
        queue.register_projection_target("consumer-v1")

        class MutatingWriter:
            def upsert(self, memory, *, intent):
                with core.canonical._transaction():
                    core.canonical._set_lifecycle(core.get(mid), MemoryLifecycle.ARCHIVED)
                return memory.memory_id

        assert ProjectionWorker(queue, MutatingWriter(), target="consumer-v1").run_once() == (1, 0)
        assert len(queue.pending(10, target="consumer-v1")) == 1
