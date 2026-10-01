from dataclasses import replace
from datetime import UTC, datetime

import pytest

from mr_mem import (
    MemoryCore,
    MemoryLifecycle,
    MemoryRetrievalQuery,
    MemoryRetrievalService,
    RetrievedMemoryCandidate,
    Scope,
    ScopeDomain,
    SemanticAdmissionService,
    SemanticMemoryCandidate,
    SourceRef,
    ThreadAutoUpdateService,
)

OCCURRED = datetime(2026, 8, 1, tzinfo=UTC)
KNOWN = datetime(2026, 10, 1, tzinfo=UTC)
SCOPE = Scope(ScopeDomain.USER, user_id="u1")


class Host:
    """The sole raw store; the sidecar receives metadata only."""

    def __init__(self):
        self.refs = {}
        self.bodies = {}

    def put(self, record, body, revision="1"):
        ref = SourceRef("native-host", "session-1", record, OCCURRED, revision)
        self.refs[ref.source_key] = ref
        self.bodies[ref.source_key] = body
        return ref

    def current_ref(self, scope, ref):
        return self.refs.get(ref.source_key) if scope == SCOPE else None


class Clock:
    def now(self):
        return KNOWN


def candidate(ref, semantic_id="preference", content="User requires reproducible evidence."):
    return SemanticMemoryCandidate(
        semantic_id=semantic_id,
        scope=SCOPE,
        content=content,
        source_refs=(ref,),
        compiler_version="host-v1",
        attributes=(("subject", "user"), ("polarity", "positive")),
    )


def service(core, host):
    return SemanticAdmissionService(
        store=core.canonical, sources=host, clock=Clock(), origin_runtime_id="host"
    )


def test_native_semantics_without_raw_mirror_survive_restart(tmp_path):
    host = Host()
    raw = "RAW-ONLY: transient test log ... also require reproducible evidence."
    ref = host.put("turn-1", raw)
    path = tmp_path / "semantic.sqlite"
    with MemoryCore(path) as core:
        committed = service(core, host).admit(candidate(ref))
        assert committed.provenance.source_refs == (ref,)
        assert committed.provenance.evidence_refs == ()
        assert committed.provenance.observation_id is None
        assert committed.occurred_at == OCCURRED
        assert committed.known_at == KNOWN
        assert committed.attributes == candidate(ref).attributes
    with MemoryCore(path) as core:
        assert service(core, host).admit(candidate(ref)) == committed
        assert core.load_all() == (committed,)
    assert host.bodies[ref.source_key] == raw
    assert b"RAW-ONLY" not in path.read_bytes()


def test_missing_stale_and_changed_replay_fail_closed(tmp_path):
    host = Host()
    ref = host.put("turn-1", "raw")
    with MemoryCore(tmp_path / "semantic.sqlite") as core:
        admission = service(core, host)
        admission.admit(candidate(ref))
        with pytest.raises(ValueError, match="conflict"):
            admission.admit(candidate(ref, content="Different meaning under the same identity."))
        host.put("turn-1", "edited", revision="2")
        with pytest.raises(ValueError, match="source"):
            admission.admit(candidate(ref, "stale"))
        host.refs.clear()
        with pytest.raises(ValueError, match="source"):
            admission.admit(candidate(ref, "missing"))


@pytest.mark.parametrize("field,value", [
    ("source_namespace", ""), ("record_id", ""), ("occurred_at", datetime(2026, 8, 1)),
])
def test_source_ref_validates_identity_and_time(field, value):
    ref = SourceRef("native-host", "session-1", "turn-1", OCCURRED, "1")
    with pytest.raises(ValueError):
        replace(ref, **{field: value})


@pytest.mark.parametrize("deleted", [False, True])
def test_source_revision_or_deletion_invalidates_only_affected_semantics(tmp_path, deleted):
    host = Host()
    ref = host.put("turn-1", "one raw record")
    other = host.put("turn-2", "independent raw record")
    path = tmp_path / "semantic.sqlite"
    with MemoryCore(path) as core:
        admission = service(core, host)
        first = admission.admit(candidate(ref))
        independent = admission.admit(candidate(other))
        new_ref = host.put("turn-1", "edited", revision="2")
        changed = core.canonical.invalidate_source(SCOPE, new_ref, deleted=deleted)
        assert changed == (first.memory_id,)
        assert core.canonical.invalidate_source(SCOPE, new_ref, deleted=deleted) == ()
        assert core.get(first.memory_id).lifecycle is MemoryLifecycle.INVALIDATED
        assert core.get(independent.memory_id) == independent
        if not deleted:
            replacement = admission.admit(replace(
                candidate(new_ref), supersedes_memory_id=first.memory_id,
                contradicts_memory_ids=(first.memory_id,),
            ))
            assert replacement.memory_id != first.memory_id
            assert core.get(first.memory_id).lifecycle is MemoryLifecycle.SUPERSEDED
    with MemoryCore(path, read_only=True) as core:
        assert core.get(independent.memory_id) == independent
        assert core.get(first.memory_id).lifecycle is not MemoryLifecycle.ACTIVE


def test_semantic_thread_and_retrieval_use_native_provenance(tmp_path):
    host = Host()
    first = host.put("turn-1", "build log plus preference")
    second = host.put("turn-2", "independent followup")
    attrs = (
        ("thread_action", "track"), ("thread_question", "How to verify changes?"),
        ("thread_summary", "Require reproducible checks."),
    )
    with MemoryCore(tmp_path / "semantic.sqlite") as core:
        admission = service(core, host)
        one = admission.admit(replace(candidate(first), attributes=attrs))
        duplicate = admission.admit(replace(candidate(first, "second-fragment"), attributes=attrs))
        two = admission.admit(replace(candidate(second), attributes=attrs))
        threads = ThreadAutoUpdateService(canonical=core.canonical, product=core.products)
        opened = threads.apply(scope=SCOPE, accepted_events=(one,), at=KNOWN)[0]
        repeated = threads.apply(scope=SCOPE, accepted_events=(duplicate,), at=KNOWN)[0]
        assert not repeated.mature
        mature = threads.apply(scope=SCOPE, accepted_events=(two,), at=KNOWN)[0]
        assert mature.thread_id == opened.thread_id
        assert mature.mature
        assert set(mature.handoff_memory_ids) == {one.memory_id, duplicate.memory_id, two.memory_id}

        class Provider:
            def search(self, query):
                return tuple(RetrievedMemoryCandidate(m.memory_id, "test") for m in (one, two))

        retrieval = MemoryRetrievalService(store=core.canonical, provider=Provider())
        assert len(retrieval.search(MemoryRetrievalQuery(SCOPE, "verification"))) == 2
        core.canonical.invalidate_source(SCOPE, first, deleted=True)
        assert tuple(
            hit.memory.memory_id for hit in retrieval.search(MemoryRetrievalQuery(SCOPE, "checks"))
        ) == (two.memory_id,)
        assert core.products.surface_threads(SCOPE, now=KNOWN) == ()
