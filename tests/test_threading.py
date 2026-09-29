from dataclasses import dataclass
from datetime import UTC, datetime

from mr_mem.memory.product import MemoryProductStore
from mr_mem.memory.store import CanonicalMemoryStore
from mr_mem.memory.threading import ThreadAutoUpdateService
from tests.conftest import memory

NOW = datetime(2026, 9, 29, tzinfo=UTC)


@dataclass(frozen=True)
class Event:
    scope: object
    attributes: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]


def event(scope, ref, summary):
    return Event(
        scope=scope,
        attributes=(
            ("thread_action", "track"),
            ("thread_question", "Will I replace my laptop?"),
            ("thread_summary", summary),
        ),
        evidence_refs=(ref,),
    )


class Compiler:
    def __init__(self):
        self.calls = []

    def compile(self, thread):
        self.calls.append(thread)
        return "baseline-1"


def test_thread_accepts_structural_semantic_events_and_matures_cross_turn(tmp_path):
    path = tmp_path / "memory.sqlite"
    canonical = CanonicalMemoryStore(path)
    first = memory("m1", content="I want a new laptop.", interaction_id="turn-1")
    second = memory("m2", content="The old laptop is slow.", interaction_id="turn-2")
    canonical._commit((first, second))
    product = MemoryProductStore(path, canonical)
    service = ThreadAutoUpdateService(canonical=canonical, product=product)

    opened = service.apply(
        scope=first.scope,
        accepted_events=(event(first.scope, "evidence-m1", "Replacement is under consideration."),),
        at=NOW,
    )[0]
    assert not opened.mature

    matured = service.apply(
        scope=first.scope,
        accepted_events=(
            event(first.scope, "evidence-m2", "Performance pressure keeps it active."),
        ),
        at=NOW,
    )[0]
    assert matured.thread_id == opened.thread_id
    assert matured.mature
    assert matured.current_support_ids == ("m1", "m2")
    service.close()


def test_same_interaction_support_cannot_fake_maturity(tmp_path):
    path = tmp_path / "memory.sqlite"
    canonical = CanonicalMemoryStore(path)
    first = memory("m1", content="first", interaction_id="same-turn")
    second = memory("m2", content="second", interaction_id="same-turn")
    canonical._commit((first, second))
    product = MemoryProductStore(path, canonical)
    service = ThreadAutoUpdateService(canonical=canonical, product=product)

    service.apply(
        scope=first.scope,
        accepted_events=(event(first.scope, "evidence-m1", "first support"),),
        at=NOW,
    )
    updated = service.apply(
        scope=first.scope,
        accepted_events=(event(first.scope, "evidence-m2", "second row same turn"),),
        at=NOW,
    )[0]
    assert not updated.mature
    service.close()


def test_accepted_higher_projection_retires_temporary_thread(tmp_path):
    path = tmp_path / "memory.sqlite"
    canonical = CanonicalMemoryStore(path)
    first = memory("m1", content="I want a new laptop.", interaction_id="turn-1")
    second = memory("m2", content="The old laptop is slow.", interaction_id="turn-2")
    canonical._commit((first, second))
    product = MemoryProductStore(path, canonical)
    compiler = Compiler()
    service = ThreadAutoUpdateService(
        canonical=canonical,
        product=product,
        projection_compiler=compiler,
    )

    opened = service.apply(
        scope=first.scope,
        accepted_events=(event(first.scope, "evidence-m1", "Replacement is under consideration."),),
        at=NOW,
    )[0]
    service.apply(
        scope=first.scope,
        accepted_events=(event(first.scope, "evidence-m2", "Replacement logic is now explicit."),),
        at=NOW,
    )
    assert len(compiler.calls) == 1
    assert compiler.calls[0].mature
    assert product.get_thread(opened.thread_id) is None
    service.close()
