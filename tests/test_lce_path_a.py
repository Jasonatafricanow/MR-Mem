"""Path A over real LCE: Thread maturity, crash replay, first-appearance commit."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pytest

from mr_mem import MemoryCore, SemanticAdmissionService, SemanticMemoryCandidate, SourceRef
from mr_mem.contracts import Scope
from mr_mem.memory.threading import ThreadAutoUpdateService
from tests.test_native_semantics import SCOPE

pytest.importorskip("lce")

from mr_mem.integrations.lce_path_a import LcePathA, thread_region  # noqa: E402

T0 = datetime(2026, 8, 1, tzinfo=UTC)


class Host:
    def __init__(self):
        self.refs = {}

    def put(self, record, minutes):
        ref = SourceRef("host", "s1", record, T0 + timedelta(minutes=minutes), "1")
        self.refs[ref.source_key] = ref
        return ref

    def current_ref(self, scope, ref):
        return self.refs.get(ref.source_key) if scope == SCOPE else None


class Clock:
    def now(self):
        return datetime(2026, 10, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Event:
    scope: Scope
    attributes: tuple[tuple[str, str], ...]
    evidence_refs: tuple[str, ...]


def admit(core, host, record, minutes, content):
    ref = host.put(record, minutes)
    svc = SemanticAdmissionService(
        store=core.canonical, sources=host, clock=Clock(), origin_runtime_id="t"
    )
    return svc.admit(
        SemanticMemoryCandidate(
            semantic_id=record, scope=SCOPE, content=content,
            source_refs=(ref,), compiler_version="test",
        )
    )


def track(memory, question, summary, mature=False):
    attrs = [("thread_action", "track"), ("thread_question", question),
             ("thread_summary", summary)]
    if mature:
        attrs.append(("thread_mature", "true"))
    return Event(SCOPE, tuple(attrs), (memory.memory_id,))


def service(core, compiler):
    return ThreadAutoUpdateService(
        canonical=core.canonical, product=core.products, projection_compiler=compiler
    )


NOW = datetime(2026, 10, 2, tzinfo=UTC)


def test_mature_thread_compiles_to_baseline_and_retires(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "I want to replace my laptop")
        b = admit(core, host, "r2", 5, "The new model is too expensive, I will wait")
        svc = service(core, lce)
        (opened,) = svc.apply(scope=SCOPE, accepted_events=(
            track(a, "Replace the laptop?", "Considering a replacement"),), at=NOW)
        assert not opened.mature
        assert lce.accepted() == ()
        svc.apply(scope=SCOPE, accepted_events=(
            track(b, "Replace the laptop?", "Postponed because of price", mature=True),),
            at=NOW + timedelta(minutes=1))
        assert core.products.get_thread(opened.thread_id) is None
        (head,) = lce.accepted()
        assert head.region_id == thread_region(opened.thread_id)
        assert head.content == "Postponed because of price"
        assert set(head.supporting_memory_ids) == {a.memory_id, b.memory_id}
        assert head.origin == "thread" and head.evidence_complete


def test_single_interaction_cannot_mature_a_thread(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "Plan A")
        svc = service(core, lce)
        (t,) = svc.apply(scope=SCOPE, accepted_events=(
            track(a, "Which plan?", "Plan A", mature=True),), at=NOW)
        assert not t.mature and lce.accepted() == ()


class RetireCrash:
    """LCE accepted the Baseline, then the process died before Thread retirement."""

    def __init__(self, inner):
        self.inner, self.armed = inner, True

    def compile(self, thread):
        baseline_id = self.inner.compile(thread)
        if self.armed:
            self.armed = False
            raise SystemExit("crash after LCE commit")
        return baseline_id


def test_crash_after_lce_commit_reconciles_without_new_revision(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "fact one")
        b = admit(core, host, "r2", 5, "fact two")
        crashy = RetireCrash(lce)
        svc = service(core, crashy)
        svc.apply(scope=SCOPE, accepted_events=(track(a, "Q?", "s1"),), at=NOW)
        with pytest.raises(SystemExit):
            svc.apply(scope=SCOPE, accepted_events=(track(b, "Q?", "s2", mature=True),),
                      at=NOW + timedelta(minutes=1))
        (thread,) = core.products.list_threads(SCOPE)
        assert thread.mature
        # Restart: the stateless rescan finds the mature Thread and replays.
        service(core, lce).apply(scope=SCOPE, accepted_events=(), at=NOW + timedelta(hours=1))
        assert core.products.list_threads(SCOPE) == ()
        (head,) = lce.accepted()
        assert head.revision_number == 1


def test_compile_failure_keeps_thread_for_next_attempt(tmp_path):
    host = Host()

    class Down:
        def compile(self, thread):
            return None

    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "x one")
        b = admit(core, host, "r2", 5, "x two")
        svc = service(core, Down())
        svc.apply(scope=SCOPE, accepted_events=(track(a, "Q?", "s1"),), at=NOW)
        svc.apply(scope=SCOPE, accepted_events=(track(b, "Q?", "s2", mature=True),),
                  at=NOW + timedelta(minutes=1))
        assert core.products.list_threads(SCOPE)[0].mature
        service(core, lce).apply(scope=SCOPE, accepted_events=(), at=NOW + timedelta(hours=1))
        assert core.products.list_threads(SCOPE) == () and len(lce.accepted()) == 1


def test_first_appearance_commit_then_thread_revises_same_lineage(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "One-day volume spikes do not confirm a reversal")
        first = lce.commit(question="When is a breakout confirmed?",
                           summary="A single volume spike is not confirmation",
                           support=(a.memory_id,))
        assert lce.commit(question="When is a breakout confirmed?",
                          summary="A single volume spike is not confirmation",
                          support=(a.memory_id,)) == first  # idempotent replay
        (head,) = lce.accepted()
        assert head.origin == "commit" and head.revision_number == 1
        b = admit(core, host, "r2", 5, "Now I also require a retest of the range")
        c = admit(core, host, "r3", 9, "Retest held, so I treat it as confirmed")
        svc = service(core, lce)
        svc.apply(scope=SCOPE, accepted_events=(
            track(b, "When is a breakout confirmed?", "Needs a retest"),), at=NOW)
        svc.apply(scope=SCOPE, accepted_events=(
            track(c, "When is a breakout confirmed?", "Spike plus held retest", mature=True),),
            at=NOW + timedelta(minutes=1))
        (head2,) = lce.accepted()
        assert head2.region_id == head.region_id
        assert head2.revision_number == 2 and head2.origin == "thread"


def test_commit_rejects_unknown_support(tmp_path):
    with MemoryCore(tmp_path / "m.sqlite"), LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        with pytest.raises(ValueError):
            lce.commit(question="Q?", summary="s", support=("semantic-missing",))
        assert lce.accepted() == ()


def test_lost_evidence_marks_but_keeps_cognition(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "complete thought")
        lce.commit(question="Q?", summary="complete thought", support=(a.memory_id,))
        core.canonical.invalidate_source(SCOPE, a.provenance.source_refs[0], deleted=True)
        (head,) = lce.accepted()
        assert head.content == "complete thought" and not head.evidence_complete


def test_revise_continues_lineage_from_head(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "considering a new laptop, needs 16GB")
        b = admit(core, host, "r2", 5, "decided not to buy until after the trip")
        with pytest.raises(ValueError):
            lce.revise(question="Replace laptop?", summary="x", support=(b.memory_id,))
        lce.commit(question="Replace laptop?", summary="Considering a 16GB model",
                   support=(a.memory_id,))
        second = lce.revise(question="Replace laptop?", summary="Postponed until after the trip",
                            support=(b.memory_id,))
        assert lce.revise(question="Replace laptop?", summary="Postponed until after the trip",
                          support=(b.memory_id,)) == second  # replay
        (head,) = lce.accepted()
        assert head.revision_number == 2 and head.origin == "revision"
        assert head.supporting_memory_ids == (a.memory_id, b.memory_id)


def test_reopened_thread_revision_keeps_previous_evidence(tmp_path):
    host = Host()
    with MemoryCore(tmp_path / "m.sqlite") as core, LcePathA(
        canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE
    ) as lce:
        a = admit(core, host, "r1", 0, "budget capped at ten thousand")
        lce.commit(question="Budget?", summary="Capped at 10k", support=(a.memory_id,))
        b = admit(core, host, "r2", 5, "vendor approved")
        c = admit(core, host, "r3", 9, "contract signed")
        svc = service(core, lce)
        svc.apply(scope=SCOPE, accepted_events=(track(b, "Budget?", "Vendor approved"),), at=NOW)
        svc.apply(scope=SCOPE, accepted_events=(
            track(c, "Budget?", "10k cap, vendor approved, signed", mature=True),),
            at=NOW + timedelta(minutes=1))
        (head,) = lce.accepted()
        assert head.revision_number == 2
        assert set(head.supporting_memory_ids) == {a.memory_id, b.memory_id, c.memory_id}
