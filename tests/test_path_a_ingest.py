"""Block signal routing: Thread, first-appearance commit, revision of an accepted line."""

from datetime import timedelta

import pytest

from mr_mem import MemoryCore
from tests.test_lce_path_a import NOW, SCOPE, Host, admit

pytest.importorskip("lce")

from mr_mem.integrations.lce_path_a import LcePathA  # noqa: E402
from mr_mem.integrations.path_a_ingest import (  # noqa: E402
    BlockSignals,
    CommitSignal,
    ThreadSignal,
    apply_block_signals,
)


@pytest.fixture
def env(tmp_path):
    core = MemoryCore(tmp_path / "m.sqlite")
    lce = LcePathA(canonical_path=tmp_path / "m.sqlite", lce_root=tmp_path / "lce", scope=SCOPE)
    host = Host()

    def apply(*blocks, at=NOW, with_lce=True):
        return apply_block_signals(
            canonical=core.canonical, products=core.products,
            lce=lce if with_lce else None, scope=SCOPE, blocks=blocks, at=at,
        )

    def block(record, minutes, content, thread=None, commit=None):
        memory = admit(core, host, record, minutes, content)
        return BlockSignals(memory.memory_id, content, thread, commit)

    yield core, lce, apply, block
    lce.close()
    core.close()


def test_thread_matures_across_batches_and_reaches_lce(env):
    core, lce, apply, block = env
    a = block("r1", 0, "thinking about a new bike",
              ThreadSignal("track", "New bike?", "Considering"))
    result = apply(a)
    assert result.questions == ("New bike?",) and not result.thread_failed
    assert lce.accepted() == () and len(core.products.list_threads(SCOPE)) == 1
    b = block("r2", 5, "bought the gravel bike",
              ThreadSignal("track", "New bike?", "Bought a gravel bike", mature=True))
    apply(b, at=NOW + timedelta(minutes=1))
    (head,) = lce.accepted()
    assert head.origin == "thread" and head.content == "Bought a gravel bike"
    assert core.products.list_threads(SCOPE) == ()


def test_commit_then_settling_signal_revises_the_line(env):
    _, lce, apply, block = env
    a = block("r1", 0, "rent is capped at 2000",
              commit=CommitSignal("Housing budget?", "Rent capped at 2000"))
    assert apply(a).promotion_failures == 0
    b = block("r2", 5, "raised the cap to 2400",
              ThreadSignal("resolve", "Housing budget?", "Rent capped at 2400"))
    apply(b)
    (head,) = lce.accepted()
    assert head.revision_number == 2 and head.origin == "revision"
    assert head.content == "Rent capped at 2400"
    assert head.supporting_memory_ids == (a.memory_id, b.memory_id)


def test_non_settling_signal_on_accepted_line_opens_a_thread(env):
    core, lce, apply, block = env
    apply(block("r1", 0, "rent is capped at 2000",
                commit=CommitSignal("Housing budget?", "Rent capped at 2000")))
    apply(block("r2", 5, "maybe I could stretch the budget",
                ThreadSignal("track", "Housing budget?", "Maybe stretch")))
    assert lce.accepted()[0].revision_number == 1
    assert len(core.products.list_threads(SCOPE)) == 1


def test_replay_is_idempotent(env):
    _, lce, apply, block = env
    a = block("r1", 0, "rent is capped at 2000",
              commit=CommitSignal("Housing budget?", "Rent capped at 2000"))
    apply(a)
    apply(a)
    (head,) = lce.accepted()
    assert head.revision_number == 1


def test_promotion_failure_is_reported_not_raised(env):
    _, lce, apply, block = env
    bad = BlockSignals("semantic-missing", "x", commit=CommitSignal("Q?", "s"))
    assert apply(bad).promotion_failures == 1
    assert lce.accepted() == ()


def test_without_lce_only_threads_are_maintained(env):
    core, lce, apply, block = env
    apply(block("r1", 0, "a", commit=CommitSignal("Q?", "s"),
                thread=ThreadSignal("track", "Other?", "t")), with_lce=False)
    assert lce.accepted() == () and len(core.products.list_threads(SCOPE)) == 1
