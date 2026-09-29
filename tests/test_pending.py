from datetime import UTC, datetime

from mr_mem import Scope, ScopeDomain
from mr_mem.memory.pending import (
    PendingStatus,
    PendingWorkingEvidence,
    PendingWorkingOverlay,
)

NOW = datetime(2026, 9, 29, tzinfo=UTC)


def item(pending_id="p1", turn="t1"):
    return PendingWorkingEvidence(
        pending_id=pending_id,
        evidence_ref=f"e-{pending_id}",
        source_turn_id=turn,
        scope=Scope(ScopeDomain.USER, user_id="u"),
        semantic_payload=(("topic", "laptop"),),
        confidence=0.9,
        status=PendingStatus.PENDING,
        origin_runtime_id="runtime",
        source_text="thinking about a laptop",
        created_at=NOW,
    )


def test_pending_overlay_is_bounded_noncanonical_lifecycle():
    overlay = PendingWorkingOverlay()
    overlay.add(item())
    assert len(overlay.get_pending()) == 1
    accepted = overlay.accept("p1")
    assert accepted is not None and accepted.status is PendingStatus.ACCEPTED
    assert len(overlay) == 0


def test_abort_and_restart_clear_only_working_overlay():
    overlay = PendingWorkingOverlay()
    overlay.add(item("p1", "turn-a"))
    overlay.add(item("p2", "turn-b"))
    assert overlay.clear_on_turn_abort("turn-a") == 1
    assert [row.pending_id for row in overlay.get_pending()] == ["p2"]
    assert overlay.clear_on_restart() == 1
    assert len(overlay) == 0
