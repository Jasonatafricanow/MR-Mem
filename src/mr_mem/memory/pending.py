"""M1A-C9W1B: Pre-Admission Pending Working Overlay.

This module is the C9-W1B implementation. It introduces a NEW bounded
context, not a rename of any existing concept. The hard rule:

  PendingWorkingEvidence != Observation
  PendingWorkingEvidence != Canonical State
  PendingWorkingEvidence != Canonical Memory

The overlay is a NON-CANONICAL, NON-DURABLE store of pre-admission evidence
that the next-turn compiler can read alongside canonical context. It is
explicitly forbidden from:
  - mutating canonical state on its own
  - reinforcing memory
  - modifying persona, relationship, or scope authority
  - surviving process restart

Lifecycle: PENDING -> ACCEPTED (promoted to canonical via fact_ingest.admit)
                     -> REJECTED (cleared without promotion)

The orchestrator owns the overlay; the compiler consumes pending items via
its existing `authorized_external_items` field. No compiler modification
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from mr_mem.contracts.common import (
    require_aware_utc,
    require_non_empty,
)
from mr_mem.contracts.scope import Scope, ScopeDomain

class PendingStatus(StrEnum):
    """Lifecycle status of a PendingWorkingEvidence entry."""

    PENDING = "pending"  # Held in overlay, not in canonical state
    ACCEPTED = "accepted"  # Promoted to canonical via fact_ingest.admit
    REJECTED = "rejected"  # Cleared without promotion


# ============================================================================
# PendingWorkingEvidence — the pre-admission overlay entry
# ============================================================================


@dataclass(frozen=True, slots=True)
class PendingWorkingEvidence:
    """Pre-admission evidence held in a bounded non-canonical overlay.

    Distinct from Observation, Canonical State, and Canonical Memory.
    Carries provenance, scope, and an evidence_ref. By default NON-DURABLE
    across process restart.

    Hard invariants:
      - source_text is the original evidence text (for audit)
      - semantic_payload is the bounded (key, value) interpretation
      - status transitions: PENDING -> ACCEPTED or PENDING -> REJECTED
      - Cannot mutate canonical state on its own
      - Cannot reinforce memory
      - Cannot modify persona, relationship, or scope authority
    """

    pending_id: str
    evidence_ref: str
    source_turn_id: str
    scope: Scope
    semantic_payload: tuple[tuple[str, str], ...]
    confidence: float
    status: PendingStatus
    origin_runtime_id: str
    source_text: str
    created_at: datetime

    def __post_init__(self) -> None:
        require_non_empty(self.pending_id, "pending_id")
        require_non_empty(self.evidence_ref, "evidence_ref")
        require_non_empty(self.source_turn_id, "source_turn_id")
        require_non_empty(self.origin_runtime_id, "origin_runtime_id")
        if not isinstance(self.scope, Scope):
            raise ValueError("scope must be a Scope")
        if not isinstance(self.status, PendingStatus):
            raise ValueError("status must be a PendingStatus")
        if isinstance(self.confidence, bool) or not isinstance(
            self.confidence, (int, float)
        ):
            raise ValueError("confidence must be a number")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        require_aware_utc(self.created_at, "created_at")
        # semantic_payload is a tuple of (key, value) pairs; reject empty
        if not self.semantic_payload:
            raise ValueError("semantic_payload must not be empty")
        for key, value in self.semantic_payload:
            require_non_empty(key, "semantic_payload keys")
            require_non_empty(value, "semantic_payload values")

    def with_status(self, new_status: PendingStatus) -> "PendingWorkingEvidence":
        """Return a new PendingWorkingEvidence with the given status."""
        if not isinstance(new_status, PendingStatus):
            raise ValueError("new_status must be a PendingStatus")
        return PendingWorkingEvidence(
            pending_id=self.pending_id,
            evidence_ref=self.evidence_ref,
            source_turn_id=self.source_turn_id,
            scope=self.scope,
            semantic_payload=self.semantic_payload,
            confidence=self.confidence,
            status=new_status,
            origin_runtime_id=self.origin_runtime_id,
            source_text=self.source_text,
            created_at=self.created_at,
        )


# ============================================================================
# PendingWorkingOverlay — non-canonical, non-durable, in-memory store
# ============================================================================


class PendingWorkingOverlay:
    """Non-canonical, non-durable in-memory store of pre-admission evidence.

    Scope is the access key: get_pending(scope) returns only items whose
    scope is subsumed by the requested scope. REJECTED and ACCEPTED items
    are removed from the store; only PENDING items are returned.

    This store is NOT durable: process restart → all items lost. This is
    the correct semantic: pending evidence is "刚刚说过但还没正式入库"
    (just said but not yet committed), and losing it on restart is consistent
    with "the user can say it again in a new session".
    """

    def __init__(self) -> None:
        self._items: dict[str, PendingWorkingEvidence] = {}

    def add(self, pending: PendingWorkingEvidence) -> None:
        """Add a new PENDING item to the overlay.

        Raises ValueError if pending_id is already present, or if the
        item is not in PENDING status.
        """
        if not isinstance(pending, PendingWorkingEvidence):
            raise ValueError("pending must be a PendingWorkingEvidence")
        if pending.status is not PendingStatus.PENDING:
            raise ValueError("only PENDING items can be added to the overlay")
        if pending.pending_id in self._items:
            raise ValueError(f"pending_id {pending.pending_id!r} already in overlay")
        self._items[pending.pending_id] = pending

    def get_pending(
        self, scope: Scope | None = None
    ) -> tuple[PendingWorkingEvidence, ...]:
        """Return all PENDING items, optionally filtered by scope.

        If scope is None: returns all PENDING items.
        If scope is given: returns PENDING items whose scope exactly matches.
        """
        items = tuple(
            item for item in self._items.values() if item.status is PendingStatus.PENDING
        )
        if scope is None:
            return items
        return tuple(item for item in items if item.scope == scope)

    def get_pending_by_turn(self, turn_id: str) -> tuple[PendingWorkingEvidence, ...]:
        """Return all PENDING items created by the given turn."""
        return tuple(
            item
            for item in self._items.values()
            if item.status is PendingStatus.PENDING and item.source_turn_id == turn_id
        )

    def get_by_pending_id(
        self, pending_id: str
    ) -> PendingWorkingEvidence | None:
        """Return the item with the given pending_id, or None."""
        return self._items.get(pending_id)

    def accept(self, pending_id: str) -> PendingWorkingEvidence | None:
        """Mark a PENDING item as ACCEPTED and remove it from the overlay.

        Returns the ACCEPTED item (with status updated), or None if the
        pending_id was not present.

        The caller is responsible for canonical promotion via fact_ingest.admit;
        this method only manages the overlay lifecycle.
        """
        item = self._items.get(pending_id)
        if item is None:
            return None
        if item.status is not PendingStatus.PENDING:
            raise ValueError(
                f"only PENDING items can be accepted; {pending_id!r} is {item.status.value}"
            )
        accepted = item.with_status(PendingStatus.ACCEPTED)
        del self._items[pending_id]
        return accepted

    def reject(self, pending_id: str) -> PendingWorkingEvidence | None:
        """Mark a PENDING item as REJECTED and remove it from the overlay.

        Returns the REJECTED item (with status updated), or None if the
        pending_id was not present.
        """
        item = self._items.get(pending_id)
        if item is None:
            return None
        if item.status is not PendingStatus.PENDING:
            raise ValueError(
                f"only PENDING items can be rejected; {pending_id!r} is {item.status.value}"
            )
        rejected = item.with_status(PendingStatus.REJECTED)
        del self._items[pending_id]
        return rejected

    def clear_on_turn_abort(self, turn_id: str) -> int:
        """Drop all PENDING items from a turn (called on abort_turn).

        Returns the number of items removed.
        """
        to_drop = [
            pending_id
            for pending_id, item in self._items.items()
            if item.source_turn_id == turn_id
            and item.status is PendingStatus.PENDING
        ]
        for pending_id in to_drop:
            del self._items[pending_id]
        return len(to_drop)

    def clear_on_restart(self) -> int:
        """Drop all items (called on process restart).

        Returns the number of items removed.
        """
        count = len(self._items)
        self._items.clear()
        return count

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, pending_id: object) -> bool:
        return pending_id in self._items


__all__ = [
    "PendingStatus",
    "PendingWorkingEvidence",
    "PendingWorkingOverlay",
]
