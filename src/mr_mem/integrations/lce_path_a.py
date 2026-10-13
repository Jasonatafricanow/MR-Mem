"""Optional LCE Path A binding over one MR-Mem canonical database.

Ported from Mind-Runtime ``integrations/lce.py`` (``LceThreadProjectionCompiler``
and ``open_lce_thread_handoff``; ADR-0033 2026-09-28 addendum):

    mature Thread -> PrecomputedDraftInput -> LCE Worktree -> Baseline

plus one scoped extension recorded in docs/PATH_A.md: cognition the semantic
authority marks as already complete at its first appearance is handed to the
same Baseline lineage without first passing through Thread maturity. The
Thread two-interaction rule itself is unchanged.

MR-Mem stays the factual authority. LCE owns only derived Draft/Baseline
state. No model is called here; summaries were already reasoned upstream.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

from mr_mem.contracts import Scope
from mr_mem.contracts.common import require_non_empty
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.product import MemoryProductStore, MemoryThread, ThreadStatus
from mr_mem.memory.store import CanonicalMemoryStore, scope_json
from mr_mem.memory.threading import _thread_identity

THREAD_REGION_PREFIX = "mr-thread:"
MAX_HANDOFF_SUPPORT = 100


class LceIntegrationUnavailable(RuntimeError):
    """The optional ``lce-core`` package is not installed."""


class SupportSelectionError(ValueError):
    """Selected support is missing, out of Scope, or not ACTIVE."""


def thread_region(thread_id: str) -> str:
    require_non_empty(thread_id, "thread_id")
    return THREAD_REGION_PREFIX + thread_id


def scope_lce_root(root: Path | str, scope: Scope) -> Path:
    """One LCE store per Scope, addressed like Mind-Runtime's per-Scope root."""
    digest = hashlib.sha256(scope_json(scope).encode("utf-8")).hexdigest()
    return Path(root) / digest


class CanonicalSubstrate:
    """LCE ``MemorySubstratePort`` resolved against canonical MR-Mem Memory.

    Connections are read-only and short-lived, so a long-lived LCE session
    never serves cached Memory after a canonical change.
    """

    def __init__(self, canonical_path: Path | str, scope: Scope) -> None:
        if not isinstance(scope, Scope):
            raise TypeError("scope must be a structured Scope")
        self.canonical_path = Path(canonical_path)
        self.scope = scope

    def memories(
        self, memory_ids: tuple[str, ...], *, active_only: bool = True
    ) -> tuple[CommittedMemory, ...]:
        if (
            not isinstance(memory_ids, tuple)
            or not 1 <= len(memory_ids) <= MAX_HANDOFF_SUPPORT
            or len(set(memory_ids)) != len(memory_ids)
            or any(not isinstance(mid, str) or not mid.strip() for mid in memory_ids)
        ):
            raise SupportSelectionError("expected 1..100 unique stable Memory IDs")
        store = CanonicalMemoryStore(self.canonical_path, read_only=True)
        try:
            found = tuple(store.get(mid) for mid in memory_ids)
        finally:
            store.close()
        if any(
            memory is None
            or memory.scope != self.scope
            or (active_only and memory.lifecycle is not MemoryLifecycle.ACTIVE)
            for memory in found
        ):
            raise SupportSelectionError("selected support is unavailable or ineligible")
        return tuple(memory for memory in found if memory is not None)

    def get_by_ids(self, memory_ids: tuple[str, ...]):
        try:
            from lce.contracts.external_memory import MemoryItemView
        except ImportError as exc:
            raise LceIntegrationUnavailable("install mr-mem[lce]") from exc
        return tuple(
            MemoryItemView(
                memory.memory_id,
                memory.content,
                tuple(ref.source_key for ref in memory.provenance.source_refs)
                or memory.provenance.evidence_refs,
                MappingProxyType({}),
            )
            for memory in self.memories(memory_ids)
        )


@dataclass(frozen=True, slots=True)
class AcceptedUnderstanding:
    """Current Baseline HEAD of one lineage, resolved back to canonical support.

    ``evidence_complete`` is False when some support no longer resolves to an
    ACTIVE canonical Memory. Such cognition is kept and marked, never silently
    dropped: it existed at its time point (docs/PATH_A.md, retention rule).
    """

    content: str
    baseline_id: str
    region_id: str
    revision_number: int
    created_at: datetime
    origin: str
    supporting_memory_ids: tuple[str, ...]
    evidence_complete: bool


class LcePathA:
    """No-model Path A session for one Scope.

    Implements MR-Mem's ``ThreadProjectionCompiler`` port, so it plugs into
    ``ThreadAutoUpdateService(projection_compiler=...)`` exactly as Mind-Runtime
    wires ``LceThreadProjectionCompiler``.
    """

    def __init__(
        self, *, canonical_path: Path | str, lce_root: Path | str, scope: Scope
    ) -> None:
        try:
            from lce.cognition.worktree import DraftRevisionStore
            from lce.store.sqlite_store import SqliteBaselineStore
        except ImportError as exc:
            raise LceIntegrationUnavailable("install mr-mem[lce]") from exc
        self.scope = scope
        self.substrate = CanonicalSubstrate(canonical_path, scope)
        root = scope_lce_root(lce_root, scope)
        self._baselines = SqliteBaselineStore(root / "baselines")
        try:
            self._drafts = DraftRevisionStore(root / "worktrees")
        except BaseException:
            self._baselines.close()
            raise

    def close(self) -> None:
        self._drafts.close()
        self._baselines.close()

    def __enter__(self) -> LcePathA:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    # -- write side -----------------------------------------------------

    def _promote(
        self,
        *,
        region_id: str,
        content: str,
        support: tuple[str, ...],
        processing_input_id: str,
        context: dict[str, Any],
    ) -> str:
        from lce.cognition.external import PrecomputedDraftInput, PrecomputedDraftIntake

        # Revalidate every support point before LCE sees the draft.
        self.substrate.memories(support)
        draft = PrecomputedDraftInput(
            region_id=region_id,
            content=content,
            supporting_memory_ids=support,
            processing_input_id=processing_input_id,
            context=context,
        )
        intake = PrecomputedDraftIntake(
            memory_substrate=self.substrate,
            baseline_store=self._baselines,
            draft_store=self._drafts,
        )
        return intake.stage_and_promote(draft).baseline.baseline_id

    def compile(self, thread: MemoryThread) -> str | None:
        """Mature Thread -> accepted Baseline ID (same contract as Mind-Runtime)."""
        if not isinstance(thread, MemoryThread):
            raise TypeError("thread must be MemoryThread")
        if thread.scope != self.scope:
            raise SupportSelectionError("Thread Scope does not match the bound LCE Scope")
        canonical = CanonicalMemoryStore(self.substrate.canonical_path, read_only=True)
        try:
            product = MemoryProductStore(
                self.substrate.canonical_path, canonical, read_only=True
            )
            try:
                authoritative = product.thread_handoff(thread.thread_id)
            finally:
                product.close()
        finally:
            canonical.close()
        if authoritative != thread:
            raise ValueError("Thread must match durable MR product state")
        if thread.status is ThreadStatus.ABANDONED:
            raise ValueError("abandoned Thread cannot be handed off")
        assert thread.working_summary is not None
        # A reopened line continues its accepted lineage (V1 section 7: B1 + M4
        # -> B2), so the new revision keeps citing the evidence B1 rested on.
        support = self._with_previous_support(
            thread_region(thread.thread_id), thread.handoff_memory_ids
        )
        payload = json.dumps(
            {
                "thread_id": thread.thread_id,
                "support": support,
                "summary": thread.working_summary,
                "updated_at": thread.updated_at.isoformat(),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return self._promote(
            region_id=thread_region(thread.thread_id),
            content=thread.working_summary,
            support=support,
            processing_input_id="mr-thread-input:"
            + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            context={
                "source": "mr-thread",
                "open_question": thread.open_question,
                "thread_status": thread.status.value,
            },
        )

    def _with_previous_support(
        self, region_id: str, support: tuple[str, ...]
    ) -> tuple[str, ...]:
        head = self._baselines.get_head(region_id)
        if head is None:
            return support
        previous = tuple(
            memory.memory_id
            for memory in self._resolvable(head.supporting_memory_ids)
        )
        merged = tuple(dict.fromkeys((*previous, *support)))
        return merged[-MAX_HANDOFF_SUPPORT:]

    def _resolvable(self, memory_ids: tuple[str, ...]) -> tuple[CommittedMemory, ...]:
        found = []
        for memory_id in memory_ids:
            try:
                found.extend(self.substrate.memories((memory_id,)))
            except SupportSelectionError:
                continue
        return tuple(found)

    def lineage_head(self, question: str):
        """Accepted HEAD of the lineage a Thread or commit on ``question`` uses."""
        return self._baselines.get_head(thread_region(_thread_identity(self.scope, question)))

    def revise(
        self, *, question: str, summary: str, support: tuple[str, ...]
    ) -> str:
        """Continue an accepted lineage from its HEAD with new support (B1 + M4 -> B2).

        The summary is the understanding the semantic authority already formed
        for this block; no model is called. The previous HEAD's support is kept
        so the new revision still rests on the evidence it continues from.
        """
        require_non_empty(question, "question")
        require_non_empty(summary, "summary")
        region = thread_region(_thread_identity(self.scope, question))
        head = self._baselines.get_head(region)
        if head is None:
            raise ValueError("no accepted lineage to revise; use commit or a Thread")
        merged = self._with_previous_support(region, support)
        payload = json.dumps(
            {"region": region, "base": head.baseline_id, "support": merged, "summary": summary},
            sort_keys=True,
            ensure_ascii=False,
        )
        if head.content == summary and set(support) <= set(head.supporting_memory_ids):
            return head.baseline_id
        return self._promote(
            region_id=region,
            content=summary,
            support=merged,
            processing_input_id="mr-revise-input:"
            + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            context={
                "source": "mr-revise",
                "open_question": question,
                "previous_baseline_id": head.baseline_id,
            },
        )

    def commit(
        self, *, question: str, summary: str, support: tuple[str, ...]
    ) -> str:
        """Hand off cognition already complete at its first appearance.

        The lineage is the same ``mr-thread:<identity>`` region a Thread on the
        same question would use, so later Thread maturity revises this Baseline
        instead of creating a parallel lineage. Replays are idempotent through
        ``processing_input_id``.
        """
        require_non_empty(question, "question")
        require_non_empty(summary, "summary")
        region = thread_region(_thread_identity(self.scope, question))
        payload = json.dumps(
            {"region": region, "support": support, "summary": summary},
            sort_keys=True,
            ensure_ascii=False,
        )
        return self._promote(
            region_id=region,
            content=summary,
            support=support,
            processing_input_id="mr-commit-input:"
            + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            context={
                "source": "mr-commit",
                "open_question": question,
                "maturity_basis": "complete_at_first_appearance",
            },
        )

    # -- read side ------------------------------------------------------

    def accepted(self) -> tuple[AcceptedUnderstanding, ...]:
        """Every Baseline HEAD of this Scope's Path A lineages."""
        items: list[AcceptedUnderstanding] = []
        origins = {
            draft.merged_baseline_id: draft.processing_input_id or ""
            for draft in self._drafts.list(status="MERGED")
            if draft.merged_baseline_id
        }
        for region_id in self._baselines.list_regions():
            if not region_id.startswith(THREAD_REGION_PREFIX):
                continue
            head = self._baselines.get_head(region_id)
            if head is None:
                continue
            try:
                self.substrate.memories(head.supporting_memory_ids)
                complete = True
            except SupportSelectionError:
                complete = False
            items.append(
                AcceptedUnderstanding(
                    content=head.content,
                    baseline_id=head.baseline_id,
                    region_id=region_id,
                    revision_number=head.revision_number,
                    created_at=head.created_at,
                    origin=_origin(origins.get(head.baseline_id, "")),
                    supporting_memory_ids=head.supporting_memory_ids,
                    evidence_complete=complete,
                )
            )
        return tuple(items)


def _origin(processing_input_id: str) -> str:
    if processing_input_id.startswith("mr-commit-input:"):
        return "commit"
    if processing_input_id.startswith("mr-thread-input:"):
        return "thread"
    if processing_input_id.startswith("mr-revise-input:"):
        return "revision"
    return "unknown"
