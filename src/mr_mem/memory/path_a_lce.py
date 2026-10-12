"""LCE Path A adapter over native MR-Mem Canonical Memory; no Path B runtime.

Only immutable Memory IDs and native source references cross this boundary.
LCE owns derived Drafts and Baseline revisions, not raw/source authority.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from mr_mem.contracts import Scope
from mr_mem.memory.core import MemoryCore
from mr_mem.memory.path_a import MatureProjection
from mr_mem.memory.providers.bm25 import lexical_tokens
from mr_mem.memory.store import scope_json


def path_a_region(scope: Scope, thread_id: str) -> str:
    scope_id = hashlib.sha256(scope_json(scope).encode()).hexdigest()[:24]
    return f"mrmem-patha:{scope_id}:{thread_id}"


@dataclass(frozen=True, slots=True)
class ReadableBaseline:
    baseline_id: str
    revision_number: int
    content: str
    supporting_memory_ids: tuple[str, ...]
    source_refs: tuple[str, ...]


class _CanonicalLceSubstrate:
    """LCE external MemorySubstratePort; does not own or mirror raw text."""
    def __init__(self, core: MemoryCore, scope: Scope) -> None:
        self.core, self.scope = core, scope

    def get_by_ids(self, memory_ids: tuple[str, ...]):
        from lce.contracts.external_memory import MemoryItemView

        memories = self.core.select(
            memory_ids, scope=self.scope, active_only=True, max_items=256,
        )
        if any(not item.provenance.source_refs for item in memories):
            raise ValueError("LCE Path A requires native SourceRefs")
        return tuple(
            MemoryItemView(
                memory_id=item.memory_id,
                content=item.content,
                source_refs=tuple(
                    ref.source_key + "|" + ref.version_key
                    for ref in item.provenance.source_refs
                ),
            )
            for item in memories
        )


class LcePathAProjection:
    """LCE Draft/HEAD projection over native Memory, with zero Path B calls.

    Caller owns MemoryCore's lifetime; this storage root contains only LCE
    derived revisions. Canonical scope/lifecycle is checked again on read.
    """
    def __init__(self, core: MemoryCore, *, scope: Scope, root: str | Path) -> None:
        from lce.cognition.external import PrecomputedDraftIntake
        from lce.cognition.worktree import DraftRevisionStore
        from lce.store.sqlite_store import SqliteBaselineStore

        self.core, self.scope = core, scope
        self.baselines = SqliteBaselineStore(root)
        self.drafts = DraftRevisionStore(root)
        self.intake = PrecomputedDraftIntake(
            memory_substrate=_CanonicalLceSubstrate(core, scope),
            baseline_store=self.baselines,
            draft_store=self.drafts,
        )

    def publish(self, projection: MatureProjection) -> str:
        from lce.cognition.external import PrecomputedDraftInput

        if projection.scope != self.scope:
            raise ValueError("Path A projection scope mismatch")
        draft = PrecomputedDraftInput(
            region_id=path_a_region(self.scope, projection.thread_id),
            content=projection.understanding,
            supporting_memory_ids=projection.supporting_memory_ids,
            processing_input_id=projection.processing_input_id,
        )
        return self.intake.stage_and_promote(draft).baseline.baseline_id

    def search(self, query: str, *, limit: int = 10) -> tuple[ReadableBaseline, ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be 1..100")
        tokens = set(lexical_tokens(query))
        if not tokens:
            return ()
        prefix = path_a_region(self.scope, "")
        ranked: list[tuple[int, ReadableBaseline]] = []
        for region_id in self.baselines.list_regions():
            if not region_id.startswith(prefix):
                continue
            baseline = self.baselines.get_head(region_id)
            if baseline is None:
                continue
            try:
                memories = self.core.select(
                    baseline.supporting_memory_ids,
                    scope=self.scope, active_only=True, max_items=256,
                )
            except ValueError:
                continue  # stale or cross-scope support must not surface
            if any(not item.provenance.source_refs for item in memories):
                continue
            searchable = " ".join((baseline.content, *(item.content for item in memories)))
            matching = tokens & set(lexical_tokens(searchable))
            if not matching:
                continue
            refs = tuple(dict.fromkeys(
                ref.source_key + "|" + ref.version_key
                for item in memories for ref in item.provenance.source_refs
            ))
            ranked.append((len(matching), ReadableBaseline(
                baseline.baseline_id, baseline.revision_number,
                baseline.content, baseline.supporting_memory_ids, refs,
            )))
        ranked.sort(key=lambda pair: (-pair[0], pair[1].baseline_id))
        return tuple(view for _, view in ranked[:limit])

    def close(self) -> None:
        self.baselines.close()
        close = getattr(self.drafts, "close", None)
        if close is not None:
            close()
