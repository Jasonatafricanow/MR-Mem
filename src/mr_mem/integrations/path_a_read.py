"""Unified outward read over canonical Memory, active Threads and Path A cognition.

Port of Mind-Runtime ``memory/retrieval_composition.py`` (ADR-0033 "Read
authority"; MEMORY_ARCHITECTURE_V1 section 9). Consumers read one ordered
evidence list instead of joining Memory, Thread and LCE stores themselves:

    accepted Baseline cognition  (+ the canonical support it rests on,
                                   + the original source text of that support)
    -> active relevant Thread     (+ its support)
    -> canonical Memory detail from the normal retrieval stack

``mode="baseline"`` returns only the canonical retrieval part, rendered the
same way, so the two modes differ only in whether projections are read.

A host may instead supply its own raw-evidence retrieval (``raw_search``): the
base list is then the host's raw hits, and projections are reached upward from
them (raw record -> SemanticBlocks compiled from it -> their Thread / LCE line),
the reverse of the projection -> evidence link used for verification. A
projection whose support is reached from the raw hits AND whose own text is
close to the query is read first (two independent routes agree).
With ``projection_shape="matter"`` every selected Thread or cognition line is
read as ONE item: the matter, its steps in order with their Block content, and
its current state (governance design v5, 5.2-5.3). Its support then no longer
appears as separate items. Times are shown only where the source gave a message
time (``evidence_time``); otherwise only the order is stated.
The ranking of projections is deliberately simple and replaceable; it was
fixed before any evaluation run (docs/PATH_A.md) and is not benchmark-tuned.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from mr_mem.contracts import Scope
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.product import MemoryProductStore, MemoryThread, ThreadStatus
from mr_mem.memory.retrieval import (
    MemoryRetrievalQuery,
    MemoryRetrievalService,
    MemorySurfaceBudget,
)
from mr_mem.memory.source import SourceRef
from mr_mem.memory.store import CanonicalMemoryStore

THREAD_REGION_PREFIX = "mr-thread:"  # same lineage prefix as integrations.lce_path_a

SearchMode = Literal["baseline", "path_a"]
RenderSource = Callable[[SourceRef], "str | None"]
Embed = Callable[[str], Sequence[float]]


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    item_id: str
    kind: str  # lce | thread | support | source | memory
    content: str
    score: float
    occurred_at: datetime | None
    parent_id: str | None = None


@dataclass(frozen=True, slots=True)
class RawHit:
    """One host raw-evidence hit and the source records it came from."""

    item_id: str
    content: str
    score: float
    occurred_at: datetime | None
    record_ids: tuple[str, ...]


RawSearch = Callable[[Scope, str, int], Sequence[RawHit]]
BlocksForRecord = Callable[[str], Sequence[str]]


@dataclass(frozen=True, slots=True)
class ReadPolicy:
    lce_limit: int = 5
    thread_limit: int = 2
    support_limit: int = 8
    source_limit: int = 8
    include_source_text: bool = True
    # Attach original source text to plain retrieval hits too, in BOTH modes, so
    # an evaluation can separate "LCE value" from "raw evidence value".
    source_text_for_hits: bool = False
    support_rank_window: int = 10
    min_cosine: float = 0.60
    # Prefix raw hits with their message time ("[YYYY-MM-DD HH:MM] ..."), both modes.
    stamp_raw_hits: bool = False
    # "flat": projection line, then its support as separate items;
    # "matter": one item per projection with its steps inside.
    projection_shape: str = "flat"
    max_characters: int = 65536

    def __post_init__(self) -> None:
        if type(self.max_characters) is not int or not 0 <= self.max_characters <= 65536:
            raise ValueError("max_characters must be in [0, 65536]")
        if self.projection_shape not in ("flat", "matter"):
            raise ValueError("projection_shape must be 'flat' or 'matter'")
        for name in (
            "lce_limit", "thread_limit", "support_limit", "source_limit", "support_rank_window",
        ):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 100:
                raise ValueError(f"{name} must be an integer in [0, 100]")


def _stamp(at: datetime | None) -> str:
    return at.strftime("%Y-%m-%d %H:%M") if at is not None else "unknown time"


def render_memory(memory: CommittedMemory, at: datetime | None = None, *,
                  known: bool = True) -> str:
    # Support whose original evidence is gone stays readable, but is marked
    # (raw text is authority; compiled products persist, greyed, never hidden).
    active = memory.lifecycle is MemoryLifecycle.ACTIVE
    marker = "" if active else " | original evidence unavailable"
    if not known:  # the source gave no message time: say nothing about time
        return f"[{marker[3:]}] {memory.content}" if marker else memory.content
    return f"[{_stamp(at or memory.occurred_at)}{marker}] {memory.content}"


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norm if norm else 0.0


class PathAReader:
    def __init__(
        self,
        *,
        canonical_path: Path | str,
        retrieval: MemoryRetrievalService,
        lce=None,
        policy: ReadPolicy | None = None,
        render_source: RenderSource | None = None,
        embed: Embed | None = None,
        raw_search: RawSearch | None = None,
        blocks_for_record: BlocksForRecord | None = None,
        evidence_time: Callable[[CommittedMemory], datetime | None] | None = None,
        line_context: Callable | None = None,
    ) -> None:
        self.canonical_path = Path(canonical_path)
        self.retrieval = retrieval
        self.lce = lce
        self.line_context = line_context
        self.policy = policy or ReadPolicy()
        self.render_source = render_source
        self.embed = embed
        self.raw_search = raw_search
        self.blocks_for_record = blocks_for_record
        self._embedding_cache: dict[str, tuple[float, ...]] = {}
        # Message time the source actually gave, or None; ordering always uses
        # occurred_at (arrival order when the source gave no time).
        self.evidence_time = evidence_time or (lambda memory: memory.occurred_at)

    def _render(self, memory: CommittedMemory) -> str:
        at = self.evidence_time(memory)
        return render_memory(memory, at, known=at is not None)

    def _known_latest(self, memories) -> datetime | None:
        times = [t for m in memories if (t := self.evidence_time(m)) is not None]
        return max(times) if times else None

    def _matter(self, head: str, memories, tail: str = "") -> str:
        """One matter as one text: header, steps in order, then current state."""
        steps = sorted(memories, key=lambda m: (m.occurred_at, m.memory_id))
        lines = [head, "Steps in order:"]
        lines += [f"{i}. {self._render(m)}" for i, m in enumerate(steps, 1)]
        if tail:
            lines.append(tail)
        return "\n".join(lines)

    def _embedded(self, text: str) -> tuple[float, ...]:
        cached = self._embedding_cache.get(text)
        if cached is None:
            assert self.embed is not None
            cached = tuple(self.embed(text))
            self._embedding_cache[text] = cached
        return cached

    def _select(
        self,
        query: str,
        candidates: dict[str, tuple[str, Sequence[str]]],
        rank_of: dict[str, int],
        limit: int,
    ) -> list[tuple[str, float]]:
        """Pick projections whose own factual support is relevant to the query.

        A projection qualifies when one of its supporting canonical Memories is
        within the top ``support_rank_window`` of the normal retrieval result
        (the same stack baseline mode uses), or - with an embedding configured -
        when its own text is close to the query. Order: best support rank, then
        dense similarity. Nothing here was tuned on evaluation data.
        """
        if not candidates or limit == 0:
            return []
        window = self.policy.support_rank_window
        query_vector = self._embedded(query) if self.embed is not None else None
        scored: list[tuple[int, int, float, str]] = []
        for key, (text, support) in candidates.items():
            best = min((rank_of[mid] for mid in support if mid in rank_of), default=10**9)
            cosine = _cosine(query_vector, self._embedded(text)) if query_vector else 0.0
            by_support = best <= window
            by_text = query_vector is not None and cosine >= self.policy.min_cosine
            if by_support or by_text:
                # both routes agreeing outranks either one alone
                scored.append((0 if by_support and by_text else 1, best, -cosine, key))
        scored.sort()
        return [(key, 1.0 / (60 + min(best, 10**6))) for _, best, _, key in scored[:limit]]

    def _support_items(
        self,
        store: CanonicalMemoryStore,
        parent: str,
        memory_ids: Sequence[str],
        rank_of: dict[str, int],
        seen_memory: set[str],
        seen_source: set[str],
        score: float,
        *,
        trajectory: bool = False,
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        refs: list[SourceRef] = []
        found = [m for mid in dict.fromkeys(memory_ids) if (m := store.get(mid)) is not None]
        # Evidence relevant to this query first, then the newest: a revised
        # lineage accumulates support and its oldest rows are rarely the point.
        found.sort(key=lambda m: (
            rank_of.get(m.memory_id, 10**9),
            -(m.occurred_at.timestamp() if m.occurred_at else 0.0),
            m.memory_id,
        ))
        found = found[: self.policy.support_limit]
        if trajectory:  # an ongoing matter reads as its steps in time order
            found.sort(key=lambda m: (m.occurred_at.timestamp() if m.occurred_at else 0.0,
                                      m.memory_id))
        for memory in found:
            memory_id = memory.memory_id
            refs.extend(memory.provenance.source_refs)
            if memory_id in seen_memory:
                continue
            seen_memory.add(memory_id)
            items.append(EvidenceItem(
                f"memory:{memory_id}", "support", self._render(memory), score,
                memory.occurred_at, parent,
            ))
        if self.policy.include_source_text:
            items.extend(self._source_items(parent, refs, seen_source, score))
        return items

    def _memories(self, store, memory_ids, seen_memory: set[str]) -> list[CommittedMemory]:
        found = [m for mid in dict.fromkeys(memory_ids) if (m := store.get(mid)) is not None]
        found = sorted(found, key=lambda m: (m.occurred_at, m.memory_id))
        found = found[-self.policy.support_limit:] if self.policy.support_limit else []
        seen_memory.update(m.memory_id for m in found)
        return found

    def _source_items(
        self, parent: str, refs: Sequence[SourceRef], seen_source: set[str], score: float
    ) -> list[EvidenceItem]:
        if self.render_source is None:
            return []
        items: list[EvidenceItem] = []
        for ref in sorted(refs, key=lambda r: (r.occurred_at, r.source_key)):
            if len(items) >= self.policy.source_limit or ref.source_key in seen_source:
                continue
            text = self.render_source(ref)
            if not text:
                continue
            seen_source.add(ref.source_key)
            items.append(EvidenceItem(
                f"source:{ref.version_key}", "source", text, score, ref.occurred_at, parent,
            ))
        return items

    def search(
        self, scope: Scope, query: str, *, limit: int, mode: SearchMode = "path_a"
    ) -> tuple[EvidenceItem, ...]:
        if mode not in ("baseline", "path_a"):
            raise ValueError("mode must be 'baseline' or 'path_a'")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer in [1, 100]")
        items: list[EvidenceItem] = []
        seen_memory: set[str] = set()
        seen_source: set[str] = set()
        store = CanonicalMemoryStore(self.canonical_path, read_only=True)
        try:
            if self.raw_search is not None:
                self._raw_read(scope, query, limit, mode, store, items, seen_memory, seen_source)
                return self._bounded(items, limit)
            hits = self.retrieval.search(
                MemoryRetrievalQuery(scope=scope, text=query, limit=limit),
                budget=MemorySurfaceBudget(max_items=limit, max_characters=65536),
            )
            if mode == "path_a":
                rank_of = {hit.memory.memory_id: position for position, hit in enumerate(hits, 1)}
                items.extend(self._projection_items(
                    scope, query, store, rank_of, seen_memory, seen_source,
                ))
            for hit in hits:
                memory = hit.memory
                if memory.memory_id in seen_memory:
                    continue
                seen_memory.add(memory.memory_id)
                score = float(hit.candidate.score or 0.0)
                items.append(EvidenceItem(
                    f"memory:{memory.memory_id}", "memory", self._render(memory),
                    score, memory.occurred_at,
                ))
                if self.policy.source_text_for_hits:
                    items.extend(self._source_items(
                        f"memory:{memory.memory_id}", memory.provenance.source_refs,
                        seen_source, score,
                    ))
        finally:
            store.close()
        return self._bounded(items, limit)

    def _bounded(self, items, limit):
        result, remaining = [], self.policy.max_characters
        for item in items:
            if len(item.content) > remaining:
                continue
            result.append(item)
            remaining -= len(item.content)
            if len(result) >= limit:
                break
        return tuple(result)

    def _raw_read(self, scope, query, limit, mode, store, items, seen_memory, seen_source) -> None:
        raw = list(self.raw_search(scope, query, limit))
        if mode == "path_a":
            # upward: a Block's rank is the best rank of a raw hit it was compiled from
            rank_of: dict[str, int] = {}
            if self.blocks_for_record is not None:
                for position, hit in enumerate(raw, 1):
                    for record_id in hit.record_ids:
                        for memory_id in self.blocks_for_record(record_id):
                            rank_of.setdefault(memory_id, position)
            items.extend(self._projection_items(
                scope, query, store, rank_of, seen_memory, seen_source,
            ))
        for hit in raw:
            content = (f"[{_stamp(hit.occurred_at)}] {hit.content}"
                       if self.policy.stamp_raw_hits and hit.occurred_at is not None
                       else hit.content)
            items.append(EvidenceItem(hit.item_id, "raw", content, hit.score, hit.occurred_at))

    def _projection_items(
        self,
        scope: Scope,
        query: str,
        store: CanonicalMemoryStore,
        rank_of: dict[str, int],
        seen_memory: set[str],
        seen_source: set[str],
    ) -> list[EvidenceItem]:
        items: list[EvidenceItem] = []
        if self.line_context is not None:
            for group in self.line_context(scope, rank_of, store):
                seen_memory.update(group.memory_ids)
                items.append(EvidenceItem(
                    group.item_id, "line", group.content, group.score, None,
                ))
            return items
        heads = tuple(self.lce.accepted()) if self.lce is not None else ()

        def newest_evidence(memory_ids) -> datetime | None:
            times = [m.occurred_at for mid in memory_ids
                     if (m := store.get(mid)) is not None and m.occurred_at is not None]
            return max(times) if times else None

        # Compare by evidence (message) time on both sides: LCE records its own
        # acceptance clock, Threads the host's activity time; neither is comparable.
        head_at = {u.region_id: newest_evidence(u.supporting_memory_ids) for u in heads}
        if heads and self.policy.lce_limit:
            accepted = {u.baseline_id: u for u in heads}
            candidates = {
                key: (u.content, u.supporting_memory_ids) for key, u in accepted.items()
            }
            for key, score in self._select(query, candidates, rank_of, self.policy.lce_limit):
                u = accepted[key]
                occurred = [
                    m.occurred_at for mid in u.supporting_memory_ids
                    if (m := store.get(mid)) is not None and m.occurred_at is not None
                ]
                as_of = max(occurred) if occurred else None
                marker = "" if u.evidence_complete else " | original evidence unavailable"
                if self.policy.projection_shape == "matter":
                    support = self._memories(store, u.supporting_memory_ids, seen_memory)
                    known = self._known_latest(support)
                    head = (f"[Accepted understanding | revision {u.revision_number}"
                            + (f" | as of {_stamp(known)}" if known else "")
                            + f"{marker}] {u.content}")
                    earlier = (self.lce.earlier_revisions(u.region_id)
                               if u.revision_number > 1 and hasattr(self.lce, "earlier_revisions")
                               else ())
                    if earlier:
                        head += "\nEarlier understanding: " + " -> ".join(earlier)
                    items.append(EvidenceItem(
                        f"lce:{u.baseline_id}", "lce", self._matter(head, support), score, known,
                    ))
                    continue
                items.append(EvidenceItem(
                    f"lce:{u.baseline_id}", "lce",
                    f"[Accepted understanding | revision {u.revision_number} | as of "
                    f"{_stamp(as_of)}{marker}] {u.content}",
                    score, as_of,
                ))
                items.extend(self._support_items(
                    store, f"lce:{u.baseline_id}", u.supporting_memory_ids, rank_of,
                    seen_memory, seen_source, score,
                ))
        if self.policy.thread_limit:
            product = MemoryProductStore(self.canonical_path, store, read_only=True)
            try:
                # A Thread older than its lineage's accepted HEAD was superseded
                # by that higher projection; never serve both (ADR-0033).
                threads: dict[str, MemoryThread] = {
                    t.thread_id: t
                    for t in product.list_threads(
                        scope, status=ThreadStatus.OPEN, include_suppressed=False
                    )
                    if not (
                        (at := head_at.get(THREAD_REGION_PREFIX + t.thread_id)) is not None
                        and (mine := newest_evidence(t.handoff_memory_ids)) is not None
                        and at >= mine
                    )
                }
            finally:
                product.close()
            candidates = {
                key: (
                    " ".join(v for v in (t.open_question, t.working_summary) if v),
                    t.handoff_memory_ids,
                )
                for key, t in threads.items()
            }
            for key, score in self._select(query, candidates, rank_of, self.policy.thread_limit):
                t = threads[key]
                summary = f" | So far: {t.working_summary}" if t.working_summary else ""
                latest = newest_evidence(t.handoff_memory_ids)
                if self.policy.projection_shape == "matter":
                    support = self._memories(store, t.handoff_memory_ids, seen_memory)
                    known = self._known_latest(support)
                    head = ("[Ongoing matter, not concluded"
                            + (f" | last update {_stamp(known)}" if known else "")
                            + f"] {t.open_question}")
                    tail = f"Current state: {t.working_summary}" if t.working_summary else ""
                    items.append(EvidenceItem(
                        f"thread:{t.thread_id}", "thread", self._matter(head, support, tail),
                        score, known,
                    ))
                    continue
                items.append(EvidenceItem(
                    f"thread:{t.thread_id}", "thread",
                    f"[Ongoing matter, not concluded | last update {_stamp(latest)}] "
                    f"{t.open_question}{summary}",
                    score, latest,
                ))
                items.extend(self._support_items(
                    store, f"thread:{t.thread_id}", t.handoff_memory_ids, rank_of,
                    seen_memory, seen_source, score, trajectory=True,
                ))
        return items
