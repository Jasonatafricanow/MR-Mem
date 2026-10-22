"""Read bounded, source-backed progression neighbourhoods without opening a writer."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from mr_mem.contracts import Scope
from mr_mem.lce.line_graph import LineGraphStore, LineGraphView, StateView, _require_utc
from mr_mem.lce.path_a import TenseStore, scope_root
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.retrieval import MemorySurfaceBudget
from mr_mem.memory.store import CanonicalMemoryStore


@dataclass(frozen=True, slots=True)
class LineContext:
    item_id: str
    content: str
    memory_ids: tuple[str, ...]
    source_keys: tuple[str, ...]
    score: float


class LineContextReader:
    def __init__(self, root: Path | str, *, max_nodes: int = 8, max_hops: int = 2):
        self.root = Path(root)
        if type(max_nodes) is not int or not 1 <= max_nodes <= 100:
            raise ValueError("max_nodes must be in [1, 100]")
        if type(max_hops) is not int or not 0 <= max_hops <= 10:
            raise ValueError("max_hops must be in [0, 10]")
        self.max_nodes, self.max_hops = max_nodes, max_hops

    def read(
        self, *, scope: Scope, ranks: Mapping[str, int], store: CanonicalMemoryStore,
        knowledge_at: datetime, now: datetime, budget: MemorySurfaceBudget,
        allowed: Callable[[CommittedMemory], bool] = lambda memory: True,
        evidence_time: Callable[[CommittedMemory], datetime | None] = lambda m: m.occurred_at,
    ) -> tuple[LineContext, ...]:
        _require_utc(knowledge_at, "knowledge_at")
        _require_utc(now, "now")
        if not ranks or not budget.max_items or not budget.max_characters:
            return ()
        base = scope_root(self.root, scope)
        graph_path, tense_path = base / "lines.sqlite", base / "tense.sqlite"
        if not graph_path.exists() and not tense_path.exists():
            return ()
        # A partial/corrupt pair must fail, rather than silently lose relations.
        graph = LineGraphStore(graph_path, read_only=True)
        try:
            tense = TenseStore(tense_path, read_only=True)
            try:
                def memory(mid):
                    m = store.get(mid)
                    return m if (m is not None and m.scope == scope
                                 and m.known_at <= knowledge_at and allowed(m)) else None

                def state(mid):
                    m = store.get(mid)
                    if m is not None and (m.scope != scope or m.known_at > knowledge_at):
                        m = None
                    return None if m is None else StateView(
                        mid, m.content, m.occurred_at,
                        tuple(ref.source_key for ref in m.provenance.source_refs),
                        m.lifecycle is MemoryLifecycle.ACTIVE,
                    )

                view = LineGraphView(store=graph, read_state=state)
                return self._read(
                    graph, view, tense, memory, ranks, knowledge_at, now, budget, evidence_time,
                )
            finally:
                tense.close()
        finally:
            graph.close()

    def _read(self, graph, view, tense, memory, ranks, cutoff, now, budget, evidence_time):
        result, covered = [], set()
        remaining = budget.max_characters

        def visible(node_id):
            if not graph.membership_active_at(node_id, cutoff):
                return None
            s = view.state_for_node_at_cutoff(node_id, cutoff=cutoff)
            return s if s is not None and memory(s.state_id) is not None else None

        def neighbours(node_id):
            return sorted(set(graph.parents_at(node_id, cutoff))
                          | set(graph.children_at(node_id, cutoff)))

        for mid, rank in sorted(ranks.items(), key=lambda pair: (pair[1], pair[0])):
            for seed in graph.nodes_for_state(mid):
                seed_state = visible(seed.node_id)
                # Do not resurrect an older state through a stale retrieval hit.
                if seed.node_id in covered or seed_state is None or seed_state.state_id != mid:
                    continue
                picked, queue, visited = {}, [(seed.node_id, 0)], set()
                partial = False
                while queue:
                    node_id, depth = queue.pop(0)
                    if node_id in visited:
                        continue
                    visited.add(node_id)
                    s = visible(node_id)
                    if s is None:
                        partial = True
                        continue
                    if len(picked) >= self.max_nodes:
                        partial = True
                        break
                    picked[node_id] = s
                    adjacent = [n for n in neighbours(node_id) if n not in visited]
                    if depth < self.max_hops:
                        queue.extend((n, depth + 1) for n in adjacent)
                    elif adjacent:
                        partial = True
                # Keep whole Blocks. Drop the most distant selected step first;
                # rerender edges so no dangling endpoint or invented shortcut remains.
                while picked:
                    content, mids, refs = self._render(
                        graph, tense, memory, picked, cutoff, now, partial, evidence_time,
                    )
                    if len(content) <= remaining:
                        result.append(LineContext(
                            f"line:{seed.node_id}", content, mids, refs, 1 / max(1, rank),
                        ))
                        covered.update(picked)
                        remaining -= len(content)
                        break
                    picked.pop(next(reversed(picked)))
                    partial = True
                if len(result) >= budget.max_items:
                    return tuple(result)
        return tuple(result)

    @staticmethod
    def _render(graph, tense, memory, picked, cutoff, now, partial, evidence_time):
        ordered = sorted(picked, key=lambda n: (picked[n].occurred_at, n))
        labels = {n: f"S{i}" for i, n in enumerate(ordered, 1)}
        lines = ["[Line neighbourhood | " + ("partial" if partial else "local") + "]"]
        mids, refs = [], set()
        for node_id in ordered:
            s = picked[node_id]
            m = memory(s.state_id)
            side = tense.side(
                s.state_id, now, s.content, knowledge_at=cutoff,
                allow_settler=lambda mid: memory(mid) is not None,
            )
            at = evidence_time(m)
            marker = "" if s.evidence_complete else " | evidence unavailable or inactive"
            stamp = "" if at is None else f" | message {at.isoformat()}"
            lines.append(f"{labels[node_id]} [{side or 'unknown tense'}{stamp}{marker}] "
                         f"{s.content}")
            mids.append(s.state_id)
            refs.update(s.source_keys)
        edges = [f"{labels[p]} -> {labels[n]}" for n in ordered
                 for p in graph.parents_at(n, cutoff) if p in picked]
        if edges:
            lines.append("Recorded progression: " + "; ".join(edges))
        return "\n".join(lines), tuple(mids), tuple(sorted(refs))
