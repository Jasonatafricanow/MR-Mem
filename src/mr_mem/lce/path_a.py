"""LCE Thread + Path A over one Scope: compiler signals -> branches and committed steps.

Design v6 4.4-4.5 (docs/LCE_PORT.md):

- A Thread is uncommitted work in progress: a matter the user is living
  through, with its open steps and an index (``anchor_node_id``) to the
  committed step it grows from. It is MR-Mem's existing ``MemoryThread``.
- Path A is what has been committed: Line trees of SemanticBlocks
  (``line_graph``). There is no main line: every branch is its own direction,
  branches never merge back, and trees of different kinds meet where they
  share a Block.
- Path A is explicit work. Retrieval puts the relevant threads and steps in
  front of the compiler (``known``); the compiler decides which matter a Block
  belongs to, which step it follows and whether it has happened. Nothing here
  decides attachment by similarity (that is Path B).

Every write uses the message time of the batch, so replay is deterministic.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from mr_mem.contracts import Scope
from mr_mem.integrations.path_a_lines import LineCard, select_known_lines
from mr_mem.lce.line_graph import LineGraphStore, LineGraphView, StateView
from mr_mem.lce.rejection import RejectionStore
from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.product import MemoryProductStore, MemoryThread, ThreadStatus
from mr_mem.memory.store import CanonicalMemoryStore, scope_json
from mr_mem.memory.threading import _thread_identity

EvidenceTime = Callable[[CommittedMemory], "datetime | None"]
Embed = Callable[[str], Sequence[float]]


@dataclass(frozen=True, slots=True)
class Attachment:
    """One compiler signal: this Block's place in one matter."""

    thread: str | None  # "T<k>" a known thread, "b<k>" the thread block k opened, None = new
    after: str | None
    committed: bool
    summary: str | None
    done: bool = False  # ignored: lines never end (design v6 4.4); kept for old journals


@dataclass(frozen=True, slots=True)
class BlockLines:
    memory_id: str
    lines: tuple[Attachment, ...]


@dataclass(frozen=True, slots=True)
class Known:
    """What the compiler sees, and how its refs resolve (journaled with the compile)."""

    payload: dict
    refs: dict[str, str] = field(default_factory=dict)  # "T1" -> thread_id, "N1" -> node_id


@dataclass
class ApplyResult:
    committed_nodes: int = 0
    opened_threads: int = 0
    opened: list[str] = field(default_factory=list)  # thread ids, in order
    failures: list[str] = field(default_factory=list)


def scope_root(root: Path | str, scope: Scope) -> Path:
    return Path(root) / hashlib.sha256(scope_json(scope).encode("utf-8")).hexdigest()


class PathATree:
    """Thread + Path A for one Scope. Not thread-safe: the runtime owner thread uses it."""

    def __init__(
        self,
        root: Path | str,
        scope: Scope,
        *,
        canonical: CanonicalMemoryStore,
        products: MemoryProductStore,
        fingerprint: str,
        evidence_time: EvidenceTime | None = None,
    ) -> None:
        self.scope = scope
        self.canonical = canonical
        self.products = products
        base = scope_root(root, scope)
        self.lines = LineGraphStore(base / "lines.sqlite")
        self.lines.set_derivation_fingerprint(fingerprint)
        self.rejections = RejectionStore(base / "rejections.sqlite")
        self.view = LineGraphView(store=self.lines, read_state=self._state)
        self.evidence_time = evidence_time or (lambda memory: memory.occurred_at)

    def close(self) -> None:
        self.lines.close()
        self.rejections.close()

    # -- canonical reads ---------------------------------------------------

    def _state(self, state_id: str) -> StateView | None:
        memory = self.canonical.get(state_id)
        if memory is None:
            return None
        return StateView(
            state_id, memory.content, memory.occurred_at,
            tuple(ref.source_key for ref in memory.provenance.source_refs),
            memory.lifecycle is MemoryLifecycle.ACTIVE,
        )

    def open_threads(self) -> tuple[MemoryThread, ...]:
        return self.products.list_threads(self.scope, status=ThreadStatus.OPEN)

    def committed_nodes(self, cutoff: datetime) -> list[tuple[str, str, StateView]]:
        """(line_id, node_id, state) for every visible committed step."""
        out = []
        for line in self.lines.list_lines():
            for node_id in self.view.visible_node_ids(line.line_id, cutoff=cutoff):
                state = self.view.state_for_node_at_cutoff(node_id, cutoff=cutoff)
                if state is not None:
                    out.append((line.line_id, node_id, state))
        return out

    # -- compile input -----------------------------------------------------

    def known(
        self,
        *,
        batch_text: str,
        now: datetime,
        threads_policy: dict[str, int],
        steps_policy: dict[str, int],
        embed: Embed | None = None,
        snippet: int = 400,
    ) -> Known:
        """Threads and committed steps relevant to this batch, with refs for the compiler."""
        refs: dict[str, str] = {}
        threads = {t.thread_id: t for t in self.open_threads()}
        thread_cards = []
        for t in threads.values():
            # a thread has no name: what it is about lives in its summary (design v6 4.5)
            anchor = self._anchor_text(t, now)
            thread_cards.append(LineCard(
                t.thread_id, t.working_summary, False, t.updated_at,
                "\n".join(x for x in (t.working_summary or "", anchor) if x),
            ))
        chosen_threads = select_known_lines(
            thread_cards, batch_text=batch_text, now=now, embed=embed, **threads_policy,
        ).lines
        steps = self.committed_nodes(now)
        step_cards = [
            LineCard(node_id, None, True, state.occurred_at, state.content)
            for _, node_id, state in steps
        ]
        chosen_steps = [c.question for c in select_known_lines(
            step_cards, batch_text=batch_text, now=now, embed=embed, due=0, **steps_policy,
        ).lines]
        # every chosen thread's branch point is shown, so "after" can name it
        for card in chosen_threads:
            anchor = threads[card.question].anchor_node_id
            if anchor and anchor not in chosen_steps and any(n == anchor for _, n, _ in steps):
                chosen_steps.append(anchor)
        state_of = {node_id: state for _, node_id, state in steps}
        line_of = {node_id: line_id for line_id, node_id, _ in steps}
        step_ref = {}
        payload_steps = []
        for node_id in sorted(chosen_steps, key=lambda n: (state_of[n].occurred_at, n)):
            ref = f"N{len(step_ref) + 1}"
            step_ref[node_id], refs[ref] = ref, node_id
            state = state_of[node_id]
            memory = self.canonical.get(state.state_id)
            known_time = None if memory is None else self.evidence_time(memory)
            payload_steps.append({
                "ref": ref, "text": state.content[:snippet],
                **({"time": known_time.strftime("%Y-%m-%d %H:%M")} if known_time else {}),
            })
        payload_threads = []
        for i, card in enumerate(chosen_threads, 1):
            t = threads[card.question]
            refs[f"T{i}"] = t.thread_id
            entry = {"ref": f"T{i}", "summary": t.working_summary or t.open_question}
            if t.anchor_node_id in step_ref:
                entry["after"] = step_ref[t.anchor_node_id]
            payload_threads.append(entry)
        rejected = self.rejections.relevant(
            region_ids=tuple({line_of[n] for n in chosen_steps if n in line_of}
                             | {t.thread_id for t in threads.values()}),
        )
        return Known(
            {"threads": payload_threads, "steps": payload_steps,
             "rejected": [r.rejected_content for r in rejected]},
            refs,
        )

    def _anchor_text(self, thread: MemoryThread, now: datetime) -> str:
        if thread.anchor_node_id is None:
            return ""
        try:
            state = self.view.state_for_node_at_cutoff(thread.anchor_node_id, cutoff=now)
        except KeyError:
            return ""
        return "" if state is None else state.content

    # -- apply compiler signals ------------------------------------------

    def apply(
        self, blocks: Sequence[BlockLines], refs: dict[str, str], at: datetime
    ) -> ApplyResult:
        """Commit happened steps onto their branch; keep open steps on their Thread."""
        result = ApplyResult()
        nodes_of_block: dict[int, list[str]] = {}
        opened_by_block: dict[int, list[str]] = {}
        for index, block in enumerate(blocks, 1):
            memory = self.canonical.get(block.memory_id)
            if memory is None:
                result.failures.append(f"block {block.memory_id} not in canonical Memory")
                continue
            for k, line in enumerate(block.lines):
                try:
                    node = self._apply_one(line, memory, refs, nodes_of_block, opened_by_block,
                                           index, k, at, result)
                except ValueError as exc:  # a rejected relation never loses the Block
                    result.failures.append(f"{block.memory_id}: {exc}")
                    continue
                if node is not None:
                    nodes_of_block.setdefault(index, []).append(node)
        return result

    def _thread_for(
        self, ref: str | None, refs: dict[str, str], opened_by_block: dict[int, list[str]]
    ) -> MemoryThread | None:
        """Identity is the code id behind a ref, never a name or wording."""
        thread_id = None
        if ref and ref.startswith("T"):
            thread_id = refs.get(ref)
        elif ref and ref.startswith("b") and ref[1:].isdigit():
            opened = opened_by_block.get(int(ref[1:]), [])
            thread_id = opened[0] if opened else None
        thread = None if thread_id is None else self.products.get_thread(thread_id)
        return thread if thread is not None and thread.status is ThreadStatus.OPEN else None

    def _parent(
        self, after: str | None, thread: MemoryThread | None, refs: dict[str, str],
        nodes_of_block: dict[int, list[str]], at: datetime,
    ) -> str | None:
        node_id = None
        if after and after.startswith("N"):
            node_id = refs.get(after)
        elif after and after.startswith("T"):
            other = self.products.get_thread(refs[after]) if after in refs else None
            node_id = None if other is None else other.anchor_node_id
        elif after and after.startswith("b") and after[1:].isdigit():
            created = nodes_of_block.get(int(after[1:]), [])
            node_id = created[0] if created else None
            if node_id is None and thread is not None:
                node_id = thread.anchor_node_id
        if node_id is None and after is None and thread is not None:
            node_id = thread.anchor_node_id
        if node_id is not None:
            try:
                self.lines.get_node(node_id)
            except KeyError:
                return None
        return node_id

    def _apply_one(
        self, line: Attachment, memory: CommittedMemory, refs: dict[str, str],
        nodes_of_block: dict[int, list[str]], opened_by_block: dict[int, list[str]],
        index: int, k: int, at: datetime, result: ApplyResult,
    ) -> str | None:
        thread = self._thread_for(line.thread, refs, opened_by_block)
        parent = self._parent(line.after, thread, refs, nodes_of_block, at)
        node_id = None
        anchor = thread.anchor_node_id if thread is not None else None
        if line.committed:
            if parent is not None:
                line_id = self.lines.get_node(parent).line_id
            else:
                line_id = self.lines.create_line((memory.memory_id,), created_at=at).line_id
            node, _, _ = self.lines.ensure_node(
                line_id, block_id=memory.memory_id, state_id=memory.memory_id,
                occurred_start=memory.occurred_at, occurred_end=memory.occurred_at,
                knowledge_at=at,
            )
            node_id = node.node_id
            if parent is not None and parent != node_id:
                self.lines.add_edge(line_id, parent, node_id, knowledge_at=at)
            result.committed_nodes += 1
            anchor = node_id
        elif parent is not None:
            anchor = parent
        # The Thread holds the open (uncommitted) steps; a commit moves its branch
        # point and clears them. With nothing open, the newest commit stands in
        # as the Thread's support (a Thread always rests on canonical Memory).
        pending = () if thread is None or line.committed else tuple(
            mid for mid in thread.current_support_ids if not self.lines.nodes_for_state(mid)
        )
        support = tuple(dict.fromkeys((*pending, memory.memory_id)))[-8:]
        when = at if thread is None else max(at, thread.updated_at)
        if thread is None:
            # a new thread: its id comes from the Block that opened it, never from words
            thread_id = _thread_identity(self.scope, f"{memory.memory_id}\n{k}")
            label = (line.summary or memory.content)[:2048]  # display only, not identity
            thread = self.products.open_thread(
                thread_id=thread_id, scope=self.scope, open_question=label,
                supporting_memory_ids=support, at=at, working_summary=line.summary,
                anchor_node_id=anchor,
            )
            opened_by_block.setdefault(index, []).append(thread.thread_id)
            result.opened.append(thread.thread_id)
            result.opened_threads += 1
        else:
            thread = self.products.update_thread(
                thread.thread_id, supporting_memory_ids=support, at=when,
                working_summary=line.summary, anchor_node_id=anchor,
            )
        return node_id
