"""LCE over one Scope: compiler signals -> Lines whose SemanticBlocks carry a tense.

Design v6 4.4-4.5 (docs/LCE_PORT.md):

- A Line is a git-like tree of SemanticBlocks that belong together by meaning
  and logic. A Block that connects to nothing stays alone; nothing is forced
  onto a line. There is no main line, and lines never end: any step can grow
  further branches, and trees of different kinds meet where they share a Block.
- Every Block on a line is on it from the moment it is said. Thread and Path A
  are not containers: they are the two sides of a terminator that moves with
  time. Past side (Path A): it happened, or its stated time has passed (what is
  frozen then is "this was the plan", whether or not it was carried out), or a
  later message settled it. Day side (Thread): still going on, can still
  change. The side is computed from the current message time, never moved.
  Only the two time axes exist (message time, known time): a time the user
  stated is content. The compiler writes it into the Block in the fixed format
  ("tomorrow evening (2024-05-07 19:00)"), and "has its time passed" compares
  that text with the current message time; no third time is stored.
- A matter has no name: its identity is a code id (``MemoryThread``), its
  meaning lives in its summary, which follows the matter as it changes, and it
  points (``anchor_node_id``) at its latest step.
- This is explicit work. Retrieval puts the relevant matters and steps in front
  of the compiler (``known``); the compiler decides what a Block connects to and
  in which tense it was said. Nothing here decides by similarity (Path B).

Every write uses the message time of the batch, so replay is deterministic.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
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
_STATED = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}))?\b")
Embed = Callable[[str], Sequence[float]]
PAST, DAY = "happened", "in progress"


@dataclass(frozen=True, slots=True)
class Attachment:
    """One compiler signal: this Block's place in one matter."""

    thread: str | None  # "T<k>" known matter, "m:<id>" matter new in this batch, None = new
    after: str | None  # the step it builds on: "N<k>", "T<k>", "b<k>" or None
    happened: bool  # already happened when said (else: still going on)
    summary: str | None


@dataclass(frozen=True, slots=True)
class BlockLines:
    memory_id: str
    lines: tuple[Attachment, ...]
    settles: tuple[str, ...] = ()  # step refs ("N<k>", "b<k>") this Block shows are settled


@dataclass(frozen=True, slots=True)
class Known:
    """What the compiler sees, and how its refs resolve (journaled with the compile)."""

    payload: dict
    refs: dict[str, str] = field(default_factory=dict)  # "T1" -> thread_id, "N1" -> node_id


@dataclass
class ApplyResult:
    attached: int = 0
    settled: int = 0
    opened: list[str] = field(default_factory=list)  # new matter (thread) ids, in order
    failures: list[str] = field(default_factory=list)


def scope_root(root: Path | str, scope: Scope) -> Path:
    return Path(root) / hashlib.sha256(scope_json(scope).encode("utf-8")).hexdigest()


def stated_end(content: str) -> datetime | None:
    """The latest time the content states in the fixed format, read like message time.

    A bare date passes at the end of that day. This reads content; it is not a
    third time axis.
    """
    ends = []
    for match in _STATED.finditer(content or ""):
        year, month, day, hour, minute = match.groups()
        try:
            if hour is None:
                ends.append(datetime(int(year), int(month), int(day), 23, 59, tzinfo=UTC))
            else:
                ends.append(datetime(int(year), int(month), int(day), int(hour), int(minute),
                                     tzinfo=UTC))
        except ValueError:
            continue
    return max(ends) if ends else None


class TenseStore:
    """How each Block was said (happened or not) and when a later Block settled it."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS said (
                memory_id TEXT PRIMARY KEY, happened INTEGER NOT NULL,
                known_at TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS settled (
                memory_id TEXT NOT NULL, settled_at TEXT NOT NULL, by_memory_id TEXT NOT NULL,
                PRIMARY KEY (memory_id, settled_at, by_memory_id));
            """
        )
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def record(self, memory_id: str, *, happened: bool, known_at: datetime) -> None:
        """First statement wins: how it was said does not change later."""
        with self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO said(memory_id, happened, known_at) VALUES (?, ?, ?)",
                (memory_id, int(happened), known_at.isoformat()),
            )

    def settle(self, memory_id: str, *, at: datetime, by: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR IGNORE INTO settled VALUES (?, ?, ?)",
                              (memory_id, at.isoformat(), by))

    def side(self, memory_id: str, now: datetime, content: str) -> str | None:
        """Which side of the terminator at message time ``now``; None when not on a line."""
        row = self.conn.execute(
            "SELECT happened FROM said WHERE memory_id = ?", (memory_id,)
        ).fetchone()
        if row is None:
            return None
        if row[0] or ((end := stated_end(content)) is not None and end <= now):
            return PAST
        settled = self.conn.execute(
            "SELECT 1 FROM settled WHERE memory_id = ? AND settled_at <= ? LIMIT 1",
            (memory_id, now.isoformat()),
        ).fetchone()
        return PAST if settled else DAY


class PathATree:
    """LCE Lines of one Scope. Not thread-safe: the runtime owner thread uses it."""

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
        self.tense = TenseStore(base / "tense.sqlite")
        self.rejections = RejectionStore(base / "rejections.sqlite")
        self.view = LineGraphView(store=self.lines, read_state=self._state)
        self.evidence_time = evidence_time or (lambda memory: memory.occurred_at)

    def close(self) -> None:
        self.lines.close()
        self.tense.close()
        self.rejections.close()

    # -- reads -------------------------------------------------------------

    def _state(self, state_id: str) -> StateView | None:
        memory = self.canonical.get(state_id)
        if memory is None:
            return None
        return StateView(
            state_id, memory.content, memory.occurred_at,
            tuple(ref.source_key for ref in memory.provenance.source_refs),
            memory.lifecycle is MemoryLifecycle.ACTIVE,
        )

    def matters(self) -> tuple[MemoryThread, ...]:
        return self.products.list_threads(self.scope, status=ThreadStatus.OPEN)

    def nodes(self, cutoff: datetime) -> list[tuple[str, str, StateView]]:
        """(line_id, node_id, state) for every visible step."""
        out = []
        for line in self.lines.list_lines():
            for node_id in self.view.visible_node_ids(line.line_id, cutoff=cutoff):
                state = self.view.state_for_node_at_cutoff(node_id, cutoff=cutoff)
                if state is not None:
                    out.append((line.line_id, node_id, state))
        return out

    def side(self, node_id: str, now: datetime) -> str | None:
        block_id = self.lines.get_node(node_id).block_id
        memory = self.canonical.get(block_id)
        return self.tense.side(block_id, now, "" if memory is None else memory.content)

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
        """Matters and steps relevant to this batch, with refs for the compiler.

        Steps still on the day side are shown first (they are what can still
        change), then recent and content-cued steps, then every chosen matter's
        latest step, so "after" can name it.
        """
        refs: dict[str, str] = {}
        matters = {t.thread_id: t for t in self.matters()}
        cards = [
            LineCard(t.thread_id, t.working_summary, False, t.updated_at,
                     "\n".join(x for x in (t.working_summary or "", self._text(t, now)) if x))
            for t in matters.values()
        ]
        chosen = select_known_lines(cards, batch_text=batch_text, now=now, embed=embed,
                                    **threads_policy).lines
        nodes = self.nodes(now)
        state_of = {node_id: state for _, node_id, state in nodes}
        line_of = {node_id: line_id for line_id, node_id, _ in nodes}
        day = sorted((n for _, n, _ in nodes if self.side(n, now) == DAY),
                     key=lambda n: state_of[n].occurred_at)[-steps_policy.get("recent", 10):]
        latest = sorted((n for _, n, _ in nodes), key=lambda n: state_of[n].occurred_at)[
            len(nodes) - steps_policy.get("latest", 0):] if steps_policy.get("latest") else []
        picked = [c.question for c in select_known_lines(
            [LineCard(n, None, True, s.occurred_at, s.content) for _, n, s in nodes],
            batch_text=batch_text, now=now, embed=embed, due=0,
            **{k: v for k, v in steps_policy.items() if k != "latest"},
        ).lines]
        anchors = [matters[c.question].anchor_node_id for c in chosen]
        steps = list(dict.fromkeys(
            n for n in (*day, *latest, *picked, *anchors) if n is not None and n in state_of))
        step_ref: dict[str, str] = {}
        payload_steps = []
        for node_id in sorted(steps, key=lambda n: (state_of[n].occurred_at, n)):
            ref = f"N{len(step_ref) + 1}"
            step_ref[node_id], refs[ref] = ref, node_id
            state = state_of[node_id]
            memory = self.canonical.get(state.state_id)
            said = None if memory is None else self.evidence_time(memory)
            payload_steps.append({
                "ref": ref, "text": state.content[:snippet], "state": self.side(node_id, now),
                **({"time": said.strftime("%Y-%m-%d %H:%M")} if said else {}),
            })
        payload_threads = []
        for i, card in enumerate(chosen, 1):
            t = matters[card.question]
            refs[f"T{i}"] = t.thread_id
            entry = {"ref": f"T{i}", "summary": t.working_summary or t.open_question}
            if t.anchor_node_id in step_ref:
                entry["latest"] = step_ref[t.anchor_node_id]
            payload_threads.append(entry)
        rejected = self.rejections.relevant(
            region_ids=tuple({line_of[n] for n in steps if n in line_of} | set(matters)))
        return Known(
            {"threads": payload_threads, "steps": payload_steps,
             "rejected": [r.rejected_content for r in rejected]},
            refs,
        )

    def _text(self, matter: MemoryThread, now: datetime) -> str:
        if matter.anchor_node_id is None:
            return ""
        try:
            state = self.view.state_for_node_at_cutoff(matter.anchor_node_id, cutoff=now)
        except KeyError:
            return ""
        return "" if state is None else state.content

    # -- apply compiler signals ------------------------------------------

    def apply(
        self, blocks: Sequence[BlockLines], refs: dict[str, str], at: datetime,
        *, open_matters: bool = True,
    ) -> ApplyResult:
        """Put each Block on the lines it connects to, in the tense it was said.

        ``open_matters=False`` is point placement: a Block is a node linked to the
        steps it carries on, and no matter (Thread product) is opened for it."""
        result = ApplyResult()
        nodes_of_block: dict[int, list[str]] = {}
        opened_by_block: dict[int, list[str]] = {}
        opened_by_key: dict[str, str] = {}  # "m:<id>" -> thread id, within this batch
        for index, block in enumerate(blocks, 1):
            memory = self.canonical.get(block.memory_id)
            if memory is None:
                result.failures.append(f"block {block.memory_id} not in canonical Memory")
                continue
            for ref in block.settles:  # an earlier open step is settled by this Block
                target = None
                if ref.startswith("b") and ref[1:].isdigit() and 0 < int(ref[1:]) < index:
                    target = blocks[int(ref[1:]) - 1].memory_id
                elif ref.startswith("N") and (node_id := refs.get(ref)):
                    try:
                        target = self.lines.get_node(node_id).block_id
                    except KeyError:
                        target = None
                if target is None:
                    result.failures.append(f"{block.memory_id}: cannot settle {ref}")
                    continue
                self.tense.settle(target, at=at, by=memory.memory_id)
                result.settled += 1
            lines = []  # only refs to an earlier place: no cycle, no made-up id (design 4.2)
            for line in block.lines:
                if line.after is None or self._earlier(line.after, refs, index):
                    lines.append(line)
                else:
                    result.failures.append(f"{block.memory_id}: dropped ref {line.after}")
            if block.lines and not lines:  # every link refused: the Block stands alone
                lines = [replace(block.lines[0], after=None)]
            for k, line in enumerate(lines):
                try:
                    node = self._attach(line, memory, refs, nodes_of_block, opened_by_block,
                                        opened_by_key, index, k, at, result, open_matters)
                except ValueError as exc:  # a refused relation never loses the Block
                    result.failures.append(f"{block.memory_id}: {exc}")
                    continue
                nodes_of_block.setdefault(index, []).append(node)
        return result

    def _matter(
        self, ref: str | None, refs: dict[str, str], opened_by_block: dict[int, list[str]],
        opened_by_key: dict[str, str],
    ) -> MemoryThread | None:
        """Identity is the code id behind a ref, never a name or wording."""
        thread_id = None
        if ref and ref.startswith("m:"):
            thread_id = opened_by_key.get(ref)
        elif ref and ref.startswith("T"):
            thread_id = refs.get(ref)
        elif ref and ref.startswith("b") and ref[1:].isdigit():
            opened = opened_by_block.get(int(ref[1:]), [])
            thread_id = opened[0] if opened else None
        thread = None if thread_id is None else self.products.get_thread(thread_id)
        return thread if thread is not None and thread.status is ThreadStatus.OPEN else None

    @staticmethod
    def _earlier(ref: str, refs: dict[str, str], index: int) -> bool:
        """A step on record ("N<k>", "T<k>") or an earlier block of this batch."""
        if ref.startswith("b"):
            return ref[1:].isdigit() and 0 < int(ref[1:]) < index
        return ref in refs

    def _parent(
        self, after: str | None, matter: MemoryThread | None, refs: dict[str, str],
        nodes_of_block: dict[int, list[str]],
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
        if node_id is not None:
            try:
                self.lines.get_node(node_id)
            except KeyError:
                return None
        return node_id

    def _attach(
        self, line: Attachment, memory: CommittedMemory, refs: dict[str, str],
        nodes_of_block: dict[int, list[str]], opened_by_block: dict[int, list[str]],
        opened_by_key: dict[str, str], index: int, k: int, at: datetime, result: ApplyResult,
        open_matters: bool = True,
    ) -> str:
        matter = self._matter(line.thread, refs, opened_by_block, opened_by_key)
        parent = self._parent(line.after, matter, refs, nodes_of_block)
        if parent is not None:
            line_id = self.lines.get_node(parent).line_id
        else:
            line_id = self.lines.create_line((memory.memory_id,), created_at=at).line_id
        # a Block citing several messages spans from the earliest to the latest (design 2)
        times = [ref.occurred_at for ref in memory.provenance.source_refs] or [memory.occurred_at]
        node, _, _ = self.lines.ensure_node(
            line_id, block_id=memory.memory_id, state_id=memory.memory_id,
            occurred_start=min(times), occurred_end=max(times),
            knowledge_at=at,
        )
        if parent is not None and parent != node.node_id:
            self.lines.add_edge(line_id, parent, node.node_id, knowledge_at=at)
        self.tense.record(memory.memory_id, happened=line.happened, known_at=at)
        result.attached += 1
        if matter is None and not open_matters:
            return node.node_id
        if matter is None:
            # a new matter: its id comes from the Block that started it, never from words
            matter = self.products.open_thread(
                thread_id=_thread_identity(self.scope, f"{memory.memory_id}\n{k}"),
                scope=self.scope, open_question=(line.summary or memory.content)[:2048],
                supporting_memory_ids=(memory.memory_id,), at=at,
                working_summary=line.summary, anchor_node_id=node.node_id,
            )
            opened_by_block.setdefault(index, []).append(matter.thread_id)
            if line.thread and line.thread.startswith("m:"):
                opened_by_key[line.thread] = matter.thread_id
            result.opened.append(matter.thread_id)
        else:
            self.products.update_thread(
                matter.thread_id, supporting_memory_ids=(memory.memory_id,),
                at=max(at, matter.updated_at), working_summary=line.summary,
                anchor_node_id=node.node_id,
            )
        return node.node_id
