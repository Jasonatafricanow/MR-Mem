"""Path A: a lived experience as a git-like tree of locked SemanticBlocks.

Ported from LCE ``cognition/line_graph.py`` (origin/master 5228475) and
audited against the governance design v6 (4.4-4.5; docs/LCE_PORT.md):

- A Line is one experience ("the Portugal trip"). Its nodes are SemanticBlocks
  that are locked (they happened). A node may have several children: parallel
  branches that are unrelated to each other but hang under one shared step
  ("arrived in Portugal" -> dinner / shopping / a walk). Multiple parents mean
  conjunctive rejoin and must be authorized by the caller.
- A node is keyed by the Block it was first created for; a corrected Block (a
  new canonical Memory that supersedes the old one) is a new *state* of the same
  node, never a new trajectory point.
- Membership and edges are revisioned by knowledge time (``known_at`` ->
  ``retired_at``): a correction retires the current relation and adds another;
  history is never rewritten.

Audit changes against the LCE original:

1. Visibility no longer requires every ancestor to be visible, and a node whose
   source evidence is gone stays readable, marked incomplete. Raw text is the
   authority; projections persist greyed and are never hidden (design 1).
2. Knowledge times are explicit everywhere (no ``datetime.now()`` defaults), so
   replay is deterministic; wall-clock ``created_at`` is audit metadata only.
3. Line identity admission by vector convergence (``LineAssembler``) is not
   here: in Path A the semantic compiler decides where a Block attaches. That
   machinery belongs to Path B.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo != UTC:
        raise ValueError(f"{name} must be an aware UTC datetime")


def _parse(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    _require_utc(parsed, "stored datetime")
    return parsed


class LineTraversalLimitExceeded(RuntimeError):
    """An exact traversal could not finish within its safety ceiling."""


class LineGraphCycleError(RuntimeError):
    """Persisted Line structure violates the DAG invariant."""


@dataclass(frozen=True, slots=True)
class LineRecord:
    line_id: str
    seed_block_ids: tuple[str, ...]
    created_at: datetime


@dataclass(frozen=True, slots=True)
class LineNode:
    node_id: str
    line_id: str
    block_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class LineNodeState:
    node_id: str
    state_id: str
    occurred_start: datetime
    occurred_end: datetime
    knowledge_at: datetime
    created_at: datetime


@dataclass(frozen=True, slots=True)
class StateView:
    """What a node state reads as, supplied by the canonical Memory store."""

    state_id: str
    content: str
    occurred_at: datetime
    source_keys: tuple[str, ...]
    evidence_complete: bool  # False: the source evidence is gone (greyed, kept)


StateReader = Callable[[str], "StateView | None"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lines (
    line_id TEXT PRIMARY KEY,
    seed_block_ids_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line_nodes (
    node_id TEXT PRIMARY KEY,
    line_id TEXT NOT NULL REFERENCES lines(line_id) ON DELETE CASCADE,
    block_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(line_id, block_id)
);
CREATE TABLE IF NOT EXISTS line_node_states (
    node_id TEXT NOT NULL REFERENCES line_nodes(node_id) ON DELETE CASCADE,
    state_id TEXT NOT NULL,
    occurred_start TEXT NOT NULL,
    occurred_end TEXT NOT NULL,
    knowledge_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(node_id, state_id)
);
CREATE TABLE IF NOT EXISTS line_edges (
    line_id TEXT NOT NULL REFERENCES lines(line_id) ON DELETE CASCADE,
    parent_node_id TEXT NOT NULL REFERENCES line_nodes(node_id) ON DELETE CASCADE,
    child_node_id TEXT NOT NULL REFERENCES line_nodes(node_id) ON DELETE CASCADE,
    PRIMARY KEY(line_id, parent_node_id, child_node_id)
);
CREATE TABLE IF NOT EXISTS line_node_memberships (
    membership_revision_id TEXT PRIMARY KEY,
    node_id TEXT NOT NULL REFERENCES line_nodes(node_id) ON DELETE CASCADE,
    known_at TEXT NOT NULL,
    retired_at TEXT,
    created_at TEXT NOT NULL,
    derivation_fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS line_edge_revisions (
    edge_revision_id TEXT PRIMARY KEY,
    line_id TEXT NOT NULL,
    parent_node_id TEXT NOT NULL,
    child_node_id TEXT NOT NULL,
    known_at TEXT NOT NULL,
    retired_at TEXT,
    created_at TEXT NOT NULL,
    derivation_fingerprint TEXT NOT NULL,
    FOREIGN KEY(line_id, parent_node_id, child_node_id)
        REFERENCES line_edges(line_id, parent_node_id, child_node_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_line_nodes_block ON line_nodes(block_id);
CREATE INDEX IF NOT EXISTS idx_line_node_states_state ON line_node_states(state_id);
CREATE INDEX IF NOT EXISTS idx_line_membership_node_time
    ON line_node_memberships(node_id, known_at);
CREATE INDEX IF NOT EXISTS idx_line_edge_revision_time
    ON line_edge_revisions(line_id, parent_node_id, child_node_id, known_at);
CREATE TABLE IF NOT EXISTS line_graph_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:24]


class LineGraphStore:
    """SQLite persistence for Line identities and their knowledge-time revisioned DAG."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.conn.commit()
        # Path A: the compiler prompt version that produced the relations.
        self._fingerprint = "unset"

    def close(self) -> None:
        self.conn.close()

    def set_derivation_fingerprint(self, value: str) -> None:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("derivation fingerprint must be nonempty")
        self._fingerprint = value.strip()

    def get_metadata(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM line_graph_metadata WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row is not None else None

    def set_metadata(self, key: str, value: str) -> None:
        if not key.strip():
            raise ValueError("metadata key must be nonempty")
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO line_graph_metadata VALUES (?, ?)", (key, value)
            )

    # -- Lines and nodes -------------------------------------------------

    @staticmethod
    def line_id_for_seed(block_ids: tuple[str, ...]) -> str:
        if not block_ids:
            raise ValueError("a Line seed requires at least one block")
        ordered = tuple(dict.fromkeys(block_ids))
        return "line_" + _digest(json.dumps(ordered, separators=(",", ":")))

    @staticmethod
    def node_id_for(line_id: str, block_id: str) -> str:
        return "lnode_" + _digest(line_id, block_id)

    def create_line(
        self, seed_block_ids: tuple[str, ...], *, created_at: datetime, commit: bool = True
    ) -> LineRecord:
        _require_utc(created_at, "created_at")
        ordered = tuple(dict.fromkeys(seed_block_ids))
        line_id = self.line_id_for_seed(ordered)
        if not self.has_line(line_id):
            self.conn.execute(
                "INSERT INTO lines VALUES (?, ?, ?)",
                (line_id, json.dumps(ordered), created_at.isoformat()),
            )
            if commit:
                self.conn.commit()
        return self.get_line(line_id)

    def has_line(self, line_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM lines WHERE line_id = ?", (line_id,)
        ).fetchone() is not None

    def get_line(self, line_id: str) -> LineRecord:
        row = self.conn.execute(
            "SELECT line_id, seed_block_ids_json, created_at FROM lines WHERE line_id = ?",
            (line_id,),
        ).fetchone()
        if row is None:
            raise KeyError(line_id)
        return LineRecord(str(row[0]), tuple(json.loads(str(row[1]))), _parse(str(row[2])))

    def list_lines(self) -> tuple[LineRecord, ...]:
        rows = self.conn.execute("SELECT line_id FROM lines ORDER BY created_at, line_id")
        return tuple(self.get_line(str(row[0])) for row in rows.fetchall())

    def ensure_node(
        self,
        line_id: str,
        *,
        block_id: str,
        state_id: str,
        occurred_start: datetime,
        occurred_end: datetime,
        knowledge_at: datetime,
        commit: bool = True,
    ) -> tuple[LineNode, bool, bool]:
        """Add ``block_id`` to the Line (or a new state of its node).

        Returns (node, node added, state added). ``state_id`` is the canonical
        Memory ID of this version of the Block; a corrected Block keeps the node.
        """
        self.get_line(line_id)
        for value, name in ((occurred_start, "occurred_start"), (occurred_end, "occurred_end"),
                            (knowledge_at, "knowledge_at")):
            _require_utc(value, name)
        if occurred_end < occurred_start:
            raise ValueError("occurred_end must not precede occurred_start")
        now = datetime.now(UTC).isoformat()  # audit only
        row = self.conn.execute(
            "SELECT node_id FROM line_nodes WHERE line_id = ? AND block_id = ?",
            (line_id, block_id),
        ).fetchone()
        added_node = row is None
        node_id = self.node_id_for(line_id, block_id) if row is None else str(row[0])
        if added_node:
            self.conn.execute(
                "INSERT INTO line_nodes VALUES (?, ?, ?, ?)", (node_id, line_id, block_id, now)
            )
        self._activate_membership(node_id, knowledge_at)
        added_state = self.conn.execute(
            "INSERT OR IGNORE INTO line_node_states VALUES (?, ?, ?, ?, ?, ?)",
            (node_id, state_id, occurred_start.isoformat(), occurred_end.isoformat(),
             knowledge_at.isoformat(), now),
        ).rowcount > 0
        if commit:
            self.conn.commit()
        return self.get_node(node_id), added_node, added_state

    def get_node(self, node_id: str) -> LineNode:
        row = self.conn.execute(
            "SELECT node_id, line_id, block_id, created_at FROM line_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            raise KeyError(node_id)
        return LineNode(str(row[0]), str(row[1]), str(row[2]), _parse(str(row[3])))

    def states_for_node(self, node_id: str) -> tuple[LineNodeState, ...]:
        rows = self.conn.execute(
            "SELECT node_id, state_id, occurred_start, occurred_end, knowledge_at, created_at "
            "FROM line_node_states WHERE node_id = ? ORDER BY knowledge_at, created_at, state_id",
            (node_id,),
        ).fetchall()
        return tuple(
            LineNodeState(str(r[0]), str(r[1]), _parse(str(r[2])), _parse(str(r[3])),
                          _parse(str(r[4])), _parse(str(r[5])))
            for r in rows
        )

    def node_for_block(self, line_id: str, block_id: str) -> LineNode | None:
        row = self.conn.execute(
            "SELECT node_id FROM line_nodes WHERE line_id = ? AND block_id = ?",
            (line_id, block_id),
        ).fetchone()
        return self.get_node(str(row[0])) if row is not None else None

    def nodes_for_state(self, state_id: str) -> tuple[LineNode, ...]:
        """Upward index: every node a canonical Memory version is a state of."""
        rows = self.conn.execute(
            "SELECT DISTINCT node_id FROM line_node_states WHERE state_id = ? ORDER BY node_id",
            (state_id,),
        ).fetchall()
        return tuple(self.get_node(str(row[0])) for row in rows)

    def nodes_for_line(self, line_id: str) -> tuple[LineNode, ...]:
        rows = self.conn.execute(
            "SELECT node_id FROM line_nodes WHERE line_id = ? ORDER BY created_at, node_id",
            (line_id,),
        ).fetchall()
        return tuple(self.get_node(str(row[0])) for row in rows)

    def lines_for_block(self, block_id: str) -> tuple[str, ...]:
        rows = self.conn.execute(
            "SELECT DISTINCT line_id FROM line_nodes WHERE block_id = ? ORDER BY line_id",
            (block_id,),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    # -- knowledge-time revisioned relations ------------------------------

    def membership_active_at(self, node_id: str, cutoff: datetime) -> bool:
        _require_utc(cutoff, "cutoff")
        at = cutoff.isoformat()
        return self.conn.execute(
            "SELECT 1 FROM line_node_memberships WHERE node_id = ? AND known_at <= ? "
            "AND (retired_at IS NULL OR retired_at > ?) LIMIT 1",
            (node_id, at, at),
        ).fetchone() is not None

    def _activate_membership(self, node_id: str, known_at: datetime) -> bool:
        if self.membership_active_at(node_id, known_at):
            return False
        ordinal = self.conn.execute(
            "SELECT COUNT(*) FROM line_node_memberships WHERE node_id = ?", (node_id,)
        ).fetchone()[0] + 1
        iso = known_at.isoformat()
        return self.conn.execute(
            "INSERT OR IGNORE INTO line_node_memberships VALUES (?, ?, ?, NULL, ?, ?)",
            ("membership_" + _digest("membership", node_id, iso, str(ordinal)), node_id, iso,
             datetime.now(UTC).isoformat(), self._fingerprint),
        ).rowcount > 0

    def active_node_ids_at(self, line_id: str, cutoff: datetime) -> tuple[str, ...]:
        return tuple(
            node.node_id for node in self.nodes_for_line(line_id)
            if self.membership_active_at(node.node_id, cutoff)
        )

    def edge_active_at(self, line_id: str, parent_id: str, child_id: str, cutoff: datetime) -> bool:
        _require_utc(cutoff, "cutoff")
        at = cutoff.isoformat()
        return self.conn.execute(
            "SELECT 1 FROM line_edge_revisions WHERE line_id = ? AND parent_node_id = ? "
            "AND child_node_id = ? AND known_at <= ? AND (retired_at IS NULL OR retired_at > ?) "
            "LIMIT 1",
            (line_id, parent_id, child_id, at, at),
        ).fetchone() is not None

    def edges_for_line_at(self, line_id: str, cutoff: datetime) -> tuple[tuple[str, str], ...]:
        _require_utc(cutoff, "cutoff")
        at = cutoff.isoformat()
        rows = self.conn.execute(
            "SELECT DISTINCT parent_node_id, child_node_id FROM line_edge_revisions "
            "WHERE line_id = ? AND known_at <= ? AND (retired_at IS NULL OR retired_at > ?) "
            "ORDER BY parent_node_id, child_node_id",
            (line_id, at, at),
        ).fetchall()
        return tuple((str(p), str(c)) for p, c in rows)

    def _related_at(self, node_id: str, cutoff: datetime, *, parents: bool) -> tuple[str, ...]:
        _require_utc(cutoff, "cutoff")
        node = self.get_node(node_id)
        at = cutoff.isoformat()
        mine, other = ("child_node_id", "parent_node_id") if parents else (
            "parent_node_id", "child_node_id")
        rows = self.conn.execute(
            f"SELECT DISTINCT {other} FROM line_edge_revisions WHERE line_id = ? AND {mine} = ? "
            "AND known_at <= ? AND (retired_at IS NULL OR retired_at > ?) ORDER BY 1",
            (node.line_id, node_id, at, at),
        ).fetchall()
        return tuple(str(row[0]) for row in rows)

    def parents_at(self, node_id: str, cutoff: datetime) -> tuple[str, ...]:
        return self._related_at(node_id, cutoff, parents=True)

    def children_at(self, node_id: str, cutoff: datetime) -> tuple[str, ...]:
        return self._related_at(node_id, cutoff, parents=False)

    def retire_edge_from(
        self, line_id: str, parent_id: str, child_id: str, cutoff: datetime, *, commit: bool = True
    ) -> bool:
        """Retire one relation from ``cutoff`` forward; earlier history is kept."""
        _require_utc(cutoff, "cutoff")
        at = cutoff.isoformat()
        changed = self.conn.execute(
            "UPDATE line_edge_revisions SET retired_at = ? WHERE line_id = ? "
            "AND parent_node_id = ? AND child_node_id = ? "
            "AND (retired_at IS NULL OR retired_at > ?)",
            (at, line_id, parent_id, child_id, at),
        ).rowcount
        if commit:
            self.conn.commit()
        return changed > 0

    def retire_membership_from(
        self, node_id: str, cutoff: datetime, *, commit: bool = True
    ) -> bool:
        _require_utc(cutoff, "cutoff")
        at = cutoff.isoformat()
        changed = self.conn.execute(
            "UPDATE line_node_memberships SET retired_at = ? WHERE node_id = ? "
            "AND (retired_at IS NULL OR retired_at > ?)",
            (at, node_id, at),
        ).rowcount
        if commit:
            self.conn.commit()
        return changed > 0

    def _reachable(self, start: str, target: str, cutoff: datetime) -> bool:
        pending, seen = [start], set()
        while pending:
            current = pending.pop()
            if current == target:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(self.children_at(current, cutoff))
        return False

    def _boundaries_from(self, line_id: str, start: datetime) -> tuple[datetime, ...]:
        """Every future edge-revision boundary at which the DAG can change."""
        cutoffs = {start}
        for known_at, retired_at in self.conn.execute(
            "SELECT known_at, retired_at FROM line_edge_revisions WHERE line_id = ?", (line_id,)
        ).fetchall():
            for value in (known_at, retired_at):
                if value is not None and (parsed := _parse(str(value))) >= start:
                    cutoffs.add(parsed)
        return tuple(sorted(cutoffs))

    def add_edge(
        self, line_id: str, parent_id: str, child_id: str, *, knowledge_at: datetime,
        commit: bool = True,
    ) -> bool:
        """Relate two nodes of one Line. False when already implied (no shortcut edges)."""
        _require_utc(knowledge_at, "knowledge_at")
        if parent_id == child_id:
            raise ValueError("a Line edge cannot self-reference")
        lines = {self.get_node(parent_id).line_id, self.get_node(child_id).line_id}
        if lines != {line_id}:
            raise ValueError("Line edges cannot cross Line identity")
        if self.edge_active_at(line_id, parent_id, child_id, knowledge_at):
            return False
        if self._reachable(parent_id, child_id, knowledge_at):
            return False  # already an ancestor: a direct edge would be a redundant shortcut
        for cutoff in self._boundaries_from(line_id, knowledge_at):
            if self._reachable(child_id, parent_id, cutoff):
                raise ValueError(f"Line edge would create a cycle at {cutoff.isoformat()}")
        self.conn.execute("INSERT OR IGNORE INTO line_edges VALUES (?, ?, ?)",
                          (line_id, parent_id, child_id))
        ordinal = self.conn.execute(
            "SELECT COUNT(*) FROM line_edge_revisions WHERE line_id = ? AND parent_node_id = ? "
            "AND child_node_id = ?",
            (line_id, parent_id, child_id),
        ).fetchone()[0] + 1
        iso = knowledge_at.isoformat()
        added = self.conn.execute(
            "INSERT OR IGNORE INTO line_edge_revisions VALUES (?, ?, ?, ?, ?, NULL, ?, ?)",
            ("edge_" + _digest("edge", line_id, parent_id, child_id, iso, str(ordinal)),
             line_id, parent_id, child_id, iso, datetime.now(UTC).isoformat(), self._fingerprint),
        ).rowcount > 0
        if commit:
            self.conn.commit()
        return added

    def reset_derived(self) -> None:
        """Delete the rebuildable Line graph generation (metadata is kept)."""
        with self.conn:
            for table in ("line_edge_revisions", "line_node_memberships", "line_edges",
                          "line_node_states", "line_nodes", "lines"):
                self.conn.execute(f"DELETE FROM {table}")


class LineGraphView:
    """Knowledge-cutoff view over one store; node content comes from canonical Memory."""

    def __init__(self, *, store: LineGraphStore, read_state: StateReader) -> None:
        self.store = store
        self.read_state = read_state

    def state_for_node_at_cutoff(self, node_id: str, *, cutoff: datetime) -> StateView | None:
        """The newest state of the node known by ``cutoff`` (a correction wins)."""
        _require_utc(cutoff, "cutoff")
        for state in reversed(self.store.states_for_node(node_id)):
            if state.knowledge_at <= cutoff and (view := self.read_state(state.state_id)):
                return view
        return None

    def visible_node_ids(self, line_id: str, *, cutoff: datetime) -> tuple[str, ...]:
        """Nodes whose membership is current at ``cutoff`` and that have a readable state.

        Unlike the LCE original, ancestry is not required to be visible and a
        node without source evidence stays visible (``evidence_complete=False``).
        """
        return tuple(
            node_id for node_id in self.store.active_node_ids_at(line_id, cutoff)
            if self.state_for_node_at_cutoff(node_id, cutoff=cutoff) is not None
        )

    def frontier(self, line_id: str, *, cutoff: datetime) -> tuple[str, ...]:
        visible = set(self.visible_node_ids(line_id, cutoff=cutoff))
        with_children = {
            parent for parent, child in self.store.edges_for_line_at(line_id, cutoff)
            if parent in visible and child in visible
        }
        return tuple(sorted(visible - with_children))

    def paths_to_frontier(
        self, line_id: str, *, cutoff: datetime, max_paths: int = 128, max_nodes: int = 256
    ) -> tuple[tuple[str, ...], ...]:
        """Every exact root-to-frontier path. Bounds fail closed, never truncate."""
        if max_paths < 1 or max_nodes < 1:
            raise ValueError("path bounds must be positive")
        visible = set(self.visible_node_ids(line_id, cutoff=cutoff))
        paths: list[tuple[str, ...]] = []
        for frontier_id in self.frontier(line_id, cutoff=cutoff):
            pending = [(frontier_id, ())]
            while pending:
                node_id, suffix = pending.pop()
                if node_id in suffix:
                    raise LineGraphCycleError(f"cycle detected at {node_id}")
                path = (node_id, *suffix)
                parents = [p for p in self.store.parents_at(node_id, cutoff) if p in visible]
                if not parents:
                    if len(paths) >= max_paths:
                        raise LineTraversalLimitExceeded(f"more than max_paths={max_paths} paths")
                    paths.append(path)
                    continue
                if len(path) >= max_nodes:
                    raise LineTraversalLimitExceeded(f"a path is longer than max_nodes={max_nodes}")
                pending.extend((parent, path) for parent in parents)
        return tuple(sorted(set(paths)))

    def ancestors(self, node_id: str, *, cutoff: datetime) -> tuple[str, ...]:
        """Every visible ancestor (the background a branch hangs from), nearest first."""
        order: list[str] = []
        seen = {node_id}
        frontier = [node_id]
        while frontier:
            nxt: list[str] = []
            for current in frontier:
                for parent in self.store.parents_at(current, cutoff):
                    if parent not in seen:
                        seen.add(parent)
                        order.append(parent)
                        nxt.append(parent)
            frontier = nxt
        return tuple(order)

    def descendants(self, node_id: str, *, cutoff: datetime) -> tuple[str, ...]:
        """The branch below one node, breadth first."""
        order: list[str] = []
        seen = {node_id}
        frontier = [node_id]
        while frontier:
            nxt: list[str] = []
            for current in frontier:
                for child in self.store.children_at(current, cutoff):
                    if child not in seen:
                        seen.add(child)
                        order.append(child)
                        nxt.append(child)
            frontier = nxt
        return tuple(order)

    def raw_closure(
        self, node_id: str, *, cutoff: datetime, max_nodes: int = 100_000
    ) -> tuple[tuple[str, ...], bool]:
        """Source keys of the node and its ancestry, and whether all evidence is present.

        Exact or failing: exceeding ``max_nodes`` raises instead of truncating.
        """
        if self.state_for_node_at_cutoff(node_id, cutoff=cutoff) is None:
            raise ValueError("raw closure target is not visible at cutoff")
        pending, seen, keys, complete = [node_id], set(), set(), True
        while pending:
            current = pending.pop()
            if current in seen:
                continue
            if len(seen) >= max_nodes:
                raise LineTraversalLimitExceeded(f"raw closure exceeded max_nodes={max_nodes}")
            seen.add(current)
            if not self.store.membership_active_at(current, cutoff):
                continue
            state = self.state_for_node_at_cutoff(current, cutoff=cutoff)
            if state is None:
                continue
            keys.update(state.source_keys)
            complete = complete and state.evidence_complete
            pending.extend(p for p in self.store.parents_at(current, cutoff) if p not in seen)
        return tuple(sorted(keys)), complete
