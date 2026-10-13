"""AMLSourceStore: the immutable raw source for benchmark runs (ADR-0001 s.10).

AML supplies source material through Add but exposes no host-native session
database during Search, so the adapter keeps raw messages here. Rows are
append-only; a replayed request must match the stored payload exactly.
Canonical Memory refers to these rows only through native ``SourceRef``s.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mr_mem.contracts import Scope, ScopeDomain
from mr_mem.memory.source import SourceRef

NAMESPACE = "aml"


class RequestConflict(ValueError):
    """A request_id was replayed with different content."""


@dataclass(frozen=True, slots=True)
class RawMessage:
    record_id: str
    user_id: str
    session_id: str
    request_id: str
    index: int
    role: str
    content: str
    occurred_at: datetime
    fingerprint: str

    @property
    def ref(self) -> SourceRef:
        return SourceRef(
            NAMESPACE, self.session_id, self.record_id, self.occurred_at,
            fingerprint=self.fingerprint,
        )


def user_scope(user_id: str) -> Scope:
    return Scope(ScopeDomain.USER, user_id=user_id)


def payload_hash(payload: dict) -> str:
    canonical = {
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
        "messages": payload["messages"],
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class AmlSourceStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS requests (
                request_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                received_at TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'received',
                compiled TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                record_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                request_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                seq INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS messages_session
                ON messages(user_id, session_id, seq);
            CREATE TABLE IF NOT EXISTS lineage_questions (
                user_id TEXT NOT NULL,
                question TEXT NOT NULL,
                first_seen TEXT NOT NULL,
                PRIMARY KEY(user_id, question)
            );
            """
        )
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(lineage_questions)")}
        if "last_active" not in columns:  # message time of the latest Block on the line
            self._conn.execute("ALTER TABLE lineage_questions ADD COLUMN last_active TEXT")
            self._conn.execute("UPDATE lineage_questions SET last_active=first_seen")
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # -- write --------------------------------------------------------

    def put_request(self, payload: dict, *, received_at: datetime) -> str:
        """Store raw messages once. Returns the request status."""
        digest = payload_hash(payload)
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_hash,status FROM requests WHERE request_id=?",
                (payload["request_id"],),
            ).fetchone()
            if row is not None:
                if row[0] != digest:
                    raise RequestConflict("request_id replayed with different content")
                return row[1]
            seq = self._conn.execute("SELECT COALESCE(MAX(seq),0) FROM messages").fetchone()[0]
            with self._conn:
                self._conn.execute(
                    "INSERT INTO requests(request_id,user_id,session_id,payload_hash,received_at)"
                    " VALUES (?,?,?,?,?)",
                    (payload["request_id"], payload["user_id"], payload["session_id"],
                     digest, received_at.isoformat()),
                )
                for index, message in enumerate(payload["messages"]):
                    occurred = _occurred_at(message.get("timestamp"), received_at, index)
                    content = message["content"]
                    self._conn.execute(
                        "INSERT INTO messages VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (
                            f"{payload['request_id']}:{index}", payload["user_id"],
                            payload["session_id"], payload["request_id"], index,
                            message["role"], content, occurred.isoformat(),
                            hashlib.sha256(content.encode("utf-8")).hexdigest(), seq + index + 1,
                        ),
                    )
            return "received"

    def set_compiled(self, request_id: str, compiled: dict) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE requests SET compiled=?, status='compiled', error=NULL "
                "WHERE request_id=? AND compiled IS NULL",
                (json.dumps(compiled, ensure_ascii=False, sort_keys=True), request_id),
            )

    def set_done(self, request_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE requests SET status='done', error=NULL WHERE request_id=?", (request_id,)
            )

    def set_error(self, request_id: str, error: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE requests SET error=? WHERE request_id=?", (error[:2000], request_id)
            )

    def remember_question(self, user_id: str, question: str, at: datetime) -> None:
        """Record that a line was touched at message time ``at``."""
        stamp = at.isoformat()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO lineage_questions(user_id,question,first_seen,last_active) "
                "VALUES (?,?,?,?) ON CONFLICT(user_id,question) DO UPDATE SET "
                "last_active=MAX(COALESCE(last_active,''), excluded.last_active)",
                (user_id, question, stamp, stamp),
            )

    # -- read ---------------------------------------------------------

    def compiled(self, request_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT compiled FROM requests WHERE request_id=?", (request_id,)
            ).fetchone()
        return None if row is None or row[0] is None else json.loads(row[0])

    def request_messages(self, request_id: str) -> tuple[RawMessage, ...]:
        return self._select("WHERE request_id=? ORDER BY idx", (request_id,))

    def context_before(self, message: RawMessage, limit: int) -> tuple[RawMessage, ...]:
        """Earlier messages of the same session, oldest first (context only)."""
        if limit <= 0:
            return ()
        with self._lock:
            seq = self._conn.execute(
                "SELECT seq FROM messages WHERE record_id=?", (message.record_id,)
            ).fetchone()[0]
        rows = self._select(
            "WHERE user_id=? AND session_id=? AND seq<? ORDER BY seq DESC LIMIT ?",
            (message.user_id, message.session_id, seq, limit),
        )
        return tuple(reversed(rows))

    def lines(self, user_id: str) -> tuple[tuple[str, datetime], ...]:
        """Every line of a user with the message time it was last touched."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT question, COALESCE(last_active, first_seen) FROM lineage_questions "
                "WHERE user_id=?",
                (user_id,),
            ).fetchall()
        return tuple((q, datetime.fromisoformat(t)) for q, t in rows)

    def questions(self, user_id: str, limit: int = 50) -> tuple[str, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT question FROM lineage_questions WHERE user_id=? "
                "ORDER BY first_seen DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return tuple(row[0] for row in rows)

    def _select(self, clause: str, args: tuple) -> tuple[RawMessage, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT record_id,user_id,session_id,request_id,idx,role,content,occurred_at,"
                f"fingerprint FROM messages {clause}",
                args,
            ).fetchall()
        return tuple(
            RawMessage(r[0], r[1], r[2], r[3], r[4], r[5], r[6],
                       datetime.fromisoformat(r[7]), r[8])
            for r in rows
        )

    def message_by_record(self, record_id: str) -> RawMessage | None:
        found = self._select("WHERE record_id=?", (record_id,))
        return found[0] if found else None

    def message_for(self, ref: SourceRef) -> RawMessage | None:
        if ref.source_namespace != NAMESPACE:
            return None
        return self.message_by_record(ref.record_id)

    # -- SourceRefReader ----------------------------------------------

    def current_ref(self, scope: Scope, ref: SourceRef) -> SourceRef | None:
        message = self.message_for(ref)
        if message is None or scope != user_scope(message.user_id):
            return None
        if message.session_id != ref.session_id:
            return None
        return message.ref

    def render(self, ref: SourceRef) -> str | None:
        message = self.message_for(ref)
        if message is None:
            return None
        stamp = message.occurred_at.strftime("%Y-%m-%d %H:%M")
        return f"[Original message | {stamp}] {message.role}: {message.content}"


def _occurred_at(timestamp: object, received_at: datetime, index: int) -> datetime:
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        return datetime.fromtimestamp(timestamp / 1000.0, tz=UTC)
    # No source time: keep arrival order deterministic within the request.
    return received_at + timedelta(microseconds=index)
