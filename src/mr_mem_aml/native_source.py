"""Metadata-only source journal. Native bodies are resolved afresh by the host."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import datetime

from mr_mem.memory.source import SourceRef
from mr_mem_aml.source_store import AmlSourceStore, RequestConflict, user_scope


def encode_ref(ref):
    return {**asdict(ref), "occurred_at": ref.occurred_at.isoformat()}


def decode_ref(value):
    return SourceRef(**{**value, "occurred_at": datetime.fromisoformat(value["occurred_at"])})


class NativeSourceStore(AmlSourceStore):
    """One host reader per logical owner; no native body is written to this DB."""

    def __init__(self, path, *, user_id, reader):
        super().__init__(path)
        self.user_id, self.reader = user_id, reader
        self._bindings = {}
        self._conn.execute("CREATE TABLE IF NOT EXISTS native_refs ("
                           "record_id TEXT PRIMARY KEY, ref TEXT NOT NULL)")
        self._conn.execute("CREATE TABLE IF NOT EXISTS native_retired_versions ("
                           "request_id TEXT, version_key TEXT, "
                           "PRIMARY KEY(request_id,version_key))")
        self._conn.execute("CREATE TABLE IF NOT EXISTS native_binding (identity TEXT PRIMARY KEY)")
        identity = reader.identity
        bound = self._conn.execute("SELECT identity FROM native_binding").fetchall()
        if not isinstance(identity, str) or not identity or (bound and bound != [(identity,)]):
            self.close()
            raise ValueError("native reader binding changed")
        self._conn.execute("INSERT OR IGNORE INTO native_binding VALUES (?)", (identity,))
        owners = {row[0] for row in self._conn.execute("SELECT DISTINCT user_id FROM requests")}
        if owners - {user_id}:
            self.close()
            raise ValueError("native journal belongs to another logical owner")
        self._conn.commit()

    def admission_attributes(self, request_id):
        return (("native_ingest_batch", request_id),)

    def bind(self, request_id, messages):
        self._bindings[request_id] = tuple(messages)

    def put_request(self, payload, *, received_at):
        messages = self._bindings[payload["request_id"]]
        if payload["user_id"] != self.user_id or not messages:
            raise ValueError("native source owner and nonempty batch required")
        refs = [encode_ref(m.ref) for m in messages]
        digest = hashlib.sha256(json.dumps(refs, sort_keys=True).encode()).hexdigest()
        with self._lock, self._conn:
            row = self._conn.execute("SELECT payload_hash,status FROM requests WHERE request_id=?",
                                     (payload["request_id"],)).fetchone()
            if row:
                if row[0] != digest:
                    raise RequestConflict("native batch source versions changed")
                return row[1]
            seq = self._conn.execute("SELECT COALESCE(MAX(seq),0) FROM messages").fetchone()[0]
            self._conn.execute(
                "INSERT INTO requests(request_id,user_id,session_id,payload_hash,received_at) "
                "VALUES (?,?,?,?,?)", (payload["request_id"], self.user_id,
                                       payload["session_id"], digest, received_at.isoformat()))
            for index, message in enumerate(messages):
                record = f"{payload['request_id']}:{index}"
                self._conn.execute(
                    "INSERT INTO messages(record_id,user_id,session_id,request_id,idx,role,"
                    "content,occurred_at,fingerprint,seq,time_given) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (record, self.user_id, message.session_id, payload["request_id"], index,
                     message.role, "", message.occurred_at.isoformat(), message.fingerprint,
                     seq + index + 1, 1))
                self._conn.execute("INSERT INTO native_refs VALUES (?,?)",
                                   (record, json.dumps(refs[index])))
        return "received"

    def refs(self, request_id):
        return tuple(decode_ref(json.loads(row[0])) for row in self._conn.execute(
            "SELECT n.ref FROM native_refs n JOIN messages m USING(record_id) "
            "WHERE m.request_id=? ORDER BY m.idx", (request_id,)))

    def jobs(self, session_id):
        return tuple(self._conn.execute(
            "SELECT request_id,status FROM requests WHERE session_id=? ORDER BY rowid",
            (session_id,)))

    def sessions(self):
        return tuple(row[0] for row in self._conn.execute(
            "SELECT DISTINCT session_id FROM requests ORDER BY session_id"))

    def cancel(self, request_id):
        with self._conn:
            self._conn.execute("UPDATE requests SET status='cancelled' WHERE request_id=?",
                               (request_id,))

    def retire_ref(self, request_id, ref):
        with self._conn:
            self._conn.execute("INSERT OR IGNORE INTO native_retired_versions VALUES (?,?)",
                               (request_id, ref.version_key))

    def retired(self, request_id):
        return {r[0] for r in self._conn.execute(
            "SELECT version_key FROM native_retired_versions WHERE request_id=?", (request_id,))}

    def time_given(self, record_id):
        return True  # Native adapters require source timestamps, never receipt-time substitutes.

    def _select(self, clause, args):
        out = []
        for metadata in super()._select(clause, args):
            status = self._conn.execute("SELECT status FROM requests WHERE request_id=?",
                                        (metadata.request_id,)).fetchone()[0]
            if status == "cancelled":
                continue
            row = self._conn.execute("SELECT ref FROM native_refs WHERE record_id=?",
                                     (metadata.record_id,)).fetchone()
            if row is None:
                raise ValueError("native source metadata missing")
            ref = decode_ref(json.loads(row[0]))
            if ref.version_key in self.retired(metadata.request_id):
                continue
            current = self.reader.read(user_scope(self.user_id), ref)
            if current is not None and current.ref == ref:
                out.append(replace(metadata, content=current.content, native_ref=ref))
        return tuple(out)

    def current_ref(self, scope, ref):
        if scope != user_scope(self.user_id):
            return None
        current = self.reader.read(scope, ref)
        return None if current is None else current.ref

    def message_for(self, ref):
        current = self.reader.read(user_scope(self.user_id), ref)
        return current if current is not None and current.ref == ref else None
