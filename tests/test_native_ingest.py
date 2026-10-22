"""Host-owned source -> existing journal/admission/tree, with no raw mirror."""

import hashlib
import sqlite3
from datetime import UTC, datetime

import pytest

from mr_mem import MemoryLifecycle, SourceRef
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.native_ingest import NativeMemory
from mr_mem_aml.native_source import NativeSourceStore
from mr_mem_aml.source_store import RawMessage, user_scope
from tests.test_line_read import AcceptedGraphCompiler


class Host:
    identity = "test-host-owner-u"

    def __init__(self):
        self.messages = {}

    def put(self, record, text):
        at = datetime(2026, 10, 1, tzinfo=UTC)
        fingerprint = hashlib.sha256(text.encode()).hexdigest()
        ref = SourceRef("native", "s", record, at, fingerprint=fingerprint)
        message = RawMessage(record, "u", "s", "", int(record), "user", text,
                             at, fingerprint, native_ref=ref)
        self.messages[record] = message
        return message

    def read(self, scope, ref):
        return self.messages.get(ref.record_id) if scope == user_scope("u") else None


def test_native_metadata_journal_rehydrates_exact_version_only(tmp_path):
    host = Host()
    message = host.put("1", "RAW-ONLY: prefer reproducible tests")
    path = tmp_path / "native.sqlite"
    source = NativeSourceStore(path, user_id="u", reader=host)
    payload = {"request_id": "r", "user_id": "u", "session_id": "s", "messages": []}
    source.bind("r", (message,))
    source.put_request(payload, received_at=datetime.now(UTC))
    assert source.request_messages("r")[0].ref == message.ref
    source.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT content FROM messages").fetchall() == [("",)]
    assert b"RAW-ONLY" not in path.read_bytes()
    source = NativeSourceStore(path, user_id="u", reader=host)
    try:
        assert source.request_messages("r")[0].content == message.content
        assert source.current_ref(user_scope("other"), message.ref) is None
        revised = host.put("1", "Edited source")
        assert source.request_messages("r") == ()
        assert source.current_ref(user_scope("u"), message.ref) == revised.ref
    finally:
        source.close()


@pytest.mark.parametrize("change", ["unchanged", "edit", "delete", "partial_edit"])
def test_native_pending_restart_revision_and_partial_admission(tmp_path, change, monkeypatch):
    host = Host()
    host.put("1", "Considering a laptop")
    host.put("2", "Repair succeeded; purchase cancelled")
    config = RuntimeConfig(tmp_path, {"type": "disabled"}, path_a_engine="tree",
                           placement="points")
    native = NativeMemory(config, user_id="u", reader=host)
    result = native.synchronize("s", host.messages.values(), compile_pending=True)
    assert result["status"] == "compiler_not_configured"
    assert not native.runtime.core.load_all()
    native.close()
    native = NativeMemory(config, user_id="u", reader=host, compiler=AcceptedGraphCompiler())
    if change == "partial_edit":
        def crash():
            raise RuntimeError("crash after canonical admission")
        with monkeypatch.context() as patch:
            patch.setattr(native.runtime, "_project", crash)
            with pytest.raises(RuntimeError, match="crash"):
                native.synchronize("s", host.messages.values(), compile_pending=True)
    else:
        assert native.synchronize("s", host.messages.values(), compile_pending=True)[
            "applied_batches"] == 1
    original = {m.memory_id for m in native.runtime.core.load_all()}
    native.close()
    if "edit" in change:
        host.put("1", "Considering a desktop computer")
    elif change == "delete":
        del host.messages["1"]
    native = NativeMemory(config, user_id="u", reader=host, compiler=AcceptedGraphCompiler())
    try:
        result = native.synchronize("s", host.messages.values(), compile_pending=True)
        assert result["pending_batches"] == 0
        memories = native.runtime.core.load_all()
        active = [m for m in memories if m.lifecycle is MemoryLifecycle.ACTIVE]
        assert len(active) == len(host.messages)
        assert {m.content for m in active} == {m.content for m in host.messages.values()}
        if change == "unchanged":
            assert {m.memory_id for m in active} == original
        if change == "partial_edit":
            assert all(m.lifecycle is MemoryLifecycle.INVALIDATED for m in memories
                       if m.memory_id in original)
        assert native.synchronize("s", host.messages.values(), compile_pending=True)[
            "applied_batches"] == 0
    finally:
        native.close()
