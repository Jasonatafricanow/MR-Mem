"""Single-writer native ingestion using the existing compilation/admission pipeline."""

from datetime import UTC, datetime
from uuid import uuid4

from mr_mem_aml.native_source import NativeSourceStore
from mr_mem_aml.runtime import MemoryRuntime
from mr_mem_aml.source_store import user_scope


class NativeMemory:
    def __init__(self, config, *, user_id, reader, compiler=None):
        if config.compile_window is not None or config.raw_base:
            raise ValueError("native batches require compile_window=null and raw_base=false")
        self.user_id, self.reader = user_id, reader
        self.scope = user_scope(user_id)
        self.source = NativeSourceStore(config.data_dir / "native_source.sqlite",
                                        user_id=user_id, reader=reader)
        try:
            self.runtime = MemoryRuntime(config, compiler=compiler, source=self.source)
        except Exception:
            self.source.close()
            raise

    def close(self):
        self.runtime.close()

    def _payload(self, rid, messages):
        return {"request_id": rid, "user_id": self.user_id,
                "session_id": messages[0].session_id, "messages": [
                    {"role": m.role, "content": m.content,
                     "timestamp": m.occurred_at.timestamp() * 1000} for m in messages]}

    def synchronize(self, session_id, messages, *, compile_pending=False):
        """A complete host-verified session snapshot; errors must not become deletion."""
        messages = tuple(messages)
        if any(m.user_id != self.user_id or m.session_id != session_id for m in messages):
            raise ValueError("native snapshot exceeds owner/session binding")
        if len({m.ref.source_key for m in messages}) != len(messages):
            raise ValueError("duplicate native source identity")
        for message in messages:
            if self.source.current_ref(self.scope, message.ref) != message.ref:
                raise ValueError("native snapshot is stale")
        covered, invalidated, cancelled = set(), set(), 0
        for rid, status in self.source.jobs(session_id):
            if status == "cancelled":
                continue
            refs = self.source.refs(rid)
            stale = False
            for ref in refs:
                current = self.source.current_ref(self.scope, ref)
                if current != ref:
                    stale = True
                    invalidated.update(self.runtime.core.canonical.invalidate_source(
                        self.scope, current or ref, deleted=current is None))
            if stale and status != "done":
                # A partially applied batch must not produce duplicates on reconstruction.
                ids = tuple(m.memory_id for m in self.runtime.core.load_all()
                            if m.scope == self.scope
                            and ("native_ingest_batch", rid) in m.attributes)
                invalidated.update(self.runtime.core.canonical.invalidate_batch(self.scope, ids))
                self.source.cancel(rid)
                cancelled += 1
            else:
                covered.update(ref.version_key for ref in refs)
        fresh = tuple(m for m in messages if m.ref.version_key not in covered)
        if fresh:
            rid = "native-" + uuid4().hex
            self.source.bind(rid, fresh)
            self.source.put_request(self._payload(rid, fresh), received_at=datetime.now(UTC))
        applied, pending = 0, 0
        for rid, status in self.source.jobs(session_id):
            if status in ("done", "cancelled"):
                continue
            pending += 1
            if not compile_pending:
                continue
            # Frozen output can be replayed even while the algorithm port is disabled.
            if (not self.runtime.compiler_configured and self.source.compiled(rid) is None):
                continue
            batch = tuple(self.reader.read(self.scope, ref) for ref in self.source.refs(rid))
            if any(m is None or m.ref != ref
                   for m, ref in zip(batch, self.source.refs(rid), strict=True)):
                raise ValueError("native source changed; synchronize again before retry")
            self.source.bind(rid, batch)
            self.runtime.add(self._payload(rid, batch))
            applied += 1
            pending -= 1
        self.runtime._project()
        return {"applied_batches": applied, "pending_batches": pending,
                "cancelled_batches": cancelled, "invalidated_memories": len(invalidated),
                "status": ("compiler_not_configured" if pending
                           and not self.runtime.compiler_configured else "pending" if pending
                           else "ready")}
