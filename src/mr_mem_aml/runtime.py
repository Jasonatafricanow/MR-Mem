"""Add/Search pipeline: raw source -> semantic compile -> canonical -> Thread/LCE.

Add (one transaction boundary per stage, replay-safe end to end):

    AMLSourceStore (immutable raw)            idempotent by request_id
      -> SemanticCompiler (one model call)    journaled; a retry reuses it
      -> canonical admission                  idempotent by semantic identity
      -> dense projection (optional)          rebuildable index
      -> Thread maintenance + Path A          stateless rescan (Mind-Runtime)
      -> first-appearance commits             idempotent by processing input

HTTP 200 is returned only after the request is persisted and searchable.
A failed model or LCE step never rolls back canonical Memory; the request is
left resumable and the caller gets a retryable error.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from mr_mem import MemoryCore, SemanticMemoryCandidate
from mr_mem.contracts import Scope
from mr_mem.integrations.path_a_ingest import (
    BlockSignals,
    CommitSignal,
    ThreadSignal,
    apply_block_signals,
)
from mr_mem.integrations.path_a_read import PathAReader
from mr_mem.memory.providers.bm25 import BM25RetrievalProvider
from mr_mem.memory.providers.hybrid import (
    HybridRRFProvider,
    HyDEAugmentedProvider,
    HyDEFallbackProvider,
    PromptHyDEExpander,
    RetrievalArm,
)
from mr_mem.memory.retrieval import MemoryRetrievalService
from mr_mem_aml.compiler import (
    PROMPT_VERSION,
    CommandBackend,
    CompileError,
    OpenAICompatibleBackend,
    SemanticCompiler,
    blocks_from_json,
    blocks_to_json,
)
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.source_store import AmlSourceStore, RequestConflict, user_scope

log = logging.getLogger("mr_mem_aml")
ORIGIN = "mr-mem-aml"


class RetryableError(RuntimeError):
    """The request is persisted but not yet searchable; retry the same request."""


class InvalidRequest(ValueError):
    pass


class ConflictingReplay(InvalidRequest):
    """Same request_id, different payload: permanent, never worth retrying."""


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def build_backend(spec: dict):
    if spec["type"] == "command":
        return CommandBackend(spec["argv"], timeout=spec.get("timeout", 600))
    return OpenAICompatibleBackend(
        base_url=spec["base_url"], model=spec["model"], api_key_env=spec["api_key_env"],
        timeout=spec.get("timeout", 600),
    )


def build_embedding(spec: dict | None):
    if not spec:
        return None
    from mr_mem_aml.embedding import GeminiEmbedding, OpenAICompatibleEmbedding

    if spec["type"] == "gemini":
        return GeminiEmbedding(
            model=spec["model"], dimension=spec["dimension"],
            api_key_env=spec.get("api_key_env", "GEMINI_API_KEY"),
        )
    if spec["type"] == "openai":
        return OpenAICompatibleEmbedding(
            base_url=spec["base_url"], model=spec["model"], dimension=spec["dimension"],
            api_key_env=spec["api_key_env"],
        )
    raise ValueError("embedding.type must be 'gemini' or 'openai'")


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal: keep ordered text parts only
        parts = [
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    raise InvalidRequest("message content must be a string or a list of parts")


def validate_add(payload: object) -> dict:
    if not isinstance(payload, dict):
        raise InvalidRequest("JSON object required")
    for name in ("request_id", "user_id", "session_id"):
        if not isinstance(payload.get(name), str) or not payload[name].strip():
            raise InvalidRequest(f"{name} must be a nonempty string")
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequest("messages must be a nonempty list")
    clean = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise InvalidRequest("each message needs a role")
        item = {"role": message["role"], "content": _text_content(message.get("content"))}
        if message.get("timestamp") is not None:
            if isinstance(message["timestamp"], bool) or not isinstance(
                message["timestamp"], (int, float)
            ):
                raise InvalidRequest("timestamp must be Unix milliseconds")
            item["timestamp"] = message["timestamp"]
        clean.append(item)
    return {
        "request_id": payload["request_id"], "user_id": payload["user_id"],
        "session_id": payload["session_id"], "messages": clean,
    }


def _echo(request: dict) -> dict:
    """AML Add contract: success=true plus the three identifiers echoed exactly."""
    return {
        "success": True,
        "request_id": request["request_id"],
        "user_id": request["user_id"],
        "session_id": request["session_id"],
    }


class MemoryRuntime:
    """Single-writer composition; one instance per data directory."""

    def __init__(self, config: RuntimeConfig, *, compiler: SemanticCompiler | None = None):
        self.config = config
        data = Path(config.data_dir)
        data.mkdir(parents=True, exist_ok=True)
        self.canonical_path = data / "memory.sqlite"
        self.core = MemoryCore(self.canonical_path)
        self.source = AmlSourceStore(data / "aml_source.sqlite")
        self.backend = None if compiler is not None else build_backend(config.compiler)
        self.compiler = compiler or SemanticCompiler(
            self.backend, assistant_policy=config.assistant_policy
        )
        self.embedding = build_embedding(config.embedding)
        self.index = None
        if self.embedding is not None:
            from mr_mem.memory.providers.qdrant import open_qdrant_index

            self.index = open_qdrant_index(
                index_root=data / "index", canonical_path=self.canonical_path,
                namespace="aml", embedding=self.embedding, create=True,
            )
            self.core.canonical.projection_queue().register_projection_target(self.index.target)
        self._lce: dict[str, object] = {}
        self._bm25: tuple[int, BM25RetrievalProvider] | None = None
        self._generation = 0
        self._vectors: dict[str, tuple[float, ...]] = {}
        self.thread_failures = 0

    def _embed_cached(self, text: str) -> tuple[float, ...]:
        """Projection texts are re-ranked per query; embed each text once."""
        vector = self._vectors.get(text)
        if vector is None:
            assert self.embedding is not None
            vector = tuple(self.embedding.embed(text))
            if len(self._vectors) > 50_000:
                self._vectors.clear()
            self._vectors[text] = vector
        return vector

    def close(self) -> None:
        for session in self._lce.values():
            session.close()
        if self.index is not None:
            self.index.close()
        self.source.close()
        self.core.close()

    # -- composition ----------------------------------------------------

    def lce_for(self, scope: Scope):
        if not self.config.lce_enabled:
            return None
        key = scope.user_id or ""
        if key not in self._lce:
            from mr_mem.integrations.lce_path_a import LcePathA

            self._lce[key] = LcePathA(
                canonical_path=self.canonical_path,
                lce_root=Path(self.config.data_dir) / "lce", scope=scope,
            )
        return self._lce[key]

    def _retrieval(self) -> MemoryRetrievalService:
        if self._bm25 is None or self._bm25[0] != self._generation:
            self._bm25 = (self._generation, BM25RetrievalProvider(self.core.load_all()))
        provider = self._bm25[1]
        if self.index is not None:
            provider = HybridRRFProvider((
                RetrievalArm("bm25", provider),
                RetrievalArm("dense", self.index.retrieval()),
            ))
        if self.config.hyde != "off":
            backend = self.backend
            if backend is None:
                raise ValueError("HyDE needs a configured completion backend")
            expander = PromptHyDEExpander(backend.complete)
            provider = (
                HyDEFallbackProvider(provider, expander)
                if self.config.hyde == "fallback"
                else HyDEAugmentedProvider(provider, expander)
            )
        return MemoryRetrievalService(store=self.core.canonical, provider=provider)

    # -- Add ------------------------------------------------------------

    def add(self, payload: object) -> dict:
        request = validate_add(payload)
        rid, scope = request["request_id"], user_scope(request["user_id"])
        try:
            status = self.source.put_request(request, received_at=datetime.now(UTC))
        except RequestConflict as exc:
            raise ConflictingReplay(str(exc)) from exc
        if status == "done":
            return {**_echo(request), "replayed": True}
        messages = self.source.request_messages(rid)
        compiled = self.source.compiled(rid)
        if compiled is None:
            try:
                blocks = self.compiler.compile(
                    messages,
                    self.source.context_before(messages[0], self.config.context_messages),
                    self.source.questions(request["user_id"]),
                )
            except CompileError as exc:
                self.source.set_error(rid, str(exc))
                raise RetryableError(f"semantic compilation failed: {exc}") from exc
            compiled = blocks_to_json(blocks)
            self.source.set_compiled(rid, compiled)
            compiled = self.source.compiled(rid)  # first journal entry wins
        blocks = blocks_from_json(compiled)
        try:
            committed = self._admit(scope, rid, blocks)
            self._project()
            self._generation += 1
            commit_failures = self._derive(scope, request["user_id"], blocks, committed)
        except RetryableError as exc:
            self.source.set_error(rid, str(exc))
            raise
        if commit_failures:
            self.source.set_error(rid, f"{commit_failures} Path A commit(s) failed")
            raise RetryableError("canonical Memory stored; Path A commit failed, retry request")
        self.source.set_done(rid)
        return {**_echo(request), "blocks": len(committed)}

    def _admit(self, scope, rid, blocks) -> list:
        admission = self.core.semantic_admission(
            sources=self.source, clock=_SystemClock(), origin_runtime_id=ORIGIN
        )
        committed = []
        for index, block in enumerate(blocks):
            refs = []
            for record_id in block.source_record_ids:
                message = self.source.message_by_record(record_id)
                if message is None:
                    raise RetryableError(f"raw record vanished: {record_id}")
                refs.append(message.ref)
            refs.sort(key=lambda ref: (ref.occurred_at, ref.record_id))
            committed.append(admission.admit(SemanticMemoryCandidate(
                semantic_id=f"aml:{rid}:{index}", scope=scope, content=block.content,
                source_refs=tuple(refs), compiler_version=PROMPT_VERSION,
            )))
        return committed

    def _project(self) -> None:
        if self.index is None:
            return
        from mr_mem.memory.projection import ProjectionWorker

        worker = ProjectionWorker(
            self.core.canonical.projection_queue(), self.index.writer(), target=self.index.target
        )
        while True:
            succeeded, failed = worker.run_once(limit=64)
            if failed:
                raise RetryableError("dense projection failed; request not yet searchable")
            if not succeeded:
                return

    def _derive(self, scope, user_id, blocks, committed) -> int:
        result = apply_block_signals(
            canonical=self.core.canonical, products=self.core.products,
            lce=self.lce_for(scope), scope=scope, at=datetime.now(UTC),
            blocks=tuple(
                BlockSignals(
                    memory.memory_id, block.content,
                    None if block.thread is None else ThreadSignal(
                        block.thread.action, block.thread.question,
                        block.thread.summary, block.thread.mature,
                    ),
                    None if block.commit is None else CommitSignal(
                        block.commit.question, block.commit.summary,
                    ),
                )
                for block, memory in zip(blocks, committed, strict=True)
            ),
        )
        now = datetime.now(UTC)
        for question in result.questions:
            self.source.remember_question(user_id, question, now)
        if result.thread_failed:
            self.thread_failures += 1
        return result.promotion_failures

    def flush(self) -> dict:
        """Rescan every user Scope and retry pending mature-Thread handoffs."""
        scopes = {m.scope for m in self.core.load_all()}
        handled = 0
        for scope in scopes:
            lce = self.lce_for(scope)
            if lce is None:
                continue
            # An empty batch is the stateless rescan: mature Threads are handed off.
            result = apply_block_signals(
                canonical=self.core.canonical, products=self.core.products,
                lce=lce, scope=scope, blocks=(), at=datetime.now(UTC),
            )
            if result.thread_failed:
                self.thread_failures += 1
            handled += 1
        return {"status": "ok", "scopes": handled}

    # -- Search ---------------------------------------------------------

    def search(self, payload: object, *, mode: str | None = None) -> dict:
        if not isinstance(payload, dict):
            raise InvalidRequest("JSON object required")
        query, user_id = payload.get("query"), payload.get("user_id")
        if not isinstance(query, str) or not query.strip():
            raise InvalidRequest("query must be a nonempty string")
        if not isinstance(user_id, str) or not user_id.strip():
            raise InvalidRequest("user_id must be a nonempty string")
        options = payload.get("options")
        if options is not None and not isinstance(options, list):
            raise InvalidRequest("options must be a list when present")
        top_k = payload.get("top_k", 100)
        if type(top_k) is not int or not 1 <= top_k <= 100:
            raise InvalidRequest("top_k must be an integer in [1, 100]")
        selected = mode or self.config.search_mode
        if selected not in ("path_a", "baseline"):
            raise InvalidRequest("mode must be path_a or baseline")
        scope = user_scope(user_id)
        reader = PathAReader(
            canonical_path=self.canonical_path,
            retrieval=self._retrieval(),
            lce=self.lce_for(scope) if selected == "path_a" else None,
            policy=self.config.read_policy,
            render_source=self.source.render,
            embed=None if self.embedding is None else self._embed_cached,
        )
        items = reader.search(scope, query[:4096], limit=top_k, mode=selected)
        return {
            "data": [
                {
                    "id": item.item_id,
                    "content": item.content,
                    "score": item.score,
                    **({"created_at": item.occurred_at.isoformat()} if item.occurred_at else {}),
                }
                for item in items[:top_k]
                if item.content.strip()
            ]
        }
