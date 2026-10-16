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
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mr_mem import MemoryCore, SemanticMemoryCandidate, ThreadStatus
from mr_mem.contracts import Scope
from mr_mem.integrations.path_a_ingest import (
    BlockSignals,
    CommitSignal,
    ThreadSignal,
    apply_block_signals,
)
from mr_mem.integrations.path_a_lines import LineCard, select_known_lines
from mr_mem.integrations.path_a_read import PathAReader, RawHit
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


class CompileFailed(RetryableError):
    """The model call failed; raw text is stored and the request can be retried."""


@dataclass(frozen=True)
class AddJob:
    request: dict
    messages: tuple
    context: tuple
    known_lines: tuple
    compiled: str | None = None
    # compile window (design 4.7): (window_id, seqs of its messages, carry allowed)
    window: tuple | None = None


class _SystemClock:
    def now(self) -> datetime:
        return datetime.now(UTC)


def build_backend(spec: dict):
    if spec["type"] == "command":
        return CommandBackend(spec["argv"], timeout=spec.get("timeout", 600),
                              stream_json=spec.get("stream_json", False))
    return OpenAICompatibleBackend(
        base_url=spec["base_url"], model=spec["model"], api_key_env=spec.get("api_key_env"),
        timeout=spec.get("timeout", 600), json_mode=spec.get("json_mode", False),
        max_tokens=spec.get("max_tokens"), temperature=spec.get("temperature", 0),
        headers=spec.get("headers"), extra_body=spec.get("extra_body"),
    )


def build_embedding(spec: dict | None):
    """``null`` (BM25 only), ``fastembed`` (local ONNX, e.g. BAAI/bge-small-en-v1.5),
    ``openai`` (any OpenAI-compatible /embeddings: OpenAI, SiliconFlow bge, TEI,
    Ollama, vLLM, ...), or ``gemini``."""
    if not spec:
        return None
    from mr_mem_aml.embedding import (
        GeminiEmbedding,
        OpenAICompatibleEmbedding,
        open_fastembed,
    )

    kind = spec["type"]
    if kind == "fastembed":
        return open_fastembed(
            model=spec["model"], dimension=spec["dimension"], model_dir=spec["model_dir"],
        )
    if kind == "gemini":
        return GeminiEmbedding(
            model=spec["model"], dimension=spec["dimension"],
            api_key_env=spec.get("api_key_env", "GEMINI_API_KEY"),
        )
    if kind == "openai":
        return OpenAICompatibleEmbedding(
            base_url=spec["base_url"], model=spec["model"], dimension=spec["dimension"],
            api_key_env=spec.get("api_key_env"), send_dimensions=spec.get("send_dimensions", False),
            headers=spec.get("headers"),
        )
    if kind == "ark":
        from mr_mem_aml.embedding import ArkEmbedding

        return ArkEmbedding(
            model=spec["model"], dimension=spec["dimension"], cache_path=spec["cache_path"],
            base_url=spec.get("base_url"),
            requests_per_second=spec.get("requests_per_second", 4.0),
        )
    raise ValueError("embedding.type must be fastembed, openai, gemini or ark")


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
            self.backend, assistant_policy=config.assistant_policy,
            schema=("flat" if config.path_a_engine != "tree"
                    else "points" if config.placement == "points" else "tree"),
        )
        self._trees: dict[str, object] = {}
        self.tree_failures: list[str] = []
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
        self.compile_rejected = 0
        self.window_failures = 0
        # upward link raw record -> Blocks compiled from it
        self._blocks_by_record: dict[str, list[str]] = {}
        for memory in self.core.canonical.load_all():
            self._link_blocks(memory)
        self.raw = None
        if config.raw_base:
            from mr_mem_aml.raw_arm import RawArm, units_for

            self.raw = RawArm(self.embedding)
            for rid, messages in self.source.all_request_messages().items():
                self.raw.add(units_for(rid, messages))

    def raw_record(self, unit_id: str) -> str | None:
        """Source record (message) of a raw hit, for evaluation mapping."""
        unit = None if self.raw is None else self.raw.unit(unit_id)
        return None if unit is None else unit.record_id

    def line_report(self) -> list[dict]:
        """Every line per user with its state, for offline inspection (e.g. split lines)."""
        out = []
        users = {m.user_id for ms in self.source.all_request_messages().values() for m in ms}
        for user_id in sorted(users):
            scope = user_scope(user_id)
            lce = self.lce_for(scope)
            threads = {t.open_question: t for t in self.core.products.list_threads(scope)}
            for question, last_active in sorted(self.source.lines(user_id), key=lambda x: x[1]):
                thread = threads.get(question)
                head = None if lce is None else lce.lineage_head(question)
                out.append({
                    "user_id": user_id, "question": question, "last_active": last_active.isoformat(),
                    "thread": None if thread is None else {
                        "status": thread.status.value, "mature": thread.mature,
                        "summary": thread.working_summary,
                        "support": len(thread.handoff_memory_ids)},
                    "lce": None if head is None else {
                        "revision": head.revision_number, "content": head.content,
                        "support": len(head.supporting_memory_ids)},
                })
        return out

    def tree_report(self) -> list[dict]:
        """Tree engine: every user's Lines (nested, each step with its side of the
        terminator at the user's latest message time), matters and lone Blocks."""
        cutoff = datetime(9999, 1, 1, tzinfo=UTC)
        out = []
        by_user: dict[str, list] = {}
        for ms in self.source.all_request_messages().values():
            for m in ms:
                by_user.setdefault(m.user_id, []).append(m)
        for user_id in sorted(by_user):
            scope = user_scope(user_id)
            tree = self.tree_for(scope)
            now = max(m.occurred_at for m in by_user[user_id])

            def text(node_id, tree=tree):
                state = tree.view.state_for_node_at_cutoff(node_id, cutoff=cutoff)
                return "" if state is None else state.content

            def nest(node_id, seen, tree=tree, now=now):
                seen.add(node_id)
                return {"step": text(node_id), "side": tree.side(node_id, now), "then": [
                    nest(child, seen) for child in tree.lines.children_at(node_id, cutoff)
                    if child not in seen]}

            lines, on_lines = [], set()
            for line in tree.lines.list_lines():
                nodes = tree.view.visible_node_ids(line.line_id, cutoff=cutoff)
                on_lines.update(tree.lines.get_node(n).block_id for n in nodes)
                roots = [n for n in nodes if not tree.lines.parents_at(n, cutoff)]
                seen: set[str] = set()
                lines.append({"line_id": line.line_id, "steps": len(nodes),
                              "tree": [nest(r, seen) for r in roots]})
            matters = [{
                "thread_id": t.thread_id, "anchor_node_id": t.anchor_node_id,
                "summary": t.working_summary,
                "latest": None if t.anchor_node_id is None else text(t.anchor_node_id),
            } for t in self.core.products.list_threads(scope)]
            lone = [m.content for m in self.core.canonical.load_all()
                    if m.scope == scope and m.memory_id not in on_lines]
            out.append({"user_id": user_id, "now": now.isoformat(), "lines": lines,
                        "threads": matters, "lone_blocks": lone,
                        "relations_refused": list(self.tree_failures)})
        return out

    def _evidence_time(self, memory):
        """Newest message time the source actually gave for this Block, else None."""
        times = [ref.occurred_at for ref in memory.provenance.source_refs
                 if self.source.time_given(ref.record_id)]
        return max(times) if times else None

    def _link_blocks(self, memory) -> None:
        for ref in memory.provenance.source_refs:
            linked = self._blocks_by_record.setdefault(ref.record_id, [])
            if memory.memory_id not in linked:
                linked.append(memory.memory_id)

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

    def tree_for(self, scope: Scope):
        """LCE Thread + Path A trees of one Scope (path_a_engine="tree")."""
        key = scope.user_id or ""
        if key not in self._trees:
            from mr_mem.lce.path_a import PathATree

            self._trees[key] = PathATree(
                Path(self.config.data_dir) / "lce_tree", scope,
                canonical=self.core.canonical, products=self.core.products,
                fingerprint=self.compiler.version, evidence_time=self._evidence_time,
            )
        return self._trees[key]

    def close(self) -> None:
        for tree in self._trees.values():
            tree.close()
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
        """Synchronous Add: prepare, compile, finish in the calling thread."""
        job = self.prepare_add(payload)
        if isinstance(job, dict):
            return job
        if job.window is not None:
            return self._run_windows(job)
        try:
            compiled = self.compile_job(job)
        except CompileFailed as exc:
            self.fail_add(job, str(exc))
            raise
        return self.finish_add(job, compiled)

    def _run_windows(self, job: AddJob) -> dict:
        """Compile every window that is due. Raw text is already stored and
        searchable, so a failed window does not fail Add: it is retried by the
        next Add or by flush, and counted."""
        request, blocks = job.request, 0
        while job is not None:
            try:
                blocks += self.finish_add(job, self.compile_job(job))["blocks"]
            except CompileFailed as exc:
                self.window_failures += 1
                log.warning("window %s not compiled yet: %s", job.window[0], str(exc)[:300])
                break
            job = self._window_job(request["user_id"], final=False, request=request)
        return {**_echo(request), "blocks": blocks}

    def _window_end(self, pending, final: bool) -> int | None:
        """How many pending messages the next window takes, or None to wait."""
        soft, hard = self.config.compile_window["soft"], self.config.compile_window["hard"]
        gap = self.config.compile_window.get("gap_minutes", 120) * 60

        def breaks(prev, nxt) -> bool:
            return (nxt.session_id != prev.session_id
                    or (prev.time_given and nxt.time_given
                        and (nxt.occurred_at - prev.occurred_at).total_seconds() >= gap)
                    or (nxt.role == "user" and prev.role != "user"))

        n = len(pending)
        for i in range(soft, min(hard, n - 1) + 1):
            if breaks(pending[i - 1][1], pending[i][1]):
                return i
        if n >= hard:
            return hard
        return n if final and n else None

    def _window_job(self, user_id: str, *, final: bool, request: dict | None = None
                    ) -> AddJob | None:
        """Owner thread: the next due compile window of a user, if any."""
        request = request or {"request_id": None, "user_id": user_id, "session_id": None}
        journaled = self.source.open_window(user_id)
        if journaled is not None:  # crash between journal and apply: apply it
            wid, first, last, compiled = journaled
            messages = self.source.messages_between(user_id, first, last)
            seqs = self.source.seqs_between(user_id, first, last)
            if len(seqs) < compiled.get("kept", len(seqs)):  # its messages not all here yet
                return None
            return AddJob(request, messages, (), {}, compiled, (wid, seqs, False))
        pending = self.source.pending(user_id)
        end = self._window_end(pending, final)
        if end is None:
            return None
        seqs = tuple(seq for seq, _ in pending[:end])
        messages = tuple(m for _, m in pending[:end])
        carry = not (final and end == len(pending))  # the last window keeps its tail
        return AddJob(
            request, messages,
            self.source.context_before(messages[0], self.config.context_messages),
            self.known_steps(user_id, messages), None, (f"{user_id}:w{seqs[0]}", seqs, carry),
        )

    def prepare_add(self, payload: object) -> AddJob | dict:
        """Owner thread: persist raw text once; return a response or a compile job."""
        request = validate_add(payload)
        rid = request["request_id"]
        try:
            status = self.source.put_request(request, received_at=datetime.now(UTC))
        except RequestConflict as exc:
            raise ConflictingReplay(str(exc)) from exc
        if status == "done":
            return {**_echo(request), "replayed": True}
        messages = self.source.request_messages(rid)
        if self.raw is not None:  # raw evidence is searchable before any compile
            from mr_mem_aml.raw_arm import units_for

            self.raw.add(units_for(rid, messages))
        if self.config.compile_window is not None:
            self.source.set_done(rid)  # stored and searchable; windows compile later
            job = self._window_job(request["user_id"], final=False, request=request)
            return job if job is not None else {**_echo(request), "blocks": 0}
        compiled = self.source.compiled(rid)
        if compiled is not None:  # journaled earlier: never pay inference twice
            return AddJob(request, messages, (), (), compiled)
        return AddJob(
            request, messages,
            self.source.context_before(messages[0], self.config.context_messages),
            (self.known_steps(request["user_id"], messages)
             if self.config.path_a_engine == "tree"
             else self.known_lines(request["user_id"], messages)),
        )

    def known_steps(self, user_id: str, messages) -> dict:
        """Owner thread: threads, committed steps and denials the tree compiler sees."""
        known = self.tree_for(user_scope(user_id)).known(
            batch_text="\n".join(m.content for m in messages)[:8000],
            now=max(m.occurred_at for m in messages),
            threads_policy=self.config.line_policy, steps_policy=self.config.step_policy,
            embed=None if self.embedding is None else self._embed_cached,
        )
        return {**known.payload, "_refs": known.refs}  # refs are journaled, never prompted

    def known_lines(self, user_id: str, messages) -> tuple[dict, ...]:
        """Owner thread: the existing lines the compiler sees for this batch."""
        scope = user_scope(user_id)
        lce = self.lce_for(scope)
        threads = {t.open_question: t for t in self.core.products.list_threads(scope)}
        cards = []
        for question, last_active in self.source.lines(user_id):
            thread = threads.get(question)
            summary, settled, support = None, True, ()
            if thread is not None and thread.status is ThreadStatus.OPEN:
                summary, settled, support = (thread.working_summary, False,
                                             thread.current_support_ids)
            elif lce is not None and (head := lce.lineage_head(question)) is not None:
                summary, support = head.content, head.supporting_memory_ids
            elif thread is not None:
                summary, support = thread.working_summary, thread.current_support_ids
            blocks = [m.content for mid in support[-3:]
                      if (m := self.core.canonical.get(mid)) is not None]
            cards.append(LineCard(question, summary, settled, last_active,
                                  "\n".join([question, summary or "", *blocks])))
        policy = self.config.line_policy
        selection = select_known_lines(
            cards, batch_text="\n".join(m.content for m in messages)[:8000],
            now=max(m.occurred_at for m in messages),
            recent=policy["recent"], cued=policy["cued"], due=policy["due"],
            embed=None if self.embedding is None else self._embed_cached,
        )
        return tuple(
            {"question": c.question, "state": "settled" if c.settled else "open",
             **({"summary": c.summary} if c.summary else {})}
            for c in selection.lines
        )

    def compile_job(self, job: AddJob) -> str:
        """Any thread: the model call. Touches no store."""
        if job.compiled is not None:
            return job.compiled
        try:
            started = time.monotonic()
            if job.window is not None and job.window[2]:
                blocks, kept = self.compiler.compile(
                    job.messages, job.context, job.known_lines, window=True)
            else:
                blocks = self.compiler.compile(job.messages, job.context, job.known_lines)
                kept = len(job.messages)
            log.info("compiled %s: %d messages -> %d blocks in %.1fs",
                     job.request["request_id"] or job.window[0],
                     len(job.messages), len(blocks), time.monotonic() - started)
        except CompileError as exc:
            if self.config.skip_rejected_compiles and "rejected the request" in str(exc):
                # The provider permanently refuses this input (e.g. content
                # moderation). Raw evidence stays stored and searchable; this
                # request just yields no compiled Blocks. Counted, never silent.
                log.warning("compile rejected, kept raw only: %s: %s",
                            job.request["request_id"] or job.window[0], str(exc)[:300])
                self.compile_rejected += 1
                return self._journal(job, (), len(job.messages))
            raise CompileFailed(str(exc)) from exc
        return self._journal(job, blocks, kept)

    def _journal(self, job: AddJob, blocks, kept: int | None = None) -> dict:
        data = blocks_to_json(blocks, self.compiler.version)
        if isinstance(job.known_lines, dict):  # tree: how the prompt's refs resolve
            data["refs"] = dict(job.known_lines.get("_refs", {}))
        if job.window is not None:
            data["kept"] = len(job.messages) if kept is None else kept
        return data

    def fail_add(self, job: AddJob, error: str) -> None:
        """Owner thread: record a failed compile so the request stays resumable."""
        self.source.set_error(job.request["request_id"], error)

    def finish_add(self, job: AddJob, compiled: str) -> dict:
        """Owner thread: journal, admit, project, derive Path A."""
        if job.window is not None:
            return self._finish_window(job, compiled)
        request = job.request
        rid, scope = request["request_id"], user_scope(request["user_id"])
        if self.source.compiled(rid) is None:
            self.source.set_compiled(rid, compiled)
        blocks = blocks_from_json(self.source.compiled(rid))  # first journal entry wins
        try:
            committed = self._admit(scope, rid, blocks)
            self._project()
            self._generation += 1
            at = max(m.occurred_at for m in job.messages)
            if self.config.path_a_engine == "tree":
                commit_failures = self._grow(
                    scope, blocks, committed, self.source.compiled(rid).get("refs", {}), at)
            else:
                commit_failures = self._derive(scope, request["user_id"], blocks, committed, at)
        except RetryableError as exc:
            self.source.set_error(rid, str(exc))
            raise
        if commit_failures:
            self.source.set_error(rid, f"{commit_failures} Path A commit(s) failed")
            raise RetryableError("canonical Memory stored; Path A commit failed, retry request")
        self.source.set_done(rid)
        return {**_echo(request), "blocks": len(committed)}

    def _finish_window(self, job: AddJob, compiled: dict) -> dict:
        wid, seqs, _ = job.window
        user_id = job.request["user_id"]
        kept = compiled.get("kept", len(job.messages))
        self.source.put_window(wid, user_id, seqs[0], seqs[kept - 1], compiled)
        scope = user_scope(user_id)
        blocks = blocks_from_json(compiled)
        committed = self._admit(scope, wid, blocks)
        self._project()
        self._generation += 1
        at = max(m.occurred_at for m in job.messages[:kept])
        self._grow(scope, blocks, committed, compiled.get("refs", {}), at)
        self.source.set_window_done(wid)
        return {"window": wid, "messages": kept, "blocks": len(committed)}

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
            self._link_blocks(committed[-1])
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

    def _grow(self, scope, blocks, committed, refs: dict, at: datetime) -> int:
        """Tree engine: compiler signals -> Threads and committed Path A steps."""
        from mr_mem.lce.path_a import Attachment, BlockLines

        result = self.tree_for(scope).apply(
            [BlockLines(memory.memory_id, tuple(
                Attachment(x.thread, x.after, x.committed, x.summary)
                for x in block.lines), block.settles)
             for block, memory in zip(blocks, committed, strict=True)],
            refs, at, open_matters=self.config.placement != "points",
        )
        if result.failures:  # a refused relation never loses a Block; counted, logged
            self.tree_failures.extend(result.failures)
            log.warning("Path A relation(s) refused: %s", "; ".join(result.failures)[:500])
        return 0

    def _derive(self, scope, user_id, blocks, committed, at: datetime) -> int:
        # ``at`` is message time: line activity must replay the same way
        result = apply_block_signals(
            canonical=self.core.canonical, products=self.core.products,
            lce=self.lce_for(scope), scope=scope, at=at,
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
        for question in result.questions:
            self.source.remember_question(user_id, question, at)
        if result.thread_failed:
            self.thread_failures += 1
        return result.promotion_failures

    def flush(self) -> dict:
        """Rescan every user Scope and retry pending mature-Thread handoffs."""
        if self.config.compile_window is not None:  # compile what is left, tail included
            windows = 0
            for user_id in self.source.users_with_pending():
                while (job := self._window_job(user_id, final=True)) is not None:
                    self.finish_add(job, self.compile_job(job))
                    windows += 1
            return {"status": "ok", "windows": windows}
        if self.config.path_a_engine == "tree":  # commits happen inside Add; nothing pends
            return {"status": "ok", "scopes": 0}
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
        text = query.strip()
        if self.config.append_options and options:
            text = text + "\n" + "\n".join(str(o) for o in options)
        raw_search = None
        if self.raw is not None:
            raw, source = self.raw, self.source

            def raw_search(scope, q, limit):
                return [RawHit(u.unit_id, u.text, score,
                               u.occurred_at if source.time_given(u.record_id) else None,
                               (u.record_id,))
                        for u, score in raw.search(scope.user_id or "", q, limit)]

        reader = PathAReader(
            canonical_path=self.canonical_path,
            retrieval=None if raw_search is not None else self._retrieval(),
            lce=(self.lce_for(scope)
                 if selected == "path_a" and self.config.path_a_engine == "flat" else None),
            policy=self.config.read_policy,
            render_source=self.source.render,
            embed=None if self.embedding is None else self._embed_cached,
            raw_search=raw_search,
            blocks_for_record=lambda record_id: self._blocks_by_record.get(record_id, ()),
            evidence_time=self._evidence_time,
        )
        items = reader.search(scope, text[:4096], limit=top_k, mode=selected)
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
