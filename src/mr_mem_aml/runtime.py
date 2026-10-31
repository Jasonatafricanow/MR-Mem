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

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from mr_mem import MemoryCore, MemoryLifecycle, SemanticMemoryCandidate, ThreadStatus
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
from mr_mem_aml.entries import CONTEXT_KEEP, is_noise
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
        return CommandBackend(
            spec["argv"],
            timeout=spec.get("timeout", 600),
            stream_json=spec.get("stream_json", False),
        )
    return OpenAICompatibleBackend(
        base_url=spec["base_url"],
        model=spec["model"],
        api_key_env=spec.get("api_key_env"),
        timeout=spec.get("timeout", 600),
        json_mode=spec.get("json_mode", False),
        max_tokens=spec.get("max_tokens"),
        temperature=spec.get("temperature", 0),
        headers=spec.get("headers"),
        extra_body=spec.get("extra_body"),
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
            model=spec["model"],
            dimension=spec["dimension"],
            model_dir=spec["model_dir"],
        )
    if kind == "gemini":
        return GeminiEmbedding(
            model=spec["model"],
            dimension=spec["dimension"],
            api_key_env=spec.get("api_key_env", "GEMINI_API_KEY"),
        )
    if kind == "openai":
        return OpenAICompatibleEmbedding(
            base_url=spec["base_url"],
            model=spec["model"],
            dimension=spec["dimension"],
            api_key_env=spec.get("api_key_env"),
            send_dimensions=spec.get("send_dimensions", False),
            headers=spec.get("headers"),
            cache_path=spec.get("cache_path"),
        )
    if kind == "ark":
        from mr_mem_aml.embedding import ArkEmbedding

        return ArkEmbedding(
            model=spec["model"],
            dimension=spec["dimension"],
            cache_path=spec["cache_path"],
            base_url=spec.get("base_url"),
            requests_per_second=spec.get("requests_per_second", 4.0),
        )
    raise ValueError("embedding.type must be fastembed, openai, gemini or ark")


def _text_content(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal: keep ordered text parts only
        parts = [
            p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
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
        "request_id": payload["request_id"],
        "user_id": payload["user_id"],
        "session_id": payload["session_id"],
        "messages": clean,
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
            self.backend,
            assistant_policy=config.assistant_policy,
            schema=(
                "flat"
                if config.path_a_engine != "tree"
                else config.placement
                if config.placement in ("points", "entries", "moves")
                else "tree"
            ),
            scene=config.scene,
            step1=config.step1,
        )
        self._trees: dict[str, object] = {}
        self.tree_failures: list[str] = []
        self.embedding = build_embedding(config.embedding)
        self.index = None
        if self.embedding is not None:
            from mr_mem.memory.providers.qdrant import open_qdrant_index

            self.index = open_qdrant_index(
                index_root=data / "index",
                canonical_path=self.canonical_path,
                namespace="aml",
                embedding=self.embedding,
                create=True,
            )
            self.core.canonical.projection_queue().register_projection_target(self.index.target)
        self._lce: dict[str, object] = {}
        self._language: dict[str, str] = {}  # placement "moves": decided once per user
        self._bm25: tuple[int, BM25RetrievalProvider] | None = None
        self._generation = 0
        self._vectors: dict[str, tuple[float, ...]] = {}
        self.thread_failures = 0
        self.compile_rejected = 0
        self._hyde = None  # the HyDE expander, built on first use
        from mr_mem_aml.decision import DecisionBypass

        # the readout-time bypass (decision.py): RuntimeHost.search calls it off the owner thread
        self.decision = DecisionBypass(config.decision) if config.decision else None
        self.compile_failed_kept_raw = (
            0  # compiles that failed after retries; their raw stays searchable
        )
        self.window_failures = 0
        # work blocks that run past the person's feedback into the assistant's next
        # message (design 4.1); counted and logged, never refused
        self.past_feedback: list[str] = []
        # upward link raw record -> Blocks compiled from it
        self._blocks_by_record: dict[str, list[str]] = {}
        self._standing: dict[
            object, list[str]
        ] = {}  # scope -> standing instructions and preferences
        # downward link Block -> {raw record: the paragraphs it points to} where it points to a part
        self._spans: dict[str, dict[str, tuple[int, ...]]] = {}
        # Block -> {"own": {record: items | None}, "parts": [...]}: what its turns and each part
        # point to
        self._pointers: dict[str, dict] = {}
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
                out.append(
                    {
                        "user_id": user_id,
                        "question": question,
                        "last_active": last_active.isoformat(),
                        "thread": None
                        if thread is None
                        else {
                            "status": thread.status.value,
                            "mature": thread.mature,
                            "summary": thread.working_summary,
                            "support": len(thread.handoff_memory_ids),
                        },
                        "lce": None
                        if head is None
                        else {
                            "revision": head.revision_number,
                            "content": head.content,
                            "support": len(head.supporting_memory_ids),
                        },
                    }
                )
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
                return {
                    "step": text(node_id),
                    "side": tree.side(node_id, now),
                    "then": [
                        nest(child, seen)
                        for child in tree.lines.children_at(node_id, cutoff)
                        if child not in seen
                    ],
                }

            lines, on_lines = [], set()
            for line in tree.lines.list_lines():
                nodes = tree.view.visible_node_ids(line.line_id, cutoff=cutoff)
                on_lines.update(tree.lines.get_node(n).block_id for n in nodes)
                roots = [n for n in nodes if not tree.lines.parents_at(n, cutoff)]
                seen: set[str] = set()
                lines.append(
                    {
                        "line_id": line.line_id,
                        "steps": len(nodes),
                        "tree": [nest(r, seen) for r in roots],
                    }
                )
            matters = [
                {
                    "thread_id": t.thread_id,
                    "anchor_node_id": t.anchor_node_id,
                    "summary": t.working_summary,
                    "latest": None if t.anchor_node_id is None else text(t.anchor_node_id),
                }
                for t in self.core.products.list_threads(scope)
            ]
            lone = [
                m.content
                for m in self.core.canonical.load_all()
                if m.scope == scope and m.memory_id not in on_lines
            ]
            out.append(
                {
                    "user_id": user_id,
                    "now": now.isoformat(),
                    "lines": lines,
                    "threads": matters,
                    "lone_blocks": lone,
                    "relations_refused": list(self.tree_failures),
                    "past_feedback": list(self.past_feedback),
                }
            )
        return out

    def _evidence_time(self, memory):
        """Newest message time the source actually gave for this Block, else None."""
        times = [
            ref.occurred_at
            for ref in memory.provenance.source_refs
            if self.source.time_given(ref.record_id)
        ]
        return max(times) if times else None

    def _link_blocks(self, memory) -> None:
        for ref in memory.provenance.source_refs:
            linked = self._blocks_by_record.setdefault(ref.record_id, [])
            if memory.memory_id not in linked:
                linked.append(memory.memory_id)
        standing = self._standing.setdefault(memory.scope, [])
        if ("standing", "true") in memory.attributes and memory.memory_id not in standing:
            standing.append(memory.memory_id)
        pointers = dict(memory.attributes).get("pointers")
        if pointers:
            self._pointers[memory.memory_id] = json.loads(pointers)
        spans = dict(memory.attributes).get("spans")
        if spans:
            self._spans[memory.memory_id] = {
                r: tuple(tuple(x) if isinstance(x, list) else x for x in p)
                for r, p in json.loads(spans).items()
            }

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

    def _embed_texts(self, texts: list[str]) -> list[tuple[float, ...]]:
        """Any thread: vectors of texts (the store is not touched)."""
        assert self.embedding is not None
        return [tuple(self.embedding.embed(t)) for t in texts]

    def tree_for(self, scope: Scope):
        """LCE Thread + Path A trees of one Scope (path_a_engine="tree")."""
        key = scope.user_id or ""
        if key not in self._trees:
            from mr_mem.lce.path_a import PathATree

            self._trees[key] = PathATree(
                Path(self.config.data_dir) / "lce_tree",
                scope,
                canonical=self.core.canonical,
                products=self.core.products,
                fingerprint=self.compiler.version,
                evidence_time=self._evidence_time,
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
                lce_root=Path(self.config.data_dir) / "lce",
                scope=scope,
            )
        return self._lce[key]

    def _retrieval(self) -> MemoryRetrievalService:
        if self._bm25 is None or self._bm25[0] != self._generation:
            self._bm25 = (self._generation, BM25RetrievalProvider(self.core.load_all()))
        provider = self._bm25[1]
        if self.index is not None:
            provider = HybridRRFProvider(
                (
                    RetrievalArm("bm25", provider),
                    RetrievalArm("dense", self.index.retrieval()),
                )
            )
        if self.config.hyde != "off":
            backend = (
                build_backend(self.config.hyde_backend)
                if self.config.hyde_backend
                else self.backend
            )
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
        chars = self.config.compile_window.get("measure") == "chars"
        gap = self.config.compile_window.get("gap_minutes", 120) * 60
        work = self.config.scene == "work"

        def turn(prev, nxt) -> bool:
            if work:  # after the person's feedback, before the assistant's next report
                return prev.role == "user" and nxt.role != "user"
            return nxt.role == "user" and prev.role != "user"  # the person speaks again

        def breaks(prev, nxt) -> bool:
            return (
                nxt.session_id != prev.session_id
                or (
                    prev.time_given
                    and nxt.time_given
                    and (nxt.occurred_at - prev.occurred_at).total_seconds() >= gap
                )
                or turn(prev, nxt)
            )

        # size[i]: what the first i pending messages count toward the caps (messages or
        # characters); entries placement drops background notices, so they do not count
        size = [0]
        for _, m in pending:
            counted = self.config.placement not in ("entries", "moves") or not is_noise(
                m.role, m.content
            )
            size.append(size[-1] + (len(m.content) if chars else 1) * int(counted))
        n = len(pending)
        for i in range(1, n):
            if size[i] > hard:
                break
            if size[i] >= soft and breaks(pending[i - 1][1], pending[i][1]):
                return i
        if size[n] >= hard:  # the most that fits under hard, at least one message
            return max(1, max(i for i in range(n + 1) if size[i] <= hard))
        return n if final and n else None

    def _window_job(
        self, user_id: str, *, final: bool, request: dict | None = None
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
            request,
            messages,
            self._context(messages[0]),
            (
                {"language": self.language_of(user_id)}
                if self.config.placement == "moves"
                else self.known_steps(user_id, messages)
            ),
            None,
            (f"{user_id}:w{seqs[0]}", seqs, carry),
        )

    def language_of(self, user_id: str) -> str:
        """Placement "moves": the person's language, decided once per user from the
        person's own messages stored so far, then kept (the prompt follows it)."""
        if user_id not in self._language:
            from mr_mem_aml.moves import person_language

            mine = [
                m.content
                for m in self.source.messages_between(user_id, 0, 2**62)
                if m.role == "user" and not is_noise(m.role, m.content)
            ]
            self._language[user_id] = person_language(mine)
        return self._language[user_id]

    def _context(self, first) -> tuple:
        """Earlier messages of the session shown as CONTEXT. Entries placement drops
        background notices and keeps the last real messages, so it reaches further
        back until it has them or the session starts."""
        limit = self.config.context_messages
        rows = self.source.context_before(first, limit)
        if self.config.placement != "entries":
            return rows
        while (
            0 < limit < 1000
            and len(rows) == limit
            and sum(not is_noise(m.role, m.content) for m in rows) < CONTEXT_KEEP
        ):
            limit *= 2
            rows = self.source.context_before(first, limit)
        return rows

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
            request,
            messages,
            self._context(messages[0]),
            (
                self.known_steps(request["user_id"], messages)
                if self.config.path_a_engine == "tree"
                else self.known_lines(request["user_id"], messages)
            ),
        )

    def known_steps(self, user_id: str, messages) -> dict:
        """Owner thread: threads, committed steps and denials the tree compiler sees."""
        known = self.tree_for(user_scope(user_id)).known(
            batch_text="\n".join(m.content for m in messages)[:8000],
            now=max(m.occurred_at for m in messages),
            threads_policy=self.config.line_policy,
            steps_policy=self.config.step_policy,
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
                summary, settled, support = (
                    thread.working_summary,
                    False,
                    thread.current_support_ids,
                )
            elif lce is not None and (head := lce.lineage_head(question)) is not None:
                summary, support = head.content, head.supporting_memory_ids
            elif thread is not None:
                summary, support = thread.working_summary, thread.current_support_ids
            blocks = [
                m.content for mid in support[-3:] if (m := self.core.canonical.get(mid)) is not None
            ]
            cards.append(
                LineCard(
                    question,
                    summary,
                    settled,
                    last_active,
                    "\n".join([question, summary or "", *blocks]),
                )
            )
        policy = self.config.line_policy
        selection = select_known_lines(
            cards,
            batch_text="\n".join(m.content for m in messages)[:8000],
            now=max(m.occurred_at for m in messages),
            recent=policy["recent"],
            cued=policy["cued"],
            due=policy["due"],
            embed=None if self.embedding is None else self._embed_cached,
        )
        return tuple(
            {
                "question": c.question,
                "state": "settled" if c.settled else "open",
                **({"summary": c.summary} if c.summary else {}),
            }
            for c in selection.lines
        )

    def compile_job(self, job: AddJob) -> str:
        """Any thread: the model call. Touches no store."""
        if job.compiled is not None:
            return job.compiled
        trace: dict = {}  # entries placement: pieces, entries and audit for the journal
        try:
            started = time.monotonic()
            if job.window is not None and (job.window[2] or self.config.placement == "moves"):
                # moves: even the last window may stop where its SBs close; the flush goes on
                blocks, kept = self.compiler.compile(
                    job.messages,
                    job.context,
                    job.known_lines,
                    window=True,
                    trace=trace,
                    tail=job.window[2],
                )
            else:
                blocks = self.compiler.compile(
                    job.messages, job.context, job.known_lines, trace=trace
                )
                kept = len(job.messages)
            log.info(
                "compiled %s: %d messages -> %d blocks in %.1fs",
                job.request["request_id"] or job.window[0],
                len(job.messages),
                len(blocks),
                time.monotonic() - started,
            )
            if trace and "moves" in trace:
                log.info(
                    "moves %s: %d SBs, %d of them the person's",
                    job.request["request_id"] or job.window[0],
                    len(blocks),
                    sum(bool(m["move"]) for m in trace["moves"]),
                )
            elif trace:
                audit = trace["audit"]
                log.info(
                    "entries %s: %d entries; unresolved %d, redirected %d, dropped %d, "
                    "user coverage %s",
                    job.request["request_id"] or job.window[0],
                    len(trace["entries"]),
                    len(audit["unresolved"]),
                    len(audit["redirected"]),
                    len(audit["dropped"]),
                    audit["user_coverage"],
                )
        except CompileError as exc:
            if self.config.keep_raw_on_compile_failure:
                log.warning(
                    "compile failed, kept raw only: %s: %s",
                    job.request["request_id"] or (job.window or ("",))[0],
                    str(exc)[:300],
                )
                self.compile_failed_kept_raw += 1
                return self._journal(job, (), len(job.messages))
            if self.config.skip_rejected_compiles and "rejected the request" in str(exc):
                # The provider permanently refuses this input (e.g. content
                # moderation). Raw evidence stays stored and searchable; this
                # request just yields no compiled Blocks. Counted, never silent.
                log.warning(
                    "compile rejected, kept raw only: %s: %s",
                    job.request["request_id"] or job.window[0],
                    str(exc)[:300],
                )
                self.compile_rejected += 1
                return self._journal(job, (), len(job.messages))
            raise CompileFailed(str(exc)) from exc
        return self._journal(job, blocks, kept, trace)

    def _journal(
        self, job: AddJob, blocks, kept: int | None = None, trace: dict | None = None
    ) -> dict:
        data = blocks_to_json(blocks, self.compiler.version)
        if isinstance(job.known_lines, dict):  # tree: how the prompt's refs resolve
            data["refs"] = dict(job.known_lines.get("_refs", {}))
        if job.window is not None:
            data["kept"] = len(job.messages) if kept is None else kept
        if trace:  # entries placement: kept for audit and replay, never ingested
            data["assembly"] = trace
        return data

    def fail_add(self, job: AddJob, error: str) -> None:
        """Owner thread: record a failed compile so the request stays resumable."""
        self.source.set_error(job.request["request_id"], error)

    def finish_add(self, job: AddJob, compiled: str) -> dict:
        """Owner thread: journal, admit, project, derive Path A; the model calls of placement made
        here, in turn. RuntimeHost drives finish_steps instead, with those calls off the owner
        thread."""
        steps = self.finish_steps(job, compiled)
        sent, thrown = None, None
        while True:
            try:
                call = steps.throw(thrown) if thrown is not None else steps.send(sent)
            except StopIteration as done:
                return done.value
            try:
                sent = (
                    [c[0](*c[1:]) for c in call] if isinstance(call, list) else call[0](*call[1:])
                )
                thrown = None
            except Exception as exc:  # noqa: BLE001 - handed back to the step that asked
                sent, thrown = None, exc

    def finish_steps(self, job: AddJob, compiled: str):
        """finish_add as a generator: yields each model call of placement as (fn, *args), or a
        list of such calls that need nothing of each other (their results come back as a list),
        and gets the result back, so only the store work needs the owner thread (2026-10-31:
        step 2, threads and summaries ran on it, one user at a time, whatever
        compile_concurrency said)."""
        if job.window is not None:
            return (yield from self._finish_window(job, compiled))
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
                    scope, blocks, committed, self.source.compiled(rid).get("refs", {}), at
                )
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
        # the first journal entry wins: a second compile of the same window applies that one, never
        # its own output, and a window already applied is not applied again
        status, compiled = self.source.window(wid)
        kept = compiled.get("kept", len(job.messages))
        if status == "done":
            return {"window": wid, "messages": kept, "blocks": 0, "replayed": True}
        scope = user_scope(user_id)
        blocks = blocks_from_json(compiled)
        committed = self._admit(scope, wid, blocks)
        self._project()
        self._generation += 1
        at = max(m.occurred_at for m in job.messages[:kept])
        if self.config.placement == "moves":
            placement = compiled.get("placement")
            if placement is None:  # step 2; a journaled one is applied without a model call
                try:
                    place = self._stretch if self.config.layer2 == "stretches" else self._place
                    placement = yield from place(user_id, scope, compiled, committed, at)
                except CompileFailed as exc:
                    if not self.config.keep_raw_on_compile_failure:
                        raise
                    log.warning("placement failed, SBs kept unlinked: %s: %s", wid, str(exc)[:300])
                    self.compile_failed_kept_raw += 1
                    placement = {}
                self.source.add_to_window(wid, "placement", placement)
            self._grow_moves(scope, blocks, committed, placement, at)
            if placement.get("threads"):
                self.source.put_threads(
                    user_id,
                    wid,
                    seqs[min(kept, len(seqs)) - 1],
                    {
                        m.memory_id: t
                        for m, t in zip(committed, placement["threads"], strict=True)
                        if t
                    },
                    placement.get("summaries", {}),
                    placement.get("origins", {}),
                )
        else:
            self._grow(scope, blocks, committed, compiled.get("refs", {}), at)
        self.source.set_window_done(wid)
        return {"window": wid, "messages": kept, "blocks": len(committed)}

    def _admit(self, scope, rid, blocks) -> list:
        admission = self.core.semantic_admission(
            sources=self.source, clock=_SystemClock(), origin_runtime_id=ORIGIN
        )
        committed = []
        for index, block in enumerate(blocks):
            refs, roles = [], []
            for record_id in block.source_record_ids:
                message = self.source.message_by_record(record_id)
                if message is None:
                    raise RetryableError(f"raw record vanished: {record_id}")
                refs.append(message.ref)
                roles.append((message.occurred_at, message.index, message.role))
            refs.sort(key=lambda ref: (ref.occurred_at, ref.record_id))
            if self.config.scene == "work" and _past_feedback([r for *_, r in sorted(roles)]):
                self.past_feedback.append(f"aml:{rid}:{index}")
                log.warning("block aml:%s:%d runs past the person's feedback", rid, index)
            committed.append(
                admission.admit(
                    SemanticMemoryCandidate(
                        semantic_id=f"aml:{rid}:{index}",
                        scope=scope,
                        content=block.content,
                        source_refs=tuple(refs),
                        compiler_version=PROMPT_VERSION,
                        attributes=((("standing", "true"),) if block.standing else ())
                        + (
                            (("spans", json.dumps({r: list(p) for r, p in block.spans})),)
                            if block.spans
                            else ()
                        )
                        + ((("pointers", block.pointers),) if block.pointers else ()),
                    )
                )
            )
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
            [
                BlockLines(
                    memory.memory_id,
                    tuple(
                        Attachment(x.thread, x.after, x.committed, x.summary) for x in block.lines
                    ),
                    block.settles,
                )
                for block, memory in zip(blocks, committed, strict=True)
            ],
            refs,
            at,
            open_matters=self.config.placement == "lines",
        )
        if result.failures:  # a refused relation never loses a Block; counted, logged
            self.tree_failures.extend(result.failures)
            log.warning("Path A relation(s) refused: %s", "; ".join(result.failures)[:500])
        return 0

    def _said_by_person(self, memory) -> bool:
        said = [self.source.message_by_record(r.record_id) for r in memory.provenance.source_refs]
        return any(m is not None and m.role == "user" for m in said)

    def _matter(
        self, j: int, hist: list[dict], node_of: dict, j_of: dict, tree, far: datetime
    ) -> int:
        """The person's earlier SB that step j follows: j itself when it is the person's, else its
        nearest ancestor that is (j when there is none)."""
        todo, seen = [node_of[hist[j]["memory"].memory_id]], set()
        while todo:
            node = todo.pop(0)
            if node in seen:
                continue
            seen.add(node)
            k = j_of.get(node)
            if k is not None and self._said_by_person(hist[k]["memory"]):
                return k
            todo += tree.lines.parents_at(node, far)
        return j

    def _span_of(self, memory, user_id: str) -> str:
        """The time range of an SB's messages (message time, from the raw's own stamps; with no
        time given, the session's date), "" when there is none."""
        refs = memory.provenance.source_refs
        given = sorted(r.occurred_at for r in refs if self.source.time_given(r.record_id))
        if given:
            a, b = given[0].strftime("%Y-%m-%d %H:%M"), given[-1].strftime("%Y-%m-%d %H:%M")
            return a if a == b else f"{a} - {b}"
        session_at = self.source.session_times(user_id)
        got = [session_at[r.record_id] for r in refs if r.record_id in session_at]
        return f"session began {min(got).strftime('%Y-%m-%d')}" if got else ""

    def _stretch(self, user_id: str, scope, compiled: dict, committed, at: datetime):
        """layer2 "stretches" (docs/LCE_FROZEN.md): one call cuts the window's SBs into
        stretches, summarizes the closed ones once, says which earlier stretch one takes up
        again and which steps in progress the person's reports settle. A generator, as _place.
        The SBs of a stretch go on one after another on the Line; a stretch carrying on the
        open one, or taking up an earlier one, goes on from its last step."""
        from mr_mem_aml.line_read import STAMP

        lang = (compiled.get("assembly") or {}).get("language") or self.language_of(user_id)
        tree = self.tree_for(scope)
        far = datetime(9999, 1, 1, tzinfo=UTC)
        mine = {m.memory_id for m in committed}
        node_of: dict[str, str] = {}
        for _, node_id, state in tree.nodes(far):
            if state.state_id not in mine:
                node_of.setdefault(state.state_id, node_id)

        def text(memory) -> str:
            return STAMP.sub("", memory.content.split("\n")[0]).strip()

        known = self.source.threads(user_id)
        live = {t: x for t, x in known.items() if any(m in node_of for m in x["members"])}
        open_ = [t for t, x in live.items() if not x["summary"]]
        open_t = max(open_, key=lambda t: live[t]["last_seq"]) if open_ else None
        open_sbs = [
            text(m)
            for b in (live[open_t]["members"] if open_t else [])
            if (m := self.core.canonical.get(b)) is not None
        ]
        closed = [t for t, x in live.items() if x["summary"]]
        earlier = sorted(closed, key=lambda t: -live[t]["last_seq"])[:8]
        if self.embedding is not None and len(closed) > len(earlier):
            import numpy as np

            texts = [text(m) for m in committed]
            missing = [x for x in [*texts, *(live[t]["summary"] for t in closed)]]
            missing = [x for x in dict.fromkeys(missing) if x not in self._vectors]
            if missing:
                got = yield (self._embed_texts, missing)
                self._vectors.update(zip(missing, got, strict=True))
            q = np.mean([np.array(self._vectors[x]) for x in texts], axis=0)
            vecs = np.array([self._vectors[live[t]["summary"]] for t in closed])
            near = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
            earlier = list(dict.fromkeys(earlier[:3] + [closed[i] for i in np.argsort(-near)]))[:8]
        dated = {b: m for b in node_of if (m := self.core.canonical.get(b)) is not None}
        progress = [
            b
            for b in sorted(dated, key=lambda b: (dated[b].occurred_at, b))
            if tree.side(node_of[b], at) == "in progress"
        ][-10:]
        sbs = [{"text": text(m), "time": self._span_of(m, user_id)} for m in committed]
        try:
            got = yield (
                self.compiler.stretches,
                lang,
                sbs,
                open_sbs,
                [live[t]["summary"] for t in earlier],
                [text(m) for b in progress if (m := self.core.canonical.get(b)) is not None],
            )
        except CompileError as exc:
            raise CompileFailed(f"stretches: {exc}") from exc

        refs: dict[str, str] = {}

        def ref(block: str) -> str:  # an earlier step, numbered for the tree as step 2's N<k>
            key = f"N{len(refs) + 1}"
            refs[key] = node_of[block]
            return key

        threads: list[str | None] = [None] * len(committed)
        after: list[list[str]] = [[] for _ in committed]
        summaries: dict[str, str] = {}
        origins: dict[str, dict] = {}
        for s in got["stretches"]:
            first = s["sbs"][0]
            if s["joins_open"] and open_t:
                t = open_t
                last = [b for b in live[open_t]["members"] if b in node_of][-1:]
                after[first] = [ref(last[0])] if last else []
            else:
                t = f"t:{committed[first].memory_id}"
                if s["from"] is not None:
                    frm = earlier[s["from"]]
                    last = [b for b in live[frm]["members"] if b in node_of][-1:]
                    if last:
                        after[first] = [ref(last[0])]
                        origins[t] = {"thread": frm, "step": last[0], "why": ""}
            for a, b in zip(s["sbs"], s["sbs"][1:], strict=False):
                after[b] = [f"b{a + 1}"]
            for i in s["sbs"]:
                threads[i] = t
            summaries[t] = s["summary"]  # "" while the stretch is still open
        settles = [
            [ref(progress[k]) for k in got["settles"].get(i, [])] for i in range(len(committed))
        ]
        return {
            "threads": threads,
            "summaries": summaries,
            "origins": origins,
            "language": lang,
            "refs": refs,
            "after": after,
            "why": [[""] * len(a) for a in after],
            "settles": settles,
            "candidates": [[] for _ in committed],
            "raw": {},
        }

    def _place(self, user_id: str, scope, compiled: dict, committed, at: datetime) -> dict:
        """Placement "moves", step 2 (a generator: reads the store on the owner thread and yields
        its model calls - step 2, threads, summaries - as (fn, *args)).
        Candidates of each SB: the top k earlier SBs by each of four searches - its text
        against SB texts and against the person's words, its move against moves, the
        person's words in it against the person's words - plus the latest earlier SBs and
        the steps still in progress (design 4.4, 4.6)."""
        from mr_mem_aml.moves import split_content

        lang = (compiled.get("assembly") or {}).get("language") or self.language_of(user_id)
        tree = self.tree_for(scope)
        far = datetime(9999, 1, 1, tzinfo=UTC)
        mine = {m.memory_id for m in committed}
        node_of: dict[str, str] = {}
        for _, node_id, state in tree.nodes(far):
            if state.state_id not in mine:
                node_of.setdefault(state.state_id, node_id)
        earlier = sorted(
            (m for m in (self.core.canonical.get(b) for b in node_of) if m is not None),
            key=lambda m: (m.occurred_at, m.memory_id),
        )

        def words(memory) -> str:
            said = [
                self.source.message_by_record(r.record_id) for r in memory.provenance.source_refs
            ]
            text = "\n".join(m.content[:2000] for m in said if m is not None and m.role == "user")
            return text[:2000] or memory.content

        session_at = self.source.session_times(user_id)

        place_of = self.source.places(user_id)

        def when(
            memory,
        ) -> str:  # where it sits (the real order: with no time given, a session's steps
            # all show one date and step 2 loses their order), then the source's time, else its
            # session's
            # so far; never the stored time
            at = [
                place_of[r.record_id]
                for r in memory.provenance.source_refs
                if r.record_id in place_of
            ]
            where = f"S{min(at)[0]} #{min(at)[1]}" if at else ""
            given = [
                r.occurred_at
                for r in memory.provenance.source_refs
                if self.source.time_given(r.record_id)
            ]
            got = [
                session_at[r.record_id]
                for r in memory.provenance.source_refs
                if r.record_id in session_at
            ]
            time = (
                min(given).strftime("%Y-%m-%d %H:%M")
                if given
                else f"session began {min(got).strftime('%Y-%m-%d')}"
                if got
                else ""
            )
            return ", ".join(x for x in (where, time) if x)

        def item(memory) -> dict:
            _, move = split_content(memory.content)
            return {
                "memory": memory,
                "text": memory.content,
                "move": move or memory.content,
                "words": words(memory),
                "time": when(memory),
            }

        hist = [item(m) for m in earlier]
        # chat scene: an SB that builds on one of this window goes on with it; only the window's
        # starts look for what they build on earlier. Any SB of the person may report that an
        # earlier step came to a result, so those are asked for settles only
        unders = [b.get("under") for b in compiled.get("blocks", [])]
        start = [
            self.config.scene != "chat" or i >= len(unders) or unders[i] is None
            for i in range(len(committed))
        ]
        asked = [
            i
            for i in range(len(committed))
            if start[i] or (self.config.scene == "chat" and self._said_by_person(committed[i]))
        ]
        sbs = [{**item(committed[i]), "settles_only": not start[i]} for i in asked]
        k, latest = self.config.place_policy["k"], self.config.place_policy["latest"]
        base = list(range(max(0, len(hist) - latest), len(hist)))
        open_steps = [
            j
            for j, h in enumerate(hist)
            if tree.side(node_of[h["memory"].memory_id], at) == "in progress"
        ][-10:]
        per_sb = []
        if self.embedding is not None and hist:
            import numpy as np

            # the texts not embedded yet, embedded off the owner thread (three per new SB)
            missing = list(
                dict.fromkeys(
                    x[f]
                    for x in (*hist, *sbs)
                    for f in ("text", "move", "words")
                    if x[f] not in self._vectors
                )
            )
            if missing:
                got = yield (self._embed_texts, missing)
                if len(self._vectors) + len(missing) > 50_000:
                    self._vectors.clear()
                self._vectors.update(zip(missing, got, strict=True))
            vec = {
                f: np.array([self._embed_cached(h[f]) for h in hist])
                for f in ("text", "move", "words")
            }
            for sb in sbs:
                q_text, q_move, q_words = (
                    np.array(self._embed_cached(sb[f])) for f in ("text", "move", "words")
                )
                picked = []
                for q, f in (
                    (q_text, "text"),
                    (q_text, "words"),
                    (q_move, "move"),
                    (q_words, "words"),
                ):
                    picked += [int(j) for j in np.argsort(-(vec[f] @ q))[:k]]
                per_sb.append(list(dict.fromkeys(picked + base + open_steps)))
        else:
            per_sb = [list(dict.fromkeys(base + open_steps)) for _ in sbs]
        chat = self.config.scene == "chat"
        if (
            chat
        ):  # each step shown under the person's matter it follows (design 4.1), the matter too
            j_of = {node_of[h["memory"].memory_id]: j for j, h in enumerate(hist)}
            far_ = datetime(9999, 1, 1, tzinfo=UTC)
            matter_of = {
                j: self._matter(j, hist, node_of, j_of, tree, far_) for js in per_sb for j in js
            }
            per_sb = [
                list(dict.fromkeys(x for j in js for x in (matter_of[j], j))) for js in per_sb
            ]
        shown = sorted(
            {j for js in per_sb for j in js}, key=lambda j: (hist[j]["memory"].occurred_at, j)
        )
        pos = {j: n for n, j in enumerate(shown)}
        cands = [
            {
                "text": hist[j]["text"],
                "time": hist[j]["time"],
                "state": tree.side(node_of[hist[j]["memory"].memory_id], at),
            }
            for j in shown
        ]
        if chat:
            for n, j in enumerate(shown):
                m = matter_of.get(j, j)
                cands[n].update(
                    matter=pos.get(m, n),
                    who="person" if self._said_by_person(hist[j]["memory"]) else "assistant",
                )
        lists = [sorted(pos[j] for j in js) for js in per_sb]
        known = self.source.threads(user_id) if chat else {}
        thread_of = {m: t for t, x in known.items() for m in x["members"]}
        # the threads the threads call sees: those of the matters step 2 is shown, and the latest
        # touched
        tids = list(
            dict.fromkeys(
                [t for j in shown if (t := thread_of.get(hist[j]["memory"].memory_id)) is not None]
                + [t for t, _ in sorted(known.items(), key=lambda kv: -kv[1]["last_seq"])[:8]]
            )
        )[:30]
        tids = [t for t in tids if known[t]["summary"]]
        step2 = (
            self.compiler.place,
            lang,
            [
                {"text": s["text"], "time": s["time"], "settles_only": s["settles_only"]}
                for s in sbs
            ],
            cands,
            lists,
        )
        try:
            if (
                chat
            ):  # its own call (inside step 2 it made the links worse), shown the same concrete
                # earlier steps step 2 was, each marked with its thread; it needs nothing of step
                # 2's answer, so both are asked at once
                for j in shown:
                    if (
                        (t := thread_of.get(hist[j]["memory"].memory_id))
                        and known[t]["summary"]
                        and t not in tids
                    ):
                        tids.append(t)
                steps = [
                    {
                        "text": hist[j]["text"],
                        "thread": tids.index(t)
                        if (t := thread_of.get(hist[j]["memory"].memory_id)) in tids
                        else None,
                    }
                    for j in shown
                ]
                placed, threaded = yield [
                    step2,
                    (
                        self.compiler.assign_threads,
                        lang,
                        [
                            {"text": s["text"], "time": s["time"], "cands": lists[n]}
                            for n, s in enumerate(sbs)
                        ],
                        [{"summary": known[t]["summary"]} for t in tids],
                        steps,
                    ),
                ]
                placed.update(threaded)
            else:
                placed = yield step2
        except CompileError as exc:
            raise CompileFailed(f"placement: {exc}") from exc

        if chat:  # only what the person reports settles a step (the assistant saying so does not)
            placed["settles"] = {
                i: [j for j in js if cands[j]["state"] == "in progress"]
                for i, js in placed["settles"].items()
                if self._said_by_person(sbs[i]["memory"])
            }
            placed["after"] = {
                i: rs for i, rs in placed["after"].items() if not sbs[i]["settles_only"]
            }

        def ref(kind: str, n: int) -> str:
            return f"N{n + 1}" if kind == "N" else f"b{asked[n] + 1}"

        at_of = {i: n for n, i in enumerate(asked)}  # window index -> its s number in the prompt
        rows = [at_of.get(i) for i in range(len(committed))]
        threads: list[str | None] = [None] * len(committed)

        def tid(key: tuple, depth: int = 0) -> str:  # a new thread is named by the SB that began it
            if key[0] == "T":
                return tids[key[1]]
            own = placed.get("thread", {}).get(key[1])  # "the thread s3 is in": s3's own answer
            if own is not None and own != key and depth < 8:
                return tid(own, depth + 1)
            return f"t:{committed[asked[key[1]]].memory_id}"

        if chat:
            for n, key in placed.get("thread", {}).items():
                threads[asked[n]] = tid(key)
            for i in range(len(committed)):  # the rest go with what step 1 put them under
                seen_, k = set(), i
                while (
                    threads[k] is None
                    and k < len(unders)
                    and unders[k] is not None
                    and k not in seen_
                ):
                    seen_.add(k)
                    k = unders[k]
                threads[i] = threads[k]
        touched = list(dict.fromkeys(x for x in threads if x is not None))
        # each touched thread's summary: its own call, from its summary so far and its SBs here
        try:
            got = yield (
                self.compiler.summarize,
                lang,
                [
                    {
                        "before": known[t]["summary"] if t in known else "",
                        "new": [
                            m.content.split("\n")[0]
                            for m, x in zip(committed, threads, strict=True)
                            if x == t
                        ],
                    }
                    for t in touched
                ],
            )
        except CompileError as exc:
            raise CompileFailed(f"thread summaries: {exc}") from exc
        summaries = {touched[n]: text for n, text in got.items()}
        origins: dict[
            str, dict
        ] = {}  # a thread begun here: the step whose result led to it, as step 2 said
        for n, (r, why) in placed.get("from", {}).items():
            t = tid(("s", n))
            if r[0] == "T":  # it came out of that thread as it stood: its latest step
                frm = tids[r[1]]
                step_ = known[frm]["members"][-1] if known[frm]["members"] else None
            else:
                step_, frm = committed[asked[r[1]]].memory_id, threads[asked[r[1]]]
            if step_ is None:
                continue
            if frm and frm != t and t not in known:
                origins[t] = {"thread": frm, "step": step_, "why": why}
        from mr_mem_aml import line_read

        for t in dict.fromkeys(
            x for x in threads if x is not None
        ):  # a new thread said nothing of: its first SB
            if t not in summaries and t not in known:
                first = next(m for m, x in zip(committed, threads, strict=True) if x == t)
                summaries[t] = "- " + line_read.STAMP.sub("", first.content.split("\n")[0]).strip()
        return {
            "threads": threads,
            "summaries": summaries,
            "origins": origins,
            "language": lang,
            "refs": {
                f"N{n + 1}": node_of[hist[j]["memory"].memory_id] for n, j in enumerate(shown)
            },
            "after": [
                [ref(*r) for r in placed["after"].get(n, [])] if n is not None else [] for n in rows
            ],
            "why": [
                [placed["why"].get((n, r), "") for r in placed["after"].get(n, [])]
                if n is not None
                else []
                for n in rows
            ],
            "settles": [
                [f"N{j + 1}" for j in placed["settles"].get(n, [])] if n is not None else []
                for n in rows
            ],
            "candidates": [[f"N{j + 1}" for j in lists[n]] if n is not None else [] for n in rows],
            "raw": placed["raw"],
        }

    def _grow_moves(self, scope, blocks, committed, placement: dict, at: datetime) -> None:
        """Placement "moves": every SB becomes a step; its links are step 2's."""
        from mr_mem.lce.path_a import Attachment, BlockLines

        after, settles = placement.get("after", []), placement.get("settles", [])
        lines = []
        for i, (block, memory) in enumerate(zip(blocks, committed, strict=True)):
            happened = block.lines[0].committed if block.lines else True
            refs = list(after[i] if i < len(after) else [])
            part = f"b{block.under + 1}" if block.under is not None and block.under != i else None
            if part is not None:  # what step 1 put it under, in the words: it is a part of that
                refs = [part] + [r for r in refs if r != part]
            refs = refs or [None]
            lines.append(
                BlockLines(
                    memory.memory_id,
                    tuple(
                        Attachment(None, r, happened, None, part=r is not None and r == part)
                        for r in refs
                    ),
                    tuple(settles[i]) if i < len(settles) else (),
                )
            )
        result = self.tree_for(scope).apply(
            lines, placement.get("refs", {}), at, open_matters=False
        )
        if result.failures:  # a refused relation never loses a Block; counted, logged
            self.tree_failures.extend(result.failures)
            log.warning("Path A relation(s) refused: %s", "; ".join(result.failures)[:500])

    def _derive(self, scope, user_id, blocks, committed, at: datetime) -> int:
        # ``at`` is message time: line activity must replay the same way
        result = apply_block_signals(
            canonical=self.core.canonical,
            products=self.core.products,
            lce=self.lce_for(scope),
            scope=scope,
            at=at,
            blocks=tuple(
                BlockSignals(
                    memory.memory_id,
                    block.content,
                    None
                    if block.thread is None
                    else ThreadSignal(
                        block.thread.action,
                        block.thread.question,
                        block.thread.summary,
                        block.thread.mature,
                    ),
                    None
                    if block.commit is None
                    else CommitSignal(
                        block.commit.question,
                        block.commit.summary,
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

    def flush(self, user_id: str | None = None) -> dict:
        """Rescan every user Scope (or only ``user_id``'s) and retry pending mature-Thread
        handoffs."""
        if self.config.compile_window is not None:  # compile what is left, tail included
            windows = 0
            for user in [user_id] if user_id else self.source.users_with_pending():
                while (job := self._window_job(user, final=True)) is not None:
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
                canonical=self.core.canonical,
                products=self.core.products,
                lce=lce,
                scope=scope,
                blocks=(),
                at=datetime.now(UTC),
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
        if selected == "path_a" and self.config.line_read is not None:
            return {
                "data": [
                    {
                        "id": item["id"],
                        "content": item["content"],
                        "score": item["score"],
                        **(
                            {"created_at": item["occurred_at"].isoformat()}
                            if item["occurred_at"]
                            else {}
                        ),
                    }
                    for item in self._line_read(scope, user_id.strip(), text[:4096])[:top_k]
                    if item["content"].strip()
                ]
            }
        raw_search = None
        if self.raw is not None:
            raw, source = self.raw, self.source

            def raw_search(scope, q, limit):
                return [
                    RawHit(
                        u.unit_id,
                        u.text,
                        score,
                        u.occurred_at if source.time_given(u.record_id) else None,
                        (u.record_id,),
                    )
                    for u, score in raw.search(scope.user_id or "", q, limit)
                ]

        reader = PathAReader(
            canonical_path=self.canonical_path,
            retrieval=None if raw_search is not None else self._retrieval(),
            lce=(
                self.lce_for(scope)
                if selected == "path_a" and self.config.path_a_engine == "flat"
                else None
            ),
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

    def decision_hits(self, payload: dict) -> tuple[str, list, str]:
        """For the decision bypass: the question, its raw hits as messages (search order), and the
        person's language. Owner thread: reads the raw index and the source store."""
        from mr_mem_aml import line_read
        from mr_mem_aml.decision import Message

        # the same raw hits the readout gave (its search text, options included, and its raw_k),
        # so a note speaks of the evidence handed over, not of another search
        user_id, question = payload["user_id"].strip(), payload["query"].strip()
        text = question
        options = payload.get("options")
        if self.config.append_options and options:
            text = text + "\n" + "\n".join(str(o) for o in options)
        policy = line_read.LineReadPolicy.of(self.config.line_read)
        records = self._raw_records(user_id, text[:4096], policy)[: self.decision.hits]
        hits = [
            Message(
                m.role,
                m.content,
                r,
                m.occurred_at.isoformat() if self.source.time_given(r) else None,
            )
            for r in records
            if (m := self.source.message_by_record(r)) is not None
        ]
        return question, hits, self.language_of(user_id)

    def _raw_records(self, user_id: str, text: str, policy) -> list[str]:
        """The raw search's messages in order (a message split into units counts once), as many
        as the readout takes: its head is the raw hits, the rest backs a fill."""
        return list(
            dict.fromkeys(
                u.record_id
                for u, _ in self.raw.search(user_id, text, max(policy.raw_k, policy.raw_fill))
            )
        )

    def _hyde_passage(self, text: str) -> str:
        """The hypothetical passage HyDE searches with ("" when it cannot be had: the question
        alone)."""
        from mr_mem.memory.providers.hybrid import PromptHyDEExpander

        if self._hyde is None:
            backend = (
                build_backend(self.config.hyde_backend)
                if self.config.hyde_backend
                else self.backend
            )
            self._hyde = PromptHyDEExpander(backend.complete) if backend is not None else False
        if not self._hyde:
            return ""
        try:
            return self._hyde.expand(text)
        except Exception as exc:  # noqa: BLE001 - HyDE is optional: the question alone still searches
            log.warning("HyDE failed, searching with the question alone: %s", str(exc)[:200])
            return ""

    def _line_read(self, scope, user_id: str, text: str) -> list[dict]:
        """Readout by the upward index (line_read.py): raw hits -> SBs -> the Line around them,
        plus raw snippets. Owner thread: reads the source, canonical and Line stores."""
        from mr_mem.memory.retrieval import MemoryRetrievalQuery
        from mr_mem_aml import line_read

        policy = line_read.LineReadPolicy.of(self.config.line_read)
        # P1's raw list: its head finds the anchors; with raw_fill the whole list backs the Lines up
        # (LCE shares P1's evidence: what the Lines do not cover is still given, as P1 gives it)
        pool = self._raw_records(user_id, text, policy)
        records = pool[: policy.raw_k]
        sb_search = None if self.index is None else self.index.retrieval()
        # HyDE: a hypothetical passage searched beside the question
        if self.config.hyde == "augmented":
            hypo = self._hyde_passage(text)
            if hypo:
                from mr_mem.memory.providers.hybrid import HyDEAugmentedProvider

                extra = [u.record_id for u, _ in self.raw.search(user_id, hypo, policy.raw_k)]
                score: dict[str, float] = {}
                for ranked_ in (records, extra):
                    for n, r in enumerate(dict.fromkeys(ranked_)):
                        score[r] = score.get(r, 0.0) + 1.0 / (60 + n + 1)
                records = sorted(score, key=lambda r: -score[r])[: policy.raw_k]
                if sb_search is not None:
                    sb_search = HyDEAugmentedProvider(sb_search, _FixedExpander(hypo))
        sb_hits = (
            []
            if sb_search is None
            else [
                c.memory_id
                for c in sb_search.search(
                    MemoryRetrievalQuery(scope=scope, text=text, limit=min(policy.sb_k, 100))
                )
            ]
        )
        tree = self.tree_for(scope)
        # now: the person's latest message, so a replayed or offline history reads as it stood then
        lines, far, now = (
            tree.lines,
            datetime(9999, 1, 1, tzinfo=UTC),
            self.source.last_time(user_id) or datetime.now(UTC),
        )

        def nodes(block: str):
            return lines.nodes_for_state(block)

        def parents(block: str, kind: str | None = None) -> list[str]:
            return list(
                dict.fromkeys(
                    lines.get_node(p).block_id
                    for n in nodes(block)
                    for p in lines.parents_at(n.node_id, far, kind=kind)
                )
            )

        def children(block: str, kind: str | None = None) -> list[str]:
            return list(
                dict.fromkeys(
                    lines.get_node(c).block_id
                    for n in nodes(block)
                    for c in lines.children_at(n.node_id, far, kind=kind)
                )
            )

        steps: dict[str, line_read.Step] = {}
        places = self.source.places(user_id)
        session_at = self.source.session_times(user_id)

        def step(block: str) -> line_read.Step:
            if block not in steps:
                memory = self.core.canonical.get(block)
                given = (
                    []
                    if memory is None
                    else [
                        r.occurred_at
                        for r in memory.provenance.source_refs
                        if self.source.time_given(r.record_id)
                    ]
                )
                at = (
                    []
                    if memory is None
                    else [
                        places[r.record_id]
                        for r in memory.provenance.source_refs
                        if r.record_id in places
                    ]
                )
                on = nodes(block)
                later = []
                for by in tree.tense.settled_by(block, now):
                    m = self.core.canonical.get(by)
                    got = (
                        []
                        if m is None
                        else [
                            places[r.record_id]
                            for r in m.provenance.source_refs
                            if r.record_id in places
                        ]
                    )
                    if got and min(got) not in later:
                        later.append(min(got))
                steps[block] = line_read.Step(
                    "" if memory is None else memory.content,
                    min(given) if given else None,
                    bool(on) and bool(tree.confirmed(on[0].node_id, now)),
                    min(at) if at else None,
                    tuple(later),
                    None
                    if memory is None or given
                    else min(
                        (
                            session_at[r.record_id]
                            for r in memory.provenance.source_refs
                            if r.record_id in session_at
                        ),
                        default=None,
                    ),
                )
            return steps[block]

        def sources(block: str) -> list[str]:
            memory = self.core.canonical.get(block)
            return [] if memory is None else [r.record_id for r in memory.provenance.source_refs]

        def message(record: str):
            m = self.source.message_by_record(record)
            return (
                None
                if m is None
                else line_read.RawMessage(
                    m.content,
                    m.role,
                    m.occurred_at if m.time_given else None,
                    places.get(record),
                    None if m.time_given else session_at.get(record),
                )
            )

        standing = [
            b
            for b in self._standing.get(scope, ())
            if (m := self.core.canonical.get(b)) is not None
            and m.lifecycle is MemoryLifecycle.ACTIVE
        ]
        question: list = []

        def relevance(block: str) -> float:  # how close the step's text is to the question
            import numpy as np

            memory = self.core.canonical.get(block)
            if memory is None or self.embedding is None:
                return 0.0
            if not question:
                question.append(np.array(self._embed_cached(text)))
            q, v = question[0], np.array(self._embed_cached(memory.content.split("\n")[0]))
            return float(q @ v) / (np.linalg.norm(q) * np.linalg.norm(v) or 1)

        up, down = parents, children
        if (
            policy.carry == "threads"
        ):  # carried on by the thread's order and its branch, not by SB links
            before: dict[str, list[str]] = {}
            after_: dict[str, list[str]] = {}
            for x in self.source.threads(user_id).values():
                mine = sorted(
                    (b for b in x["members"] if self.core.canonical.get(b) is not None),
                    key=lambda b: (step(b).place or (0, 0), b),
                )
                for a, b in zip(mine, mine[1:], strict=False):
                    after_.setdefault(a, []).append(b)
                    before.setdefault(b, []).append(a)
                if (o := x.get("origin")) and mine:
                    before.setdefault(mine[0], []).append(o["step"])
                    after_.setdefault(o["step"], []).append(mine[0])

            def up(block: str, kind: str | None = None) -> list[str]:
                return [
                    *(parents(block, "part") if kind != "after" else []),
                    *(before.get(block, []) if kind != "part" else []),
                ]

            def down(block: str, kind: str | None = None) -> list[str]:
                return [
                    *(children(block, "part") if kind != "after" else []),
                    *(after_.get(block, []) if kind != "part" else []),
                ]

        thread_hits: list[str] = []
        if (
            policy.thread_k and self.embedding is not None
        ):  # the question against each thread's summary
            import numpy as np

            known = self.source.threads(user_id)
            tids = [t for t, x in known.items() if x["summary"]]
            if tids:
                if not question:
                    question.append(np.array(self._embed_cached(text)))
                q = question[0]
                vecs = np.array([self._embed_cached(known[t]["summary"]) for t in tids])
                score = vecs @ q / (np.linalg.norm(vecs, axis=1) * np.linalg.norm(q) + 1e-9)
                thread_hits = [tids[i] for i in np.argsort(-score)[: policy.thread_k]]
        ranked = [
            b
            for b in line_read.rank_blocks(
                records,
                lambda r: self._blocks_by_record.get(r, ()),
                sb_hits,
                policy.rrf_k,
                closest=lambda blocks: max(blocks, key=relevance),
            )
            if self.core.canonical.get(b) is not None
        ]
        hit_blocks = [
            max(blocks, key=relevance)
            for r in (records if policy.hit_anchors else ())
            if (
                blocks := [
                    b for b in self._blocks_by_record.get(r, ()) if self.core.canonical.get(b)
                ]
            )
        ]
        return line_read.compose(
            text,
            ranked,
            records,
            fill_records=pool,
            uncompiled_records=[
                r
                for r in records
                if not any(self.core.canonical.get(b) for b in self._blocks_by_record.get(r, ()))
            ],
            step=step,
            sources=sources,
            parents=up,
            children=down,
            message=message,
            lang=self.language_of(user_id),
            policy=policy,
            standing=standing,
            span=lambda block, record: self._spans.get(block, {}).get(record),
            relevance=relevance,
            pointers=self._pointers.get,
            part_of=lambda b: up(b, "part"),
            parts=lambda b: down(b, "part"),
            carries=lambda b: [*up(b, "after"), *down(b, "after")],
            everything=lambda: [
                b
                for b in self._pointers
                if (m := self.core.canonical.get(b)) is not None and m.scope == scope
            ],
            threads=lambda: self.source.threads(user_id),
            thread_hits=thread_hits,
            hit_blocks=hit_blocks,
        )


def _past_feedback(roles: list[str]) -> bool:
    """An assistant message after a message of the person, in one work block."""
    return (
        any(r == "assistant" for r in roles[roles.index("user") + 1 :])
        if "user" in roles
        else False
    )


class _FixedExpander:
    """A HyDE expander that hands back a passage already written for this query."""

    def __init__(self, passage: str) -> None:
        self._passage = passage

    def expand(self, query: str) -> str:
        return self._passage
