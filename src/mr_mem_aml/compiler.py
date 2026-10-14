"""Semantic compiler transport for the AML runtime.

One model call per Add request produces, as sibling outputs of the same
understanding pass: the SemanticBlocks, optional Thread signals and optional
first-appearance commits. Path A never calls a model again; it only promotes
what this pass already reasoned (MEMORY_ARCHITECTURE_V1 section 6.1).

Block definition follows LCE ``SEMANTIC_BLOCK_DESIGN_FREEZE_2026-09-25``.
No SemanticPoint is produced: standalone MR-Mem has no Point consumer.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mr_mem_aml.source_store import RawMessage

PROMPT_VERSION = "mr-mem-aml-block-compiler-v3"
TREE_PROMPT_VERSION = "mr-mem-aml-tree-compiler-v4"
AssistantPolicy = Literal["attributed", "context_only"]
Schema = Literal["flat", "tree"]


class CompileError(RuntimeError):
    """The compiler could not return a valid proposal."""


class CompletionBackend(Protocol):
    def complete(self, prompt: str) -> str: ...


class CommandBackend:
    """Answer on stdout from a CLI (agy, grok and similar).

    The prompt goes to stdin unless an argv element contains ``{prompt}``
    (substituted inline) or ``{prompt_file}`` (path of a UTF-8 temp file).
    Prefer stdin or ``{prompt_file}`` on Windows, where argv is limited to
    32k characters.
    """

    def __init__(self, argv: list[str], *, timeout: float = 600, attempts: int = 4) -> None:
        if not argv:
            raise ValueError("command argv must be nonempty")
        self.argv, self.timeout, self.attempts = list(argv), timeout, attempts

    def _run(self, prompt: str) -> subprocess.CompletedProcess:
        inline = any("{prompt}" in part for part in self.argv)
        via_file = any("{prompt_file}" in part for part in self.argv)
        if not inline and not via_file:
            return subprocess.run(
                self.argv, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout, check=False,
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompt.txt"
            path.write_text(prompt, encoding="utf-8")
            argv = [
                part.replace("{prompt_file}", str(path)).replace("{prompt}", prompt)
                for part in self.argv
            ]
            return subprocess.run(
                argv, stdin=subprocess.DEVNULL, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout, check=False,
            )

    def complete(self, prompt: str) -> str:
        last = ""
        for attempt in range(1, self.attempts + 1):
            try:
                result = self._run(prompt)
            except subprocess.TimeoutExpired:
                last = "timeout"
            else:
                if result.returncode == 0 and result.stdout.strip():
                    return result.stdout
                last = (result.stderr or result.stdout or "")[-2000:]
            time.sleep(min(2**attempt, 30))
        raise CompileError(f"compiler command failed: {last}")


class OpenAICompatibleBackend:
    """``/chat/completions`` of any OpenAI-compatible endpoint.

    Covers hosted APIs (OpenAI, DeepSeek, SiliconFlow, OpenRouter, xAI, Gemini's
    OpenAI endpoint, ...) and local servers (Ollama, vLLM, LM Studio, llama.cpp).
    ``api_key_env=None`` sends no Authorization header (typical for local
    servers). ``json_mode`` asks for ``response_format: json_object`` where the
    endpoint supports it; ``extra_body`` is merged into the request verbatim.
    """

    def __init__(
        self, *, base_url: str, model: str, api_key_env: str | None = None,
        timeout: float = 600, attempts: int = 6, json_mode: bool = False,
        max_tokens: int | None = None, temperature: float = 0,
        headers: dict[str, str] | None = None, extra_body: dict | None = None,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.attempts = timeout, attempts
        self.json_mode, self.max_tokens, self.temperature = json_mode, max_tokens, temperature
        self.headers, self.extra_body = dict(headers or {}), dict(extra_body or {})

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.headers}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise CompileError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def complete(self, prompt: str) -> str:
        payload: dict = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_body)
        body = json.dumps(payload).encode("utf-8")
        headers = self._headers()
        last = ""
        for attempt in range(1, self.attempts + 1):
            request = urllib.request.Request(self.url, data=body, method="POST", headers=headers)
            wait = min(2**attempt, 60)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = data["choices"][0]["message"]["content"]
                if isinstance(content, str) and content.strip():
                    return content
                last = "empty completion"
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:500].decode("utf-8", "replace")
                last = f"HTTP {exc.code}: {detail}"
                if exc.code in (400, 401, 403, 404, 422):
                    raise CompileError(f"compiler endpoint rejected the request: {last}") from exc
                retry_after = exc.headers.get("Retry-After") if exc.headers else None
                if retry_after and retry_after.strip().isdigit():
                    wait = min(int(retry_after), 120)
            except Exception as exc:  # noqa: BLE001 - transport retry boundary
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.attempts:
                time.sleep(wait)
        raise CompileError(f"compiler endpoint failed: {last}")


@dataclass(frozen=True, slots=True)
class ThreadSignalOut:
    action: str  # track | resolve
    question: str
    summary: str | None
    mature: bool


@dataclass(frozen=True, slots=True)
class CommitOut:
    question: str
    summary: str


@dataclass(frozen=True, slots=True)
class LineOut:
    """Tree schema: this block's place in one matter (design v6 4.4-4.5)."""

    thread: str | None  # "T<k>" known thread, "b<k>" thread opened by block k, None = new
    after: str | None  # "N<k>" known step, "T<k>" a thread's branch point, "b<k>" block k
    committed: bool  # it happened and can no longer change at this message time
    summary: str | None
    done: bool = False  # ignored since tree prompt v4 (lines never end); older journals


@dataclass(frozen=True, slots=True)
class CompiledBlock:
    content: str
    source_record_ids: tuple[str, ...]
    thread: ThreadSignalOut | None
    commit: CommitOut | None
    lines: tuple[LineOut, ...] = ()


INSTRUCTIONS = """You are the semantic compiler of a long-term memory system.
You receive NEW conversation messages to compile and earlier CONTEXT messages of
the same session (read-only, never cite them). Compile the NEW messages into
SemanticBlocks and, when warranted, line-of-thought signals.

A SemanticBlock is the smallest unit that faithfully and completely preserves
what was actually meant at that time, using as much surrounding context as is
needed to avoid distortion.

Block rules:
1. Boundaries follow semantic completeness, not messages or sentences. One block
   may cite several NEW messages that must be read together; one message may
   yield several blocks when it contains independent matters.
2. A reader who never sees the original text must be able to reconstruct the
   meaning: keep who said it (user, assistant, a third party), the modality
   (fact, plan, intention, wish, question, possibility, reported claim,
   correction, refusal), and every concrete specific (names, numbers, amounts,
   dates and times, places, titles, items). Resolve relative time ("yesterday",
   "next Friday", "this afternoon") against the message time when unambiguous,
   write it as YYYY-MM-DD or YYYY-MM-DD HH:MM, and keep the original expression
   as well ("this afternoon (2024-05-06 15:00)").
3. Time-local: record the state at that moment. Never add later outcomes.
4. Keep what is unknown unknown; do not guess or generalize.
5. Useful granularity: neither one giant block nor atomized fragments.
6. Skip content without lasting information (greetings, thanks, filler).
7. {assistant_rule}
8. Write each block in the conversation's language as plain declarative
   third-person sentences ("The user ...", "The assistant ...").

Line-of-thought signals (optional, per block; never both on one block):
- "thread": the block belongs to a matter the user is living through across
  turns: a decision being weighed, a plan, a trip, a problem, a changing
  situation or preference. One matter usually spans several blocks (what was
  done, why, what was observed, what was thought, what may follow): attach each
  of them to the same line instead of packing the whole matter into one block or
  opening a second line for it.
  "question": short stable phrasing of the matter; when KNOWN_LINES has the same
  matter (open or settled), reuse its question text exactly.
  "action": "track" while the matter is still developing; "resolve" when, at
  this message time, the matter has ended: a decision was made, a plan was
  carried out or dropped, a problem was solved.
  "summary": the current understanding of the whole matter after this block,
  concise and specific, with dates in the format above.
  - Decisions and opinions are settled when made ("no budget, not buying it":
    resolve). If the user later reconsiders the same matter, continue that line
    (same question) and say in the summary what changed against the earlier
    reasons.
  - Time-bound real-world plans (a flight this afternoon, a meeting tomorrow)
    stay open until the messages report what actually happened; then record that
    and continue with what follows. Never assume a plan happened because its time
    has passed, and never invent outcomes or alternatives that were not stated.
- "commit": only when the cited NEW messages already contain a complete,
  self-contained reasoned judgment of the user at that time (for example a
  decision together with its reasons, or a worked-out principle or conclusion)
  that does not continue a line in KNOWN_LINES. "question": the stable question
  it answers. "summary": the judgment with its key reasons and conditions,
  faithful to the user's own words. Never commit casual facts, bare preferences
  without reasoning, or the assistant's views.
- KNOWN_LINES lists existing lines relevant now: {{"question", "state": open |
  settled, "summary"}}. A block that continues one of them uses "thread" with its
  exact question, never "commit".

Return only JSON, no prose:
{{"blocks": [{{"content": "...", "sources": ["m1"],
  "thread": {{"action": "track", "question": "...", "summary": "..."}} or null,
  "commit": {{"question": "...", "summary": "..."}} or null}}]}}
"""

TREE_INSTRUCTIONS = INSTRUCTIONS[: INSTRUCTIONS.index("Line-of-thought signals")] + """\
9. When one message holds something that already happened and something still
   open ("we agreed to have dinner tomorrow": the agreement happened, the dinner
   has not), make them two blocks.

Matters (optional, per block, as "lines"): the user's life is made of matters
(a trip, a purchase being weighed, a project, a problem). A matter grows step by
step; steps that have happened are fixed, the step in progress can still change.
LINES shows what is relevant now:
- "threads": matters in progress, {{"ref": "T1", "summary", "after": the step it
  currently hangs from}}. A thread has no name; its summary says what it is
  about now.
- "steps": steps that already happened, {{"ref": "N1", "text", "time"}}.
- "rejected": understandings the user denied; never produce them again from the
  same evidence.
For each block give "lines": a list with one entry per matter the block belongs
to (empty when it belongs to none). One block may belong to several matters of
different kinds (a dinner in Lisbon belongs to the trip and to food tastes).
Each entry:
  "thread": which matter: a thread ref from LINES ("T2") to continue it, "b<k>"
  to continue the matter that block k of this output started, or null to start
  a new matter. Continue a thread only when it is really the same matter, not
  merely similar words (two different plans are two matters). A matter that
  changed course (new deadline, narrower scope, a different destination) is
  still the same matter: continue it.
  "after": the step this block builds on within this matter: a step ref ("N3"),
  a thread ref ("T2") for its current branch point, "b<k>" for the k-th block of
  this output (1-based), or null when nothing known comes before it. It is often
  the latest step of the matter, but not necessarily: a sub-task grows out of
  the step it belongs to (dinner plans after arriving in Lisbon), and parallel
  sub-tasks may grow out of the same step. Time order is a strong hint, never a
  rule: do not attach a block to the previous step only because it came last.
  "committed": true when what the block states has actually happened and cannot
  change any more at this message time (a booking made, a flight taken, an
  arrival, a decision made with its reasons); false while it is still open (a
  plan, an intention, an outcome not yet known). A plan's time passing is not
  evidence that it happened.
  "summary": the current state of the whole matter after this block: what it is
  about now, how it started, the key steps so far and where it stands; concise,
  with dates in the format above. The summary is how this matter is found later,
  so it follows the matter as it changes (a superseded plan is described as
  superseded, not as current).
- Matters never end. "committed" is the only judgment: what has happened is
  fixed, what has not can still change, and any matter can develop further
  later. A sub-task that got done is a committed step of its own branch; the
  larger matter it belongs to (the project, the trip) goes on. Never close or
  conclude a matter.
- A block may belong to several lines at once (the project timeline and the
  sub-task it serves, or two different kinds of matter): give one entry each.
- When the user later develops something that was decided earlier (wants the
  laptop again after deciding not to buy it), continue from the old step
  ("after" = that step) and compare against the old reasons in the summary.
  Never rewrite the old step.
- Never invent outcomes or alternatives that were not stated.

Return only JSON, no prose:
{{"blocks": [{{"content": "...", "sources": ["m1"],
  "lines": [{{"thread": "T1", "after": "N1", "committed": true,
  "summary": "..."}}]}}]}}
"""

ASSISTANT_RULES = {
    "attributed": (
        "Assistant messages may be compiled when they carry information the user "
        "received or acted on (recommendations, answers, facts, plans); always "
        "attribute them to the assistant. Never present them as the user's view."
    ),
    "context_only": (
        "Assistant messages are context only: cite user messages only and never "
        "turn assistant content into a block of its own."
    ),
}


def _time(message) -> dict:
    """Message time only when the source gave one; otherwise order is all we know."""
    if getattr(message, "time_given", True):
        return {"time": message.occurred_at.isoformat()}
    return {}


def _extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in compiler output")
    return json.loads(candidate[start : end + 1])


def _text(value: object, name: str, limit: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    return value.strip()


class SemanticCompiler:
    def __init__(
        self,
        backend: CompletionBackend,
        *,
        assistant_policy: AssistantPolicy = "attributed",
        attempts: int = 3,
        schema: Schema = "flat",
    ) -> None:
        if assistant_policy not in ASSISTANT_RULES:
            raise ValueError("assistant_policy must be 'attributed' or 'context_only'")
        if schema not in ("flat", "tree"):
            raise ValueError("schema must be 'flat' or 'tree'")
        self.backend = backend
        self.assistant_policy = assistant_policy
        self.attempts = attempts
        self.schema = schema
        self.version = TREE_PROMPT_VERSION if schema == "tree" else PROMPT_VERSION

    def prompt(
        self,
        new: tuple[RawMessage, ...],
        context: tuple[RawMessage, ...],
        known_lines: tuple,
    ) -> str:
        if self.schema == "tree":  # {"threads": [...], "steps": [...], "rejected": [...]}
            lines = dict(known_lines) if isinstance(known_lines, dict) else {}
            head = {"LINES": {k: v for k, v in lines.items() if v and not k.startswith("_")}}
        else:
            # str entries (older journals) or {"question", "state", "summary"}
            head = {"KNOWN_LINES": [
                line if isinstance(line, (str, dict)) else line.__dict__
                for line in known_lines
            ]}
        material = {
            **head,
            "CONTEXT": [
                {"role": m.role, **_time(m), "text": m.content}
                for m in context
            ],
            "NEW": [
                {"id": f"m{i + 1}", "role": m.role, **_time(m), "text": m.content}
                for i, m in enumerate(new)
            ],
        }
        template = TREE_INSTRUCTIONS if self.schema == "tree" else INSTRUCTIONS
        rules = template.format(assistant_rule=ASSISTANT_RULES[self.assistant_policy])
        return rules + "\nINPUT:\n" + json.dumps(material, ensure_ascii=False, indent=1)

    def parse(self, raw: str, new: tuple[RawMessage, ...]) -> tuple[CompiledBlock, ...]:
        data = _extract_json(raw)
        blocks = data.get("blocks") if isinstance(data, dict) else None
        if not isinstance(blocks, list):
            raise ValueError("'blocks' must be a list")
        ids = {f"m{i + 1}": m for i, m in enumerate(new)}
        out: list[CompiledBlock] = []
        for item in blocks:
            if not isinstance(item, dict):
                raise ValueError("each block must be an object")
            content = _text(item.get("content"), "content", 4000)
            sources = item.get("sources")
            if not isinstance(sources, list) or not sources or any(s not in ids for s in sources):
                raise ValueError("sources must cite NEW message ids")
            cited = [ids[s] for s in dict.fromkeys(sources)]
            if self.assistant_policy == "context_only":
                cited = [m for m in cited if m.role == "user"]
                if not cited:
                    raise ValueError("context_only policy: block cites no user message")
            out.append(CompiledBlock(
                content=content,
                source_record_ids=tuple(m.record_id for m in cited),
                thread=None if self.schema == "tree" else _thread(item.get("thread")),
                commit=None if self.schema == "tree" else _commit(item.get("commit"), cited),
                lines=_lines(item.get("lines")) if self.schema == "tree" else (),
            ))
        return tuple(out)

    def compile(
        self,
        new: tuple[RawMessage, ...],
        context: tuple[RawMessage, ...],
        known_lines: tuple[str, ...] = (),
    ) -> tuple[CompiledBlock, ...]:
        if not new:
            return ()
        prompt = self.prompt(new, context, known_lines)
        last = ""
        for _ in range(self.attempts):
            raw = self.backend.complete(prompt)
            try:
                return self.parse(raw, new)
            except (ValueError, KeyError, TypeError) as exc:
                last = str(exc)
                prompt_retry = (
                    f"\n\nYour previous output was invalid ({last}). Return only the JSON."
                )
                prompt = self.prompt(new, context, known_lines) + prompt_retry
        raise CompileError(f"invalid compiler output after {self.attempts} attempts: {last}")


def _thread(value: object) -> ThreadSignalOut | None:
    """Malformed optional signals are dropped; the block itself is kept."""
    if not isinstance(value, dict):
        return None
    action = value.get("action")
    question = value.get("question")
    if action not in ("track", "resolve") or not isinstance(question, str) or not question.strip():
        return None
    summary = value.get("summary")
    summary = summary.strip()[:4000] if isinstance(summary, str) and summary.strip() else None
    return ThreadSignalOut(action, question.strip()[:2000], summary, value.get("mature") is True)


def _lines(value: object) -> tuple[LineOut, ...]:
    """Malformed entries are dropped; the block itself is kept."""
    if not isinstance(value, list):
        return ()
    out: list[LineOut] = []
    for entry in value:
        if not isinstance(entry, dict):
            continue
        thread, after = entry.get("thread"), entry.get("after")
        thread = thread.strip() if isinstance(thread, str) and thread.strip() else None
        if thread is not None and thread.lower() in ("null", "none", "new"):
            thread = None
        after = after.strip() if isinstance(after, str) and after.strip() else None
        summary = entry.get("summary")
        summary = summary.strip()[:4000] if isinstance(summary, str) and summary.strip() else None
        out.append(LineOut(thread, after, entry.get("committed") is True,
                           summary, entry.get("done") is True))
    return tuple(out)


def _commit(value: object, cited: list[RawMessage]) -> CommitOut | None:
    if not isinstance(value, dict) or not any(m.role == "user" for m in cited):
        return None
    question, summary = value.get("question"), value.get("summary")
    if not all(isinstance(v, str) and v.strip() for v in (question, summary)):
        return None
    return CommitOut(question.strip()[:2000], summary.strip()[:4000])


def blocks_to_json(blocks: tuple[CompiledBlock, ...], version: str = PROMPT_VERSION) -> dict:
    return {
        "prompt_version": version,
        "blocks": [
            {
                "content": b.content,
                "sources": list(b.source_record_ids),
                "thread": None if b.thread is None else {
                    "action": b.thread.action, "question": b.thread.question,
                    "summary": b.thread.summary, "mature": b.thread.mature,
                },
                "commit": None if b.commit is None else {
                    "question": b.commit.question, "summary": b.commit.summary,
                },
                **({"lines": [
                    {"thread": x.thread, "after": x.after, "committed": x.committed,
                     "summary": x.summary, "done": x.done}
                    for x in b.lines
                ]} if b.lines else {}),
            }
            for b in blocks
        ],
    }


def blocks_from_json(data: dict) -> tuple[CompiledBlock, ...]:
    return tuple(
        CompiledBlock(
            content=b["content"],
            source_record_ids=tuple(b["sources"]),
            thread=None if b["thread"] is None else ThreadSignalOut(**b["thread"]),
            commit=None if b["commit"] is None else CommitOut(**b["commit"]),
            lines=tuple(LineOut(**x) for x in b.get("lines", ())),
        )
        for b in data["blocks"]
    )
