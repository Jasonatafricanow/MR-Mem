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
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from mr_mem_aml.source_store import RawMessage

PROMPT_VERSION = "mr-mem-aml-block-compiler-v2"
AssistantPolicy = Literal["attributed", "context_only"]


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
    """``/chat/completions`` of any OpenAI-compatible endpoint (e.g. xAI, OpenAI)."""

    def __init__(
        self, *, base_url: str, model: str, api_key_env: str,
        timeout: float = 600, attempts: int = 4,
    ) -> None:
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key_env = model, api_key_env
        self.timeout, self.attempts = timeout, attempts

    def complete(self, prompt: str) -> str:
        key = os.environ.get(self.api_key_env)
        if not key:
            raise CompileError(f"environment variable {self.api_key_env} is not set")
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        last = ""
        for attempt in range(1, self.attempts + 1):
            request = urllib.request.Request(
                self.url, data=body, method="POST",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
            except Exception as exc:  # noqa: BLE001 - transport retry boundary
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, 30))
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
class CompiledBlock:
    content: str
    source_record_ids: tuple[str, ...]
    thread: ThreadSignalOut | None
    commit: CommitOut | None


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
   "next Friday") against the message time when unambiguous and keep the
   original expression as well.
3. Time-local: record the state at that moment. Never add later outcomes.
4. Keep what is unknown unknown; do not guess or generalize.
5. Useful granularity: neither one giant block nor atomized fragments.
6. Skip content without lasting information (greetings, thanks, filler).
7. {assistant_rule}
8. Write each block in the conversation's language as plain declarative
   third-person sentences ("The user ...", "The assistant ...").

Line-of-thought signals (optional, per block; never both on one block):
- "thread": the block advances an explicit, still-developing line the user is
  working through across turns (a pending decision, plan, problem, changing
  preference or situation). "action": "track" (opens or continues the line) or
  "resolve" (the line is settled or closed). "question": short stable phrasing
  of the line; reuse the exact text from KNOWN_LINES when it is the same line.
  "summary": the current understanding of the line after this block, concise
  and specific. "mature": true only when the line has reached a settled
  understanding worth keeping as long-term cognition.
- "commit": only when the cited NEW messages already contain a complete,
  self-contained reasoned judgment of the user at that time (for example a
  decision together with its reasons, or a worked-out principle or conclusion),
  so that no further turns are needed to understand it. "question": the stable
  question it answers (reuse KNOWN_LINES when it is the same line). "summary":
  the judgment with its key reasons and conditions, faithful to the user's own
  words. Never commit casual facts, bare preferences without reasoning, or the
  assistant's views.
- If a block settles or continues a line already listed in KNOWN_LINES, use
  "thread" with that exact question ("resolve", or "track" with "mature": true
  when settled) instead of "commit". "commit" is only for complete judgments
  that do not continue a known line.

Return only JSON, no prose:
{{"blocks": [{{"content": "...", "sources": ["m1"],
  "thread": {{"action": "track", "question": "...", "summary": "...", "mature": false}} or null,
  "commit": {{"question": "...", "summary": "..."}} or null}}]}}
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
    ) -> None:
        if assistant_policy not in ASSISTANT_RULES:
            raise ValueError("assistant_policy must be 'attributed' or 'context_only'")
        self.backend = backend
        self.assistant_policy = assistant_policy
        self.attempts = attempts

    def prompt(
        self,
        new: tuple[RawMessage, ...],
        context: tuple[RawMessage, ...],
        known_lines: tuple[str, ...],
    ) -> str:
        material = {
            "KNOWN_LINES": list(known_lines),
            "CONTEXT": [
                {"role": m.role, "time": m.occurred_at.isoformat(), "text": m.content}
                for m in context
            ],
            "NEW": [
                {"id": f"m{i + 1}", "role": m.role, "time": m.occurred_at.isoformat(),
                 "text": m.content}
                for i, m in enumerate(new)
            ],
        }
        rules = INSTRUCTIONS.format(assistant_rule=ASSISTANT_RULES[self.assistant_policy])
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
                thread=_thread(item.get("thread")),
                commit=_commit(item.get("commit"), cited),
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


def _commit(value: object, cited: list[RawMessage]) -> CommitOut | None:
    if not isinstance(value, dict) or not any(m.role == "user" for m in cited):
        return None
    question, summary = value.get("question"), value.get("summary")
    if not all(isinstance(v, str) and v.strip() for v in (question, summary)):
        return None
    return CommitOut(question.strip()[:2000], summary.strip()[:4000])


def blocks_to_json(blocks: tuple[CompiledBlock, ...]) -> dict:
    return {
        "prompt_version": PROMPT_VERSION,
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
        )
        for b in data["blocks"]
    )
