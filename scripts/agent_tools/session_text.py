"""The cleaning a session transcript goes through before compile, so that a message read back from the
original file reads as it was compiled: noise stripped, secrets redacted, pasted text marked.
INC_RE / EXC_RE keep a corpus about one project apart from turns about other projects."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

BLOCK_RE = re.compile(
    r"<(system-reminder|task-notification|command-name|command-message|command-args|"
    r"local-command-stdout|local-command-stderr|local-command-caveat|"
    r"bash-stdout|bash-stderr|ide_opened_file|ide_selection|"
    r"open_and_recently_viewed_files|attached-files|file-attachment|"
    r"user-prompt-submit-hook|commit_attribution|total_tokens|"
    r"environment_details|side-sessions-event|agent-message|"
    r"tools-used-before-summary|mcp_instructions|agent_skills)"
    r"(?:\s[^>]*)?>.*?</\1>",
    re.I | re.S,
)
WRAP_RE = re.compile(r"</?(?:user_query|user_info|timestamp)\s*>", re.I)
PASTE_RE = re.compile(
    r"<pasted_content\b[^>]*>(.*?)</pasted_content\b[^>]*>",
    re.I | re.S,
)
SECRET_RE = re.compile(
    r"sk-[A-Za-z0-9_\-]{8,}"
    r"|ibk_[A-Za-z0-9_\-]{8,}"
    r"|Bearer\s+\S+"
    r"|\b(?:api[_-]?key|access[_-]?key|secret|password|passwd|token)\b\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{12,}"
    r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{8,}"
    r"|AKIA[0-9A-Z]{16}",
    re.I,
)
USAGE_RE = re.compile(r"\[Usage limit reached.*?\]", re.S)
INC_RE = re.compile(
    r"mr-?mem|mrmem|\blce\b|path\s*[- ]?[ab]\b|patha|pathb|晨昏线|"
    r"semantic\s*block|语义块|mrmem-eval|\baml\b",
    re.I,
)
EXC_RE = re.compile(
    r"habitat|shopify|涨停|tradingweb|stock-dashboard|stock-system",
    re.I,
)
KEEP_ASIDE_RE = re.compile(r"像|那样|上次|类似|别动|不要动|先不管|无关|别碰|不是这个")
CUE_RE = re.compile(
    r"GPT|Gemini|Codex|Grok|Astra|handoff|交接|审核报告|执行报告|"
    r"日志如下|运行日志|另一个\s*(?:session|会话|模型)|别的\s*(?:AI|模型)|"
    r"转贴|粘贴给|贴过来|贴上",
    re.I,
)
LOOK_RE = re.compile(r"看|下面|以下|这份|这段|贴")


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text)


def to_ms(ts: str) -> int:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return int(datetime.fromisoformat(ts).timestamp() * 1000)


def raw_text(content, role: str) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for part in content:
        if isinstance(part, str):
            chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind in ("tool_result", "tool_use", "thinking", "redacted_thinking", "image"):
            continue
        if kind == "text" or (role == "user" and "text" in part):
            chunks.append(str(part.get("text") or ""))
    return "\n".join(chunks)


def strip_noise(text: str) -> str:
    text = BLOCK_RE.sub("", text)
    text = WRAP_RE.sub("", text)
    text = USAGE_RE.sub("", text)
    return text.strip()


def mark_paste(text: str, prev_user: str) -> tuple[str, bool]:
    pastes = PASTE_RE.findall(text)
    outside = PASTE_RE.sub(" ", text)
    body = PASTE_RE.sub(lambda m: m.group(1).strip(), text).strip()
    body = strip_noise(body)
    marked = False
    if pastes:
        inside = "\n".join(pastes)
        if len(inside) >= 200 and (CUE_RE.search(outside) or CUE_RE.search(inside[:1200])):
            marked = True
    elif (
        len(body) >= 1500
        and len(prev_user) <= 500
        and CUE_RE.search(prev_user)
        and LOOK_RE.search(prev_user)
    ):
        marked = True
    if marked:
        body = "[用户转贴]\n" + body
    return body, marked


def other_project(text: str) -> bool:
    if INC_RE.search(text):
        return False
    if not EXC_RE.search(text):
        return False
    if KEEP_ASIDE_RE.search(text):
        return False
    return True


def extract(path: Path, every_project: bool = False) -> dict:
    stem = path.stem
    messages = []
    human_users = 0
    dropped_other = []
    skipped = Counter()
    order = 0
    prev_user = ""
    skip_assistants = False
    seen_events: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                skipped["bad_json"] += 1
                continue
            if not isinstance(obj, dict):
                continue
            kind = obj.get("type")
            if kind not in ("user", "assistant"):
                continue
            if obj.get("isSidechain"):
                skipped["sidechain"] += 1
                continue
            ts = obj.get("timestamp")
            if not isinstance(ts, str):
                skipped["no_time"] += 1
                continue
            msg = obj.get("message") or {}
            role = msg.get("role") or ("user" if kind == "user" else "assistant")
            if role == "user":
                if obj.get("toolUseResult") is not None:
                    skipped["tool_result"] += 1
                    continue
                if obj.get("isCompactSummary"):
                    skipped["compact_summary"] += 1
                    continue
                if obj.get("isMeta"):
                    skipped["meta"] += 1
                    continue
                text = strip_noise(raw_text(msg.get("content"), "user"))
                if not text:
                    skipped["empty_user"] += 1
                    continue
                event_key = ("user", text, ts)
                if event_key in seen_events:
                    skipped["duplicate_event"] += 1
                    continue
                seen_events.add(event_key)
                human_users += 1
                text, pasted = mark_paste(text, prev_user)
                text = redact(text)
                if not text:
                    skipped["empty_user"] += 1
                    continue
                if not every_project and other_project(text):
                    dropped_other.append({"timestamp": ts, "head": text.replace("\n", " ")[:120]})
                    skip_assistants = True
                    prev_user = text[:500]
                    continue
                skip_assistants = False
                prev_user = text[:500]
                messages.append(
                    {
                        "role": "user",
                        "content": text,
                        "timestamp": to_ms(ts),
                        "line": line_no,
                        "order": order,
                        "pasted": pasted,
                    }
                )
                order += 1
            else:
                if skip_assistants:
                    skipped["assistant_after_other_project"] += 1
                    continue
                text = redact(strip_noise(raw_text(msg.get("content"), "assistant")))
                if not text:
                    skipped["empty_assistant"] += 1
                    continue
                event_key = ("assistant", text, ts)
                if event_key in seen_events:
                    skipped["duplicate_event"] += 1
                    continue
                seen_events.add(event_key)
                messages.append(
                    {
                        "role": "assistant",
                        "content": text,
                        "timestamp": to_ms(ts),
                        "line": line_no,
                        "order": order,
                        "pasted": False,
                    }
                )
                order += 1
    messages.sort(key=lambda m: (m["timestamp"], m["order"]))
    return {
        "stem": stem,
        "file": path.name,
        "messages": messages,
        "human_users": human_users,
        "dropped_other": dropped_other,
        "skipped": dict(skipped),
    }
