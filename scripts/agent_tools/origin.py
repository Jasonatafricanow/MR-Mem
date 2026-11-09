"""Where a compiled message is in the agent's own session file (11-09).

The compile input is a cleaned list of messages. This re-derives that list from the original file
with each message's line number, so a record can point at the original instead of at a copy.
A message read back goes through the same cleaning (noise stripped, secrets redacted).
No model call, no copy of the text."""

import json
import re
from datetime import datetime
from itertools import islice
from pathlib import Path

import session_text
from compression import zstd

CLAUDE = Path.home() / ".claude" / "projects"
CODEX = Path.home() / ".codex" / "sessions"
ARCHIVE = CODEX.parent / "archived_sessions"
ASK = "## My request:"
IMAGE = re.compile(r"<image name=\[([^\]]*)\][^>]*>\s*</image>")


def _open(path: Path):
    if path.suffix == ".zst":
        return zstd.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open(encoding="utf-8", errors="replace")


def _is_codex(path: Path) -> bool:
    return path.is_relative_to(CODEX.parent)


def _codex_text(text: str) -> str:
    """The person's words, as clean_codex.person_text keeps them."""
    if text.startswith("# Files mentioned by the user") and ASK in text:
        text = text.split(ASK, 1)[1]
    return IMAGE.sub(lambda m: f"[{m.group(1)}]", text).strip()


def codex_messages(path: Path) -> list[dict]:
    out = []
    with _open(path) as h:
        for n, line in enumerate(h, 1):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            p = e.get("payload") or {}
            if e.get("type") != "response_item" or p.get("type") != "message":
                continue
            role = p.get("role")
            text = "".join(
                c.get("text", "") for c in p.get("content") or [] if isinstance(c, dict)
            ).strip()
            if not text or role not in ("user", "assistant"):
                continue
            if role == "user":
                if text.startswith(("<", "The following is the Codex agent history")):
                    continue
                text = _codex_text(text)
                if not text:
                    continue
            ms = int(
                datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp() * 1000
            )
            out.append(
                {"role": role, "content": session_text.redact(text), "timestamp": ms, "line": n}
            )
    return out


def find_file(stem: str) -> Path | None:
    """A session's file wherever it is now: Codex archives old sessions and compresses them, which
    moves the file and changes its name but not its lines."""
    name = stem.replace("codex-", "rollout-", 1) if stem.startswith("codex-") else stem
    if not name.startswith("rollout-"):
        return next(CLAUDE.glob(f"*/{name}.jsonl"), None)
    for root in (CODEX, ARCHIVE):
        for suffix in (".jsonl", ".jsonl.zst"):
            if (hit := next(root.rglob(name + suffix), None)) is not None:
                return hit
    return None


def current(path: Path) -> Path | None:
    """Where the file once at ``path`` is now, None if it is gone."""
    return (
        path if path.exists() else find_file(path.name.removesuffix(".zst").removesuffix(".jsonl"))
    )


def messages(stem: str) -> tuple[Path, list[dict]] | None:
    """(original file, its messages in compile order with line numbers), or None if the file is gone."""
    path = find_file(stem)
    if path is None:
        return None
    return path, codex_messages(path) if stem.startswith("codex-") else session_text.extract(path)[
        "messages"
    ]


def read(path: Path) -> list[dict]:
    """A session file's messages with line numbers, whatever project it is about; [] if it is gone."""
    now = current(path)
    if now is None:
        return []
    return (
        codex_messages(now)
        if _is_codex(now)
        else session_text.extract(now, every_project=True)["messages"]
    )


def recent_files(n: int) -> list[Path]:
    """The agents' top-level session files, most recently touched first."""
    files = [*CLAUDE.glob("*/*.jsonl"), *CODEX.rglob("*.jsonl"), *ARCHIVE.rglob("*.jsonl")]
    ordered = sorted(files, key=lambda f: -f.stat().st_mtime)
    return list(islice((f for f in ordered if not _codex_subagent(f)), n))


def _codex_subagent(path: Path) -> bool:
    """Codex keeps its subagents' sessions (guardian reviews) beside the person's, with their words."""
    if not _is_codex(path):
        return False
    with path.open(encoding="utf-8", errors="replace") as h:
        try:
            e = json.loads(h.readline())
        except ValueError:
            return False
    source = (e.get("payload") or {}).get("source")
    return e.get("type") == "session_meta" and isinstance(source, dict) and "subagent" in source
