"""The working agent writes SBs from what it already has in context (11-09). No model call here, no
copy of the raw text: this only finds where each SB is in the agent's own session file and appends
it to live/sb.jsonl.

An agent knows neither its session file nor line numbers, so each SB names where it starts by quoting
the opening of a message ("from": a whole short message, or its first words). The session is the most
recently touched file in which all the quotes are found, in order, after what was already written
from it. Each SB runs to just before the next one's start; the last runs to the file's last message
so far. Anything invalid or not found fails the whole write, nothing is appended."""

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path

import origin

STORE = Path(os.environ.get("MRMEM_LIVE_DIR", "live")) / "sb.jsonl"
STATUS = ("confirmed", "likely", "negated", "unknown")
RECENT = 30  # session files looked at, most recently touched first


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s)


def locate(msgs: list[dict], quote: str, after: int) -> int | None:
    """The line of the first message after ``after`` that is the quote; else that starts with it."""
    q = norm(quote)
    later = [(m["line"], norm(m["content"])) for m in msgs if m["line"] > after]
    return next((n for n, t in later if t == q), None) or next(
        (n for n, t in later if t.startswith(q)), None
    )


def load() -> list[dict]:
    if not STORE.exists():
        return []
    return [json.loads(x) for x in STORE.read_text(encoding="utf-8").splitlines() if x.strip()]


def check(sbs: list[dict]) -> None:
    if not sbs:
        raise ValueError("no SBs given")
    for sb in sbs:
        if not str(sb.get("from") or "").strip() or not str(sb.get("text") or "").strip():
            raise ValueError("each SB needs 'from' and 'text'")
        if sb.get("status") not in STATUS:
            raise ValueError(f"status must be one of {STATUS}")
        if not str(sb.get("when") or "").startswith(("before", "after")):
            raise ValueError("when must be 'before' or 'after'")


def write(sbs: list[dict]) -> list[dict]:
    """Append the SBs; returns what was appended."""
    check(sbs)
    written = load()
    for path in origin.recent_files(RECENT):
        msgs = origin.read(path)
        if not msgs:
            continue
        # the message the last write ended on may open the next SB
        at = max(
            (
                r["source"]["last_line"] - 1
                for r in written
                if Path(r["source"]["file"]).name == path.name
            ),
            default=0,
        )
        starts = []
        for sb in sbs:
            at = locate(msgs, sb["from"], at)
            if at is None:
                break
            starts.append(at)
        if len(starts) == len(sbs):
            break
    else:
        raise ValueError("quotes not found in order in any recent session, nothing written")
    now = datetime.now(UTC).isoformat()
    records = [
        {
            "text": sb["text"].strip(),
            "when": sb["when"],
            "status": sb["status"],
            "standing": bool(sb.get("standing", False)),
            "source": {"file": str(path), "first_line": first, "last_line": end},
            "written_at": now,
        }
        for sb, first, end in zip(sbs, starts, [s - 1 for s in starts[1:]] + [msgs[-1]["line"]])
    ]
    STORE.parent.mkdir(exist_ok=True)
    with STORE.open("a", encoding="utf-8") as h:
        for r in records:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    return records
