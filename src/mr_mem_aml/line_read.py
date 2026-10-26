"""Readout by the upward index (design 5.3, 5.5; round 11 of the ten-day test, 2026-10-26).

What was reasoned once is not reasoned again: the answerer gets the compiled Lines, and a little raw
text for the details an SB leaves in the raw messages.

  1. raw search (P1: BM25 + dense, RRF) gives the raw hits; each hit's record leads up to the SBs
     compiled from it (route 1), and the question against SB texts gives a second list (route 2);
     an SB is scored by reciprocal rank over both: two routes agreeing only raise its rank, they are
     not two pieces of evidence;
  2. the top SBs are walked on the Line graph: up a few steps (where it came from) and down to the
     latest step (what it turned into), in time order, each step marked as happened or in progress;
  3. whole Line pieces fill the line part of the budget; raw snippets fill the rest, first from the
     messages the chosen SBs cite, each the paragraph sharing most words with the question.

Pure functions over callables, so the graph, the stores and the searches stay with the runtime."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from mr_mem.memory.providers.bm25 import lexical_tokens

SIDE = {"zh": {"past": "已发生", "day": "进行中"}, "en": {"past": "happened", "day": "in progress"}}
WHO = {"zh": {"user": "用户", "assistant": "助手"}, "en": {"user": "Person", "assistant": "Agent"}}
HEAD = {"zh": ("【一段】", "【原文片段】"), "en": ("[A line]", "[Raw excerpts]")}


@dataclass(frozen=True, slots=True)
class LineReadPolicy:
    raw_k: int = 30       # raw hits taken from the raw search
    sb_k: int = 30        # SBs taken from the question against SB texts
    sbs: int = 4          # SBs walked on the Line
    up: int = 2           # steps up from each
    down: int = 6         # at most this many steps down to the latest
    line_chars: int = 1200
    budget: int = 2000
    snippet: int = 250
    rrf_k: int = 60
    tz_minutes: int = 0   # times are shown at this offset from UTC (480 = Beijing): the person's dates

    @classmethod
    def of(cls, value: dict | None) -> LineReadPolicy:
        policy = cls(**(value or {}))
        if not 0 < policy.line_chars <= policy.budget or min(policy.raw_k, policy.sb_k, policy.sbs) < 1:
            raise ValueError("line_read needs 0 < line_chars <= budget and positive raw_k, sb_k, sbs")
        return policy


@dataclass(frozen=True, slots=True)
class Step:
    text: str
    time: datetime | None  # None when the source gave no message time
    past: bool


@dataclass(frozen=True, slots=True)
class RawMessage:
    text: str
    role: str
    time: datetime | None


def rank_blocks(raw_records: Sequence[str], blocks_for_record: Callable[[str], Sequence[str]],
                sb_hits: Sequence[str], rrf_k: int = 60) -> list[str]:
    """SB ids by reciprocal rank over route 1 (raw hit -> its SBs) and route 2 (SB search)."""
    score: dict[str, float] = {}
    for rank, record in enumerate(raw_records, 1):
        for block in blocks_for_record(record):
            score[block] = score.get(block, 0.0) + 1 / (rrf_k + rank)
    for rank, block in enumerate(sb_hits, 1):
        score[block] = score.get(block, 0.0) + 1 / (rrf_k + rank)
    first = {b: i for i, b in enumerate(dict.fromkeys([*(b for r in raw_records for b in blocks_for_record(r)), *sb_hits]))}
    return sorted(score, key=lambda b: (-score[b], first[b]))


def walk(block: str, parents: Callable[[str], Sequence[str]], children: Callable[[str], Sequence[str]],
         up: int, down: int) -> set[str]:
    """The block, ``up`` levels of where it came from, and at most ``down`` steps of what it became."""
    out, frontier = {block}, [block]
    for _ in range(up):
        frontier = [p for f in frontier for p in parents(f)]
        out.update(frontier)
    below: list[str] = []
    frontier = [block]
    while frontier and len(below) < down:
        frontier = [c for f in frontier for c in children(f)]
        below += frontier
    return out | set(below[:down])


def snippet(text: str, question: str, limit: int) -> str:
    """The paragraph sharing most words with the question, cut to ``limit`` characters."""
    paras = [p.strip() for p in text.replace("\r", "").split("\n") if p.strip()] or [text.strip()]
    words = set(lexical_tokens(question))
    return max(paras, key=lambda p: len(words & set(lexical_tokens(p))))[:limit]


def _when(t: datetime | None, tz_minutes: int = 0) -> str:
    return "" if t is None else (t + timedelta(minutes=tz_minutes)).strftime("%m-%d %H:%M")


def compose(question: str, ranked: Sequence[str], raw_records: Sequence[str], *,
            step: Callable[[str], Step], sources: Callable[[str], Sequence[str]],
            parents: Callable[[str], Sequence[str]], children: Callable[[str], Sequence[str]],
            message: Callable[[str], RawMessage | None], lang: str, policy: LineReadPolicy) -> list[dict]:
    """Evidence items: Line pieces ("line") first, then raw snippets ("source"), within the budget."""
    side, who, (line_head, raw_head) = SIDE[lang], WHO[lang], HEAD[lang]
    top = list(ranked[:policy.sbs])
    items, used, seen = [], 0, set()
    for n, block in enumerate(top):
        chain = sorted(walk(block, parents, children, policy.up, policy.down) - seen,
                       key=lambda b: (step(b).time is None, step(b).time or datetime.min, b))
        seen.update(chain)
        if not chain:
            continue
        lines = [f"[{_when(step(b).time, policy.tz_minutes)} {side['past' if step(b).past else 'day']}] {step(b).text}".replace("[ ", "[", 1)
                 for b in chain]
        text = line_head + "\n" + "\n".join(lines)
        if used + len(text) > policy.line_chars:
            if not items:  # the first piece alone is too long: cut it to the line part
                items.append({"id": f"line:{block}", "kind": "line", "content": text[:policy.line_chars],
                              "score": float(len(top) - n), "occurred_at": step(block).time})
                used = policy.line_chars
            break
        items.append({"id": f"line:{block}", "kind": "line", "content": text, "score": float(len(top) - n),
                      "occurred_at": step(block).time})
        used += len(text)
    cited = {r for b in top for r in sources(b)}
    order = [r for r in raw_records if r in cited] + [r for r in raw_records if r not in cited]
    head = raw_head + "\n"
    for n, record in enumerate(order):
        m = message(record)
        if m is None:
            continue
        text = head + f"[{_when(m.time, policy.tz_minutes)} {who.get(m.role, m.role)}] {snippet(m.text, question, policy.snippet)}".replace("[ ", "[", 1)
        if used + len(text) > policy.budget:
            break
        head = ""
        items.append({"id": f"raw:{record}", "kind": "source", "content": text, "score": -float(n),
                      "occurred_at": m.time})
        used += len(text)
    return items
