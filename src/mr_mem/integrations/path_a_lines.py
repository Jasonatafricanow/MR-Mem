"""Which existing lines the semantic compiler sees when it compiles new messages.

The compiler decides whether new content continues an existing line (Thread or
accepted LCE lineage) or starts a new one, so it must see the relevant part of
the projection layer. The layer keeps growing; each compile reads only the lines
that matter now, from three sources:

- recent: lines most recently touched, by message time;
- cued: lines the new messages themselves recall (lexical BM25, plus dense
  cosine when an embedder is given, fused by reciprocal rank);
- due: open lines whose summary names a time that has arrived (or arrives within
  ``due_horizon``): a time-bound plan must evolve once its time comes.

A line that is not selected is dormant for this compile only. It stays in the
projection layer, and read paths can still recall it. Times are message times,
never the wall clock, so a replay selects the same lines.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from mr_mem.memory.providers.bm25 import lexical_tokens

# Compiler rule: resolved dates are written as YYYY-MM-DD, optionally HH:MM.
_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2}))?\b")


@dataclass(frozen=True, slots=True)
class LineCard:
    question: str
    summary: str | None
    settled: bool  # the matter has ended (past tense); False = still developing
    last_active: datetime  # message time of the latest Block on the line
    search_text: str  # question, summary and a few supporting Blocks


@dataclass(frozen=True, slots=True)
class LineSelection:
    lines: tuple[LineCard, ...]
    recent: int
    cued: int
    due: int


def stated_times(text: str) -> tuple[datetime, ...]:
    """Absolute times written in the compiler's fixed format (UTC-naive -> UTC)."""
    out = []
    for match in _DATE.finditer(text or ""):
        year, month, day, hour, minute = match.groups()
        try:
            out.append(datetime(int(year), int(month), int(day),
                                int(hour or 0), int(minute or 0), tzinfo=UTC))
        except ValueError:
            continue
    return tuple(out)


def _bm25_order(query: str, docs: Sequence[str], k1: float = 1.2, b: float = 0.75) -> list[int]:
    q = set(lexical_tokens(query))
    if not q or not docs:
        return []
    toks = [lexical_tokens(d) for d in docs]
    avg = sum(len(t) for t in toks) / len(toks) or 1.0
    df = Counter(term for t in toks for term in set(t))
    n = len(docs)
    scores = []
    for i, t in enumerate(toks):
        tf = Counter(t)
        s = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            s += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * len(t) / avg))
        if s > 0:
            scores.append((s, i))
    scores.sort(key=lambda item: (-item[0], item[1]))
    return [i for _, i in scores]


def _cosine_order(query: str, docs: Sequence[str],
                  embed: Callable[[str], Sequence[float]]) -> list[int]:
    qv = embed(query)
    qn = math.sqrt(sum(x * x for x in qv)) or 1.0
    scored = []
    for i, d in enumerate(docs):
        dv = embed(d)
        dn = math.sqrt(sum(x * x for x in dv)) or 1.0
        scored.append((sum(a * c for a, c in zip(qv, dv, strict=True)) / (qn * dn), i))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [i for _, i in scored]


def select_known_lines(
    cards: Sequence[LineCard],
    *,
    batch_text: str,
    now: datetime,
    recent: int = 10,
    cued: int = 10,
    due: int = 10,
    due_horizon: timedelta = timedelta(hours=24),
    embed: Callable[[str], Sequence[float]] | None = None,
    rrf_k: int = 60,
) -> LineSelection:
    """Pick the lines the compiler sees for one batch (deterministic)."""
    chosen: dict[str, LineCard] = {}

    def take(card: LineCard) -> bool:
        if card.question in chosen:
            return False
        chosen[card.question] = card
        return True

    # due: open time-bound lines whose stated time has come (earliest first)
    due_cards = []
    for card in cards:
        if card.settled:
            continue
        times = [t for t in stated_times(f"{card.question}\n{card.summary or ''}")
                 if t <= now + due_horizon]
        if times:
            due_cards.append((min(times), card.question, card))
    due_cards.sort(key=lambda item: (item[0], item[1]))
    n_due = sum(take(card) for _, _, card in due_cards[:due])

    by_recency = sorted(cards, key=lambda c: (c.last_active, c.question), reverse=True)
    n_recent = sum(take(card) for card in by_recency[:recent])

    rest = [c for c in cards if c.question not in chosen]
    n_cued = 0
    if rest and cued > 0 and batch_text.strip():
        texts = [c.search_text for c in rest]
        fused: dict[int, float] = {}
        orders = [_bm25_order(batch_text, texts)]
        if embed is not None:
            orders.append(_cosine_order(batch_text, texts, embed))
        for order in orders:
            for rank, i in enumerate(order):
                fused[i] = fused.get(i, 0.0) + 1.0 / (rrf_k + rank + 1)
        for i, _ in sorted(fused.items(), key=lambda item: (-item[1], item[0]))[:cued]:
            n_cued += take(rest[i])
    return LineSelection(tuple(chosen.values()), n_recent, n_cued, n_due)
