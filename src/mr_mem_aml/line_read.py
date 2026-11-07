"""Readout by the upward index (design 5.3, 5.5; round 11 of the ten-day test, 2026-10-26).

What was reasoned once is not reasoned again: the answerer gets the compiled Lines, and the raw text
the chosen SBs point to, for the details an SB leaves there.

  1. raw search (P1: BM25 + dense, RRF) gives the raw hits; each hit's record leads up to the SBs
     compiled from it (route 1), and the question against SB texts gives a second list (route 2);
     an SB is scored by reciprocal rank over both: two routes agreeing only raise its rank, they are
     not two pieces of evidence;
  2. the top SBs are the anchors (relay): with them the steps along the Line nearest the question
     (``reach``, ``extend``), and the steps on the Line between two of them by their first line
     (design 5.3), in time order: bounded by the anchors, however long the Line;
  3. by the downward index, the raw the chosen SBs were compiled from: the paragraphs each points
     to, or the whole message when it points to all of it, in message order. Nothing is cut to a
     character budget: the cutting was done when the SBs were cut and pointed, so what is given is
     bounded by how many SBs are chosen and how far their Lines are walked. Raw hits only rank SBs.

Pure functions over callables, so the graph, the stores and the searches stay with the runtime."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from difflib import SequenceMatcher, unified_diff

from mr_mem_aml.moves import code_lines, is_code, paragraphs, span_key

SIDE = {
    "zh": {"past": "已发生", "day": "未确认"},
    "en": {"past": "happened", "day": "not confirmed"},
}
WHO = {"zh": {"user": "用户", "assistant": "助手"}, "en": {"user": "Person", "assistant": "Agent"}}
HEAD = {"zh": ("【一段】", "【原文片段】"), "en": ("[A line]", "[Raw excerpts]")}
CARRIED = {"zh": "【这些步骤的原话】", "en": "[The words of these steps]"}
THREAD = {
    "zh": "【问题所在的一段：总结，以及其中和问题最相关的步骤】",
    "en": "[The stretch the question is on: its summary, and its steps nearest the question]",
}
FILL = {
    "zh": "【检索到、上面没有覆盖的其他原文】",
    "en": "[Other raw messages the search found, not covered above]",
}
STANDING = {"zh": "【长期指令和偏好】", "en": "[Standing instructions and preferences]"}
FORGET = {
    "zh": "【人要求忘掉的事：每条只管它说的那一件，人说过的其它事照旧】",
    "en": (
        "[What the person asked to be forgotten: each request covers only the one thing it"
        " names; everything else the person has said still holds]"
    ),
}
FORGOT = re.compile(
    r"\bforget\b|\bforgotten\b|忘掉|忘记|忘了", re.I
)  # an SB written as a request to forget
CHANGES = {"zh": "相对 {} 的改动（- 删去，+ 新增）", "en": "changes from {} (- removed, + added)"}
STAMP = re.compile(
    r"\[(confirmed|likely|negated|unknown)(?:; [^\]]*)?\]\s*$"
)  # the status a chat SB was said with
TOLD = {"zh": "助手对人说", "en": "Agent told the person"}
LATER = {"zh": "（后来有了结果，见 {}）", "en": "(later settled, see {})"}


@dataclass(frozen=True, slots=True)
class LineReadPolicy:
    raw_k: int = 30  # raw hits taken from the raw search
    sb_k: int = 30  # SBs taken from the question against SB texts
    sbs: int = 4  # anchor SBs: the SBs the fused ranking puts first
    rrf_k: int = 60
    tz_minutes: int = (
        0  # times are shown at this offset from UTC (480 = Beijing): the person's dates
    )
    raw_fill: int = (
        0  # P1 where the Lines do not reach: after them and the raw they point to, this many
    )
    # raw hits (search order, whole) that no SB given here comes from
    # which raw the fill takes as already given: "shown", only the words shown; "single", those
    # and the source of an SB over one message; "sources", every source of an SB shown. Compared
    # on the same cached vectors (2026-10-31, BEAM 100k/500k, PM2 32k, LoCoMo): "shown" best on
    # three, above "sources" on all four
    fill_skip: str = "shown"
    # the terminator's marks on the steps shown (as said, "later settled", happened / in progress);
    # False: none, the answer model gets only the two time axes (docs/LCE_FROZEN.md, 2026-10-31)
    marks: bool = True
    # the raw an SB shown brings: "all" its words; "hits" only those the raw search hit
    raw_of_sb: str = "all"
    # every raw hit brings its SB (of a message's SBs the one the question is about) as an anchor,
    # past ``sbs``: without a fill a hit whose SB ranks below ``sbs`` was lost (2026-10-31, BEAM:
    # 128 of the person's 1800 top-30 hits, 12 in the top 10)
    hit_anchors: bool = False
    # each SB shown carries the words it projects (its pointers' paragraphs or characters) inside
    # its own item, beside the raw hits: an SB is a projection, not a record, and the details a
    # question asks for stay in the words (BEAM holdout 11-02: 91% of the gold messages reached an
    # SB, 44% reached the answer as words). Inside the item, not as raw items, so the top_k count
    # holds; ``sb_raw_chars`` caps the words carried in all, the furthest SBs' first (AML cuts the
    # answer input at 117,760 tokens from the end)
    sb_raw: bool = True
    sb_raw_chars: int = 300_000
    # with assistant "told" the assistant's words stay behind the index, carried here too unless
    # asked for: they doubled the readout and buried the person's (BEAM holdout, 11-05)
    sb_raw_assistant: bool = False
    # docs/LCE_FROZEN.md §5: a raw hit is given under the SB it belongs to, with its message time,
    # not in a list of its own. Same messages, same words; only where they sit. A hit with no SB
    # in the Lines (uncompiled, its SB among the standing ones) stays a raw item
    raw_under_sb: bool = True
    # set by compose_within from ``readout_tokens``: the words ``sb_raw`` carries stop within this
    # estimate of tokens
    sb_raw_tokens: int = 0
    # > 0: the whole readout within this estimate of tokens: what ``sb_raw`` carries gets what the
    # rest leaves (AML cuts the answer input from the tail; a fixed cap on the words alone cannot
    # know how long the rest is, nor Chinese from English)
    readout_tokens: int = 0
    # the projection (relay): the anchors, then the steps along the Line nearest the question, and
    # the steps between two of them
    gap: int = 3  # the longest run of steps shown between two anchors
    reach: int = 3  # how far along the Line (either way) from an anchor a step may be
    extend: int = 16  # how many such steps join the anchors, the most relevant first
    thread_k: int = (
        0  # the threads whose summaries are nearest the question: each by its summary and
    )
    thread_sbs: int = (
        12  # its SBs nearest the question (this many), in order - the stretch the question is on
    )
    # each stretch summary given with where it sits, its first and last step (session, position,
    # message time): a stretch carries the time range of its messages, so that the course can be
    # put on the time line (docs/LCE_FROZEN.md 4); without it an old value in one summary and the
    # new one in another could not be told apart
    thread_span: bool = False
    # the person's requests to forget, given apart from the other standing instructions under a
    # line saying each covers only what it names
    forget_scope: bool = False
    assistant: str = (
        "step"  # the assistant's SBs: "step" as steps like the person's; "told": by their first
    )
    # line, under the person's step they answer, as what the person was told (the
    # person decides, design 1.2), their raw not given; "told_raw": so, raw given
    code_diff: bool = (
        False  # code seen in an earlier excerpt is given as its changed lines (git-like)
    )

    @classmethod
    def of(cls, value: dict | None) -> LineReadPolicy:
        policy = cls(**(value or {}))
        if min(policy.raw_k, policy.sb_k, policy.sbs) < 1:
            raise ValueError("line_read needs positive raw_k, sb_k, sbs")
        if policy.assistant not in ("step", "told", "told_raw"):
            raise ValueError("line_read assistant must be 'step', 'told' or 'told_raw'")
        return policy


@dataclass(frozen=True, slots=True)
class Step:
    text: str
    time: datetime | None  # None when the source gave no message time
    past: bool
    place: tuple[int, int] | None = None  # (session, message number) of its first source: the order
    later: tuple[tuple[int, int], ...] = ()  # places of the later steps that settled it
    session: datetime | None = None  # no message time: the time the source gave its session so far


@dataclass(frozen=True, slots=True)
class RawMessage:
    text: str
    role: str
    time: datetime | None
    place: tuple[int, int] | None = None
    session: datetime | None = None  # as Step.session


def rank_blocks(
    raw_records: Sequence[str],
    blocks_for_record: Callable[[str], Sequence[str]],
    sb_hits: Sequence[str],
    rrf_k: int = 60,
    closest: Callable[[Sequence[str]], str] | None = None,
) -> list[str]:
    """SB ids by reciprocal rank over route 1 (raw hit -> its SBs) and route 2 (SB search).
    ``closest(blocks)``: of a hit message's SBs, the one the question is about; the hit's rank
    goes to it alone, so a message cut into many SBs does not push the next hits down (the
    others still come by route 2). None: every SB of the message takes the hit's rank."""
    score: dict[str, float] = {}
    for rank, record in enumerate(raw_records, 1):
        blocks = list(blocks_for_record(record))
        for block in [closest(blocks)] if closest and len(blocks) > 1 else blocks:
            score[block] = score.get(block, 0.0) + 1 / (rrf_k + rank)
    for rank, block in enumerate(sb_hits, 1):
        score[block] = score.get(block, 0.0) + 1 / (rrf_k + rank)
    first = {
        b: i
        for i, b in enumerate(
            dict.fromkeys([*(b for r in raw_records for b in blocks_for_record(r)), *sb_hits])
        )
    }
    return sorted(score, key=lambda b: (-score[b], first[b]))


def between(hits: set[str], parents: Callable[[str], Sequence[str]], gap: int) -> set[str]:
    """The steps on the Line between two hit steps: going up from a hit, those passed on the way to
    another hit at most ``gap`` steps up. The projection is bounded by the hits, not by the Line."""
    out: set[str] = set()
    for hit in hits:
        frontier = [(hit, ())]
        for _ in range(gap + 1):
            nxt = []
            for node, path in frontier:
                for p in parents(node):
                    if p in hits:
                        out.update(path)
                    elif p not in path:
                        nxt.append((p, (*path, p)))
            frontier = nxt
    return out


def near(
    anchors: Sequence[str],
    parents: Callable[[str], Sequence[str]],
    children: Callable[[str], Sequence[str]],
    reach: int,
) -> set[str]:
    """The steps at most ``reach`` edges from an anchor, up or down the Line, anchors excluded."""
    seen, frontier = set(anchors), list(anchors)
    for _ in range(reach):
        frontier = [x for f in frontier for x in (*parents(f), *children(f)) if x not in seen]
        seen.update(frontier)
    return seen - set(anchors)


def pieces(steps: set[str], parents: Callable[[str], Sequence[str]]) -> list[set[str]]:
    """The steps grouped by the Line edges among them."""
    group = {s: {s} for s in steps}
    for s in steps:
        for p in parents(s):
            if p in steps and group[p] is not group[s]:
                merged = group[p] | group[s]
                for x in merged:
                    group[x] = merged
    return list({id(g): g for g in group.values()}.values())


def cut_chars(text: str, spans) -> str:
    """The character spans of a message, in order, joined; "…" where something is left out."""
    if spans is None:
        return text.strip()
    out, last = [], None
    for s, e in sorted(spans):
        if last is not None and s < last:  # overlaps the one before
            if e > last:
                out[-1] += text[last:e]
                last = e
            continue
        piece = text[s:e].strip()
        if piece:
            if out and text[last:s].strip():
                out.append("…")
            out.append(piece)
            last = e
    return "\n".join(out) or text.strip()


def cut(text: str, keep: set | None) -> str:
    """The parts ``keep`` of a message, as step 1 numbered them: paragraphs k (1-based) and
    (k, first line, last line) of a code paragraph ("…" where something is left out); the whole
    message for None or when none of them is there."""
    if keep is None:
        return text.strip()
    paras, out, last = paragraphs(text), [], None
    for item in sorted(keep, key=span_key):
        k = item if isinstance(item, int) else item[0]
        if not 1 <= k <= len(paras):
            continue
        if isinstance(item, int):
            piece, at = paras[k - 1], (k, 0)
        else:
            lines = code_lines(paras[k - 1]) if is_code(paras[k - 1]) else paras[k - 1].split("\n")
            piece, at = "\n".join(lines[max(item[1], 1) - 1 : item[2]]), (k, item[1])
        if not piece:
            continue
        if out and last is not None and not (at[0] == last + 1 and at[1] <= 1):
            out.append("…")
        out.append(piece)
        last = k
    return "\n".join(out) or text.strip()


SESSION_FROM = {"zh": "会话始于 {}", "en": "session began {}"}


def _when(
    t: datetime | None,
    tz_minutes: int = 0,
    place: tuple[int, int] | None = None,
    session: datetime | None = None,
    lang: str = "en",
) -> str:
    """The message time when the source gave one, then where it sits: session and message number.
    With no message time, the date the source gave its session, said as the session's, not the
    step's."""
    at = "" if t is None else (t + timedelta(minutes=tz_minutes)).strftime("%Y-%m-%d %H:%M")
    where = "" if place is None else f"S{place[0]} #{place[1]}"
    if t is None and session is not None:
        where += ("，" if lang == "zh" else ", ") + SESSION_FROM[lang].format(
            (session + timedelta(minutes=tz_minutes)).strftime("%Y-%m-%d")
        )
    return " ".join(x for x in (at, where.lstrip("，, ")) if x)


def estimate_tokens(text: str) -> int:
    """About what a model counts: four characters of English, one and a half of Chinese."""
    wide = sum(ord(c) > 127 for c in text)
    return int((len(text) - wide) / 4 + wide / 1.5)


def _order(s: Step) -> tuple:
    return (s.place is None, s.place or (0, 0), s.time is None, s.time or datetime.min)


def compose(
    question: str,
    ranked: Sequence[str],
    raw_records: Sequence[str],
    *,
    fill_records: Sequence[str] | None = None,
    uncompiled_records: Sequence[str] = (),
    step: Callable[[str], Step],
    sources: Callable[[str], Sequence[str]],
    parents: Callable[[str], Sequence[str]],
    children: Callable[[str], Sequence[str]],
    message: Callable[[str], RawMessage | None],
    lang: str,
    policy: LineReadPolicy,
    standing: Sequence[str] = (),
    span: Callable[[str, str], Sequence[int] | None] = lambda block, record: None,
    relevance: Callable[[str], float] = lambda block: 0.0,
    pointers: Callable[[str], dict | None] = lambda block: None,
    threads: Callable[[], dict[str, dict]] = dict,
    thread_hits: Sequence[str] = (),
    hit_blocks: Sequence[str] = (),
    blocks_of: Callable[[str], Sequence[str]] = lambda record: (),
) -> list[dict]:
    """Evidence items: the person's standing instructions and preferences first, whatever the
    question (they apply to what is asked, not to what it is about), then Line pieces ("line"),
    then the raw the chosen SBs point to ("source"). ``span(block, record)``: the paragraphs of
    that source message the block points to, None for the whole message. ``relevance(block)``:
    how close a step is to the question. ``pointers(block)``: the fragments of the words it comes
    from ({"own": {record: spans | None}, "parts": [...]}), None for the old spans."""
    side, who, (line_head, raw_head) = SIDE[lang], WHO[lang], HEAD[lang]

    def at_b(b: str) -> str:  # where and when a step sits
        s = step(b)
        return _when(s.time, policy.tz_minutes, s.place, s.session, lang)

    def at_m(m: RawMessage) -> str:  # where and when a message sits
        return _when(m.time, policy.tz_minutes, m.place, m.session, lang)

    top = list(ranked[: policy.sbs])
    if policy.hit_anchors:  # ``hit_blocks``: the SB of each raw hit, in raw-search order
        top = list(dict.fromkeys([*top, *hit_blocks]))
    items, seen = [], set()

    def unmarked(text: str) -> str:  # the terminator is the memory side's own (docs/LCE_FROZEN.md)
        first = text.split("\n")[0]
        return text if policy.marks else STAMP.sub("", first).rstrip() + text[len(first) :]

    if standing:
        chain = sorted(standing, key=lambda b: (*_order(step(b)), b))

        def said_as(b: str) -> str:
            return f"[{at_b(b)}] {unmarked(step(b).text)}".replace("[ ", "[", 1).replace(
                "[] ", "", 1
            )

        # a request to forget is scoped to the one thing it names: given apart and said so, as
        # listed among the standing instructions it read as "keep away from all of this"
        # (PersonaMem: related preferences never asked to be forgotten were avoided too)
        forget = [
            b for b in chain if policy.forget_scope and FORGOT.search(step(b).text.split("\n")[0])
        ]
        rest = [b for b in chain if b not in forget]
        text = "\n".join(
            [
                *([STANDING[lang], *map(said_as, rest)] if rest else []),
                *([FORGET[lang], *map(said_as, forget)] if forget else []),
            ]
        )
        items.append(
            {
                "id": "standing",
                "kind": "line",
                "content": text,
                "score": float(len(top) + 1),
                "occurred_at": step(chain[-1]).time,
            }
        )
        seen = set(chain)

    def line(
        b: str, whole: bool = True
    ) -> str:  # a step between two hits gives its first line only
        text = step(b).text if whole else step(b).text.split("\n")[0]
        if not policy.marks:
            return f"[{at_b(b)}] {unmarked(text)}".replace("[ ", "[", 1).replace("[] ", "", 1)
        stamp = STAMP.search(text.split("\n")[0])
        if stamp:  # it carries its own status as said; the Line adds only what came later
            later = (
                f" {LATER[lang].format(', '.join(f'S{a} #{n}' for a, n in step(b).later))}"
                if step(b).later
                else ""
            )
            return f"[{at_b(b)}] {text}{later}".replace("[ ", "[", 1).replace("[] ", "", 1)
        return f"[{at_b(b)} {side['past' if step(b).past else 'day']}] {text}".replace("[ ", "[", 1)

    said = lambda b: any((m := message(r)) is not None and m.role == "user" for r in sources(b))  # noqa: E731

    def show(
        chain: list[str],
        whole: Callable[[str], bool] = lambda b: True,
        hung: Callable[[str], Sequence[str]] = lambda b: (),
    ) -> str:
        """The steps in order; with "told", each SB of the assistant by its first line under the
        person's step it answers (the nearest step of the person above it here), or where it is.
        ``hung``: the raw hits given right under their step."""
        if policy.assistant == "step":
            return "\n".join(x for b in chain for x in (line(b, whole(b)), *hung(b)))
        here, under = set(chain), {}
        for b in chain:
            if said(b):
                continue
            host, front, met = None, [b], {b}
            while front and host is None:
                front = [x for f in front for x in parents(f) if x not in met]
                met.update(front)
                host = next((x for x in front if x in here and said(x)), None)
            under.setdefault(host, []).append(b)

        def told(b: str, indent: str) -> str:
            text = STAMP.sub("", step(b).text.split("\n")[0]).rstrip()
            return f"{indent}[{at_b(b)} {TOLD[lang]}] {text}".replace("[ ", "[", 1)

        out = []
        for b in chain:
            if said(b):
                out += [
                    line(b, whole(b)),
                    *hung(b),
                    *(told(a, "  - ") for a in under.get(b, ())),
                ]
            elif b in under.get(None, ()):
                out.append(told(b, ""))
        return "\n".join(out)

    # the fused search finds anchors; the Line around them finds the rest
    around = sorted(
        near(top, parents, children, policy.reach) - seen, key=lambda b: (-relevance(b), b)
    )
    top = [*top, *around[: policy.extend]]
    if (
        thread_hits
    ):  # the question hit these threads by their summaries: each with its nearest steps
        known = threads()
        for n, t in enumerate(x for x in thread_hits if x in known):
            mine = [b for b in known[t]["members"] if step(b).text and b not in seen]
            near_ = sorted(
                sorted(mine, key=lambda b: (-relevance(b), b))[: policy.thread_sbs],
                key=lambda b: (*_order(step(b)), b),
            )
            seen |= set(near_)
            span = ""
            if policy.thread_span:  # where the stretch sits: its first and its last step
                here = sorted(
                    (b for b in known[t]["members"] if step(b).text),
                    key=lambda b: (*_order(step(b)), b),
                )
                if here:
                    a, z = at_b(here[0]), at_b(here[-1])
                    span = f"[{a}{' - ' + z if z != a else ''}] "
            items.append(
                {
                    "id": f"thread:{t}",
                    "kind": "line",
                    "score": float(len(top)) + 0.4 - n / 100,
                    "occurred_at": step(near_[0]).time if near_ else None,
                    "content": THREAD[lang]
                    + "\n"
                    + span
                    + known[t]["summary"]
                    + ("\n" + show(near_) if near_ else ""),
                }
            )
    def downward() -> list[tuple[str, RawMessage, str]]:
        """The raw given: (record, message, its line), in message order."""
        # downward index: what the chosen SBs point to, whether or not the raw search hit it; a
        # message one SB points to whole is given whole
        keep: dict[str, set[int] | None] = {}
        chars: dict[str, list | None] = {}  # fragments: character spans cut with the SB

        def merge(record: str, p) -> None:
            have = keep.get(record, set())
            keep[record] = None if p is None or have is None else have | set(p)

        def merge_chars(record: str, p) -> None:
            have = chars.get(record, [])
            chars[record] = None if p is None or have is None else have + [tuple(x) for x in p]

        # "hits": of an SB's words only the messages the raw search hit (an SB over a whole talk
        # would pour all its messages out and crowd the hits out of the budget; 2026-10-31, LoCoMo)
        hit = set(raw_records) if policy.raw_of_sb == "hits" else None
        for block in top:
            if policy.assistant == "told" and not said(
                block
            ):  # what the person was told: by its index only
                continue
            # a message the raw search hit is given whole: the search matched the message, not the
            # part of it this SB points to, and a message cut into several SBs would otherwise show
            # only the part of the SB chosen for it
            if (ptr := pointers(block)) is None:
                for record in sources(block):
                    if hit is None or record in hit:
                        merge(record, None if hit is not None else span(block, record))
                continue
            own = {
                r: p for r, p in ptr["own"].items() if not any(r in d for d in ptr["parts"])
            }  # a part points into it
            for d in [own, *ptr["parts"]]:
                for record, p in d.items():
                    if hit is not None and record not in hit:
                        continue
                    if hit is not None:
                        merge(record, None)
                    elif ptr.get("chars"):
                        merge_chars(record, p)
                    else:
                        merge(
                            record,
                            None
                            if p is None
                            else [tuple(x) if isinstance(x, list) else x for x in p],
                        )
        # A raw hit without a usable SB must survive even when optional fill is off.
        for record in uncompiled_records:
            merge(record, None)
        found = [(r, m) for r in dict.fromkeys([*keep, *chars]) if (m := message(r)) is not None]
        found.sort(
            key=lambda x: (
                x[1].place is None,
                x[1].place or (0, 0),
                x[1].time is None,
                x[1].time or datetime.min,
            )
        )
        earlier: list[tuple[str, list[str]]] = []  # code already given: (where, its lines)

        def changes(text: str, where: str) -> str:  # code seen before: only its changed lines
            def one(m: re.Match) -> str:
                body = m.group(2).split("\n")
                near = max(
                    earlier, key=lambda e: SequenceMatcher(None, e[1], body).ratio(), default=None
                )
                earlier.append((where, body))
                if near is None or SequenceMatcher(None, near[1], body).ratio() < 0.5:
                    return m.group(0)
                diff = [
                    x
                    for x in unified_diff(near[1], body, lineterm="", n=0)
                    if x[:1] in "+-" and not x.startswith(("+++", "---"))
                ]
                return (
                    f"{m.group(1)}\n# {CHANGES[lang].format(near[0])}\n" + "\n".join(diff) + "\n```"
                )

            return re.sub(r"(```[^\n]*)\n(.*?)\n?```", one, text, flags=re.S)

        given_ = []
        for record, m in found:
            text = (
                cut_chars(m.text, chars[record]) if record in chars else cut(m.text, keep[record])
            )
            if policy.code_diff:
                text = changes(text, _when(None, 0, m.place) or record)
            given_.append(
                (record, m, f"[{at_m(m)} {who.get(m.role, m.role)}] {text}".replace("[ ", "[", 1))
            )
        return given_

    given: list[tuple[str, RawMessage, str]] | None = None
    under_sb: set[str] = set()
    carried_budget = [policy.sb_raw_chars]
    carried_tokens = [policy.sb_raw_tokens]
    hanging: dict[str, list[tuple[RawMessage, str]]] = {}

    def hang(blocks: Sequence[str], m: RawMessage, text: str) -> bool:
        """Under the closest of ``blocks`` (the SB a hit's rank goes to, rank_blocks); under "told"
        an assistant's step gives its first line only, so a person's step holds it."""
        at = [b for b in blocks if policy.assistant != "told" or said(b)]
        if at:
            hanging.setdefault(max(at, key=relevance), []).append((m, text))
        return bool(at)

    def hung_under(b: str) -> list[str]:  # what hangs under a step, in message order
        return [
            text
            for _, text in sorted(
                hanging.get(b, ()),
                key=lambda x: (x[0].place is None, x[0].place or (0, 0), x[0].time or datetime.min),
            )
        ]

    carried: set[str] = set(raw_records)  # the raw hits are given as raw items anyway

    def words(chain: list[str]) -> str:
        """policy.sb_raw: the words the steps of ``chain`` point to, in message order, each
        message once in the readout, within what is left of ``sb_raw_chars``. With
        ``raw_under_sb`` each goes under its step and nothing is returned."""
        want: dict[str, list] = {}
        pointed: dict[str, list[str]] = {}
        for block in chain:
            if (ptr := pointers(block)) is None:
                for record in sources(block):
                    want.setdefault(record, []).append(("keep", span(block, record)))
                    pointed.setdefault(record, []).append(block)
                continue
            for d in [ptr["own"], *ptr["parts"]]:
                for record, p in d.items():
                    want.setdefault(record, []).append(("chars" if ptr.get("chars") else "keep", p))
                    pointed.setdefault(record, []).append(block)
        got = sorted(
            (
                (r, m)
                for r in want
                if r not in carried
                and (m := message(r)) is not None
                and (policy.sb_raw_assistant or policy.assistant != "told" or m.role == "user")
            ),
            key=lambda x: (x[1].place is None, x[1].place or (0, 0), x[1].time or datetime.min),
        )
        out = []
        for record, m in got:
            ps = want[record]
            if any(p is None for _, p in ps):
                given = m.text.strip()
            elif all(kind == "chars" for kind, _ in ps):
                given = cut_chars(m.text, [tuple(x) for _, p in ps for x in p])
            else:
                given = cut(
                    m.text,
                    {
                        tuple(x) if isinstance(x, list) else x
                        for kind, p in ps
                        if kind == "keep"
                        for x in p
                    },
                )
            line = f"[{at_m(m)} {who.get(m.role, m.role)}] {given}".replace("[ ", "[", 1)
            cost = estimate_tokens(line)
            if len(line) > carried_budget[0] or (policy.sb_raw_tokens and cost > carried_tokens[0]):
                break
            carried_budget[0] -= len(line)
            carried_tokens[0] -= cost
            carried.add(record)
            if policy.raw_under_sb:
                hang(pointed[record], m, line)
            else:
                out.append(line)
        return ("\n" + CARRIED[lang] + "\n" + "\n".join(out)) if out else ""

    hits = [b for b in top if b not in seen]
    rank = {b: n for n, b in enumerate(hits)}
    shown = (set(hits) | between(set(hits), parents, policy.gap)) - seen
    if policy.raw_under_sb:
        given = downward()
        for record, m, text in given:
            if hang([b for b in blocks_of(record) if b in shown], m, text):
                under_sb.add(record)
    for group in sorted(
        pieces(shown, parents), key=lambda g: min(rank.get(b, len(hits)) for b in g)
    ):
        chain = sorted(group, key=lambda b: (*_order(step(b)), b))
        best = min(rank.get(b, len(hits)) for b in group)
        head = (
            hits[best] if best < len(hits) else chain[0]
        )  # only steps between hits given elsewhere
        listed = words(chain) if policy.sb_raw else ""  # before show: it may hang them
        items.append(
            {
                "id": f"line:{head}",
                "kind": "line",
                "content": line_head
                + "\n"
                + show(chain, lambda b: b in rank, hung_under)
                + listed,
                "score": float(len(top) - best),
                "occurred_at": step(head).time,
            }
        )
    if given is None:
        given = downward()
    rest = [g for g in given if g[0] not in under_sb]
    for n, (record, m, text) in enumerate(rest):
        items.append(
            {
                "id": f"raw:{record}",
                "kind": "source",
                "content": (raw_head + "\n" if n == 0 else "") + text,
                "score": -float(n),
                "occurred_at": m.time,
            }
        )
    if policy.raw_fill:  # fusion: the raw search fills in what the compiled Lines do not cover
        # an SB shows a summary, not the words it comes from: under "sources" an assistant's long
        # answer told in one line, or a LoCoMo nickname in a talk, never reached the answer
        stand_in = {
            r
            for b in {*top, *seen}
            if policy.fill_skip == "sources"
            or (policy.fill_skip == "single" and len(sources(b)) == 1)
            for r in sources(b)
        }
        covered = {*(g[0] for g in given), *stand_in}
        pool = (
            raw_records if fill_records is None else fill_records
        )  # P1's own list, as deep as it goes
        extra = [(r, m) for r in pool if r not in covered and (m := message(r)) is not None][
            : policy.raw_fill
        ]
        for k, (record, m) in enumerate(extra):
            text = (
                FILL[lang] + "\n" if k == 0 else ""
            ) + f"[{at_m(m)} {who.get(m.role, m.role)}] {m.text}".replace("[ ", "[", 1)
            items.append(
                {
                    "id": f"raw:{record}",
                    "kind": "source",
                    "content": text,
                    "score": -float(len(rest) + k),
                    "occurred_at": m.time,
                }
            )
    return items


def compose_within(question: str, ranked: Sequence[str], raw_records: Sequence[str], **kw) -> list:
    """compose; with ``readout_tokens``, the readout without the carried words is measured first,
    and they get what it leaves."""
    policy = kw.pop("policy")
    if not (policy.readout_tokens and policy.sb_raw):
        return compose(question, ranked, raw_records, policy=policy, **kw)
    rest = compose(question, ranked, raw_records, policy=replace(policy, sb_raw=False), **kw)
    used = sum(estimate_tokens(i["content"]) for i in rest)
    left = max(1, policy.readout_tokens - used)  # 0 would mean no cap
    return compose(question, ranked, raw_records, policy=replace(policy, sb_raw_tokens=left), **kw)
