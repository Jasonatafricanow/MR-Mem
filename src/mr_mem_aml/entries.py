"""Entry compilation: the model writes entries, code assembles the SemanticBlocks.

A trial schema next to "points" (tested 2026-10-17; not in the design doc until
the user has reviewed it). One model call goes through the NEW messages in order
and writes, message by message, what each contributed ("entries"): what it says,
the numbered pieces it comes from, whether it had happened, and what earlier
thing it was built on, each with a quote copied from that target. Code then
builds the SemanticBlocks and their links. The model never groups messages and
never writes a block ref. The output is the points format (content, sources,
after, happened, settles), so Path A ingest is unchanged; entries and pieces are
kept in the compile journal only.

Input (code, no model): background notices and "No response requested." are
dropped; CONTEXT is the last real messages before the window; STEPS lose exact
duplicates and keep their refs; every NEW message is cut into pieces numbered by
code (m4.1, m4.2 ...): a person's message at sentence ends and line breaks, an
assistant's at paragraphs, headings, list items and table rows. Code decides the
language "says" is written in from the person's messages.

Work scene (design 4.1: a block is closed by the person's feedback). Frames come
from the role order: a run of assistant messages and the run of the person's
messages after it.
- The person's entries that use something of this frame's assistant messages
  are its feedback. One feedback takes the whole run as its situation, except
  reports in the run that carry out something else (a late report of other
  work): those stand on their own. Several feedbacks each take the sections of
  the report they point at (a section runs from a heading, a bold line or a
  top-level numbered item to the next one); feedbacks pointing at the same
  sections are one block.
- Assistant entries nobody answered: a report of work done ("did", or carrying
  out something the person asked) is a block of its own; proposals and
  explanations are dropped, and the links they wrote pass to the blocks cut from
  the same message (to the frame's feedback blocks when none is).
- The person's other entries stand on their own (new requests, statements).
- A trailing assistant run nobody has answered yet is carried to the next window.
Chat scene: frames are a run of the user's messages and the assistant run after
it; the assistant's entries go with the user entries they answer, its other
entries stand on their own; nothing is dropped.

Links come only from the entries' "uses" (design 4.2: what is not written is not
linked). A use pointing outside its block, at an earlier block or a STEPS ref,
becomes "after" when its quote is found in the target (the longest contiguous
run covers at least half of the quote); a quote that clearly belongs to one
other candidate (>= 80 %) redirects the link there, logged. Anything else stays
in the audit as unresolved and is never linked. One rule the user approved
(2026-10-17), work scene only: when an assistant report is cut into several
blocks and a part names no source of its own, it takes the sources the same
report names elsewhere (share_report_links, logged); nothing links to "the
block before". In chat an answer is not a report and shares nothing.
"""

from __future__ import annotations

import difflib
import json
import re

CONTEXT_KEEP = 6  # real messages before the window shown as CONTEXT

# --- input -------------------------------------------------------------------

_NOTICE = re.compile(r"^\s*<task-notification>")


def is_noise(role: str, text: str) -> bool:
    """Background notices and the "No response requested." that follows them."""
    text = (text or "").strip()
    return role == "system" or text == "No response requested." or bool(_NOTICE.match(text))


_END = re.compile(r"(?<=[。？！?!])")
_MARK = re.compile(r"^\s*(#{1,6}\s|[-*•]\s|\d+[.)、．]\s?|\|)")
_TABLE_RULE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _user_pieces(text: str) -> list[str]:
    out: list[str] = []
    for line in text.split("\n"):
        for part in _END.split(line):
            if not part.strip():
                continue
            if out and not re.search(r"[\w一-鿿]", part):
                out[-1] += part  # punctuation alone sticks to what it closes
            else:
                out.append(part.strip())
    return out or [text]


def _assistant_pieces(text: str) -> list[str]:
    out: list[str] = []
    cur: list[str] | None = None
    fence = False
    for line in text.split("\n"):
        if fence:
            cur.append(line)
            if line.strip().startswith("```"):
                out.append("\n".join(cur))
                cur, fence = None, False
            continue
        if line.strip().startswith("```"):
            if cur:
                out.append("\n".join(cur))
            cur, fence = [line], True
            continue
        if not line.strip():
            if cur:
                out.append("\n".join(cur))
                cur = None
            continue
        if _TABLE_RULE.match(line) and cur is None and out:
            out[-1] += "\n" + line  # a table's separator row stays with its header
            continue
        if _MARK.match(line):
            if cur:
                out.append("\n".join(cur))
            cur = [line]
            if line.lstrip().startswith("|"):
                out.append(line)
                cur = None
            continue
        if cur is None:
            cur = [line]
        else:
            cur.append(line)
    if cur:
        out.append("\n".join(cur))
    return out or [text]


def pieces_of(role: str, text: str) -> list[str]:
    return _user_pieces(text) if role == "user" else _assistant_pieces(text)


def clean_steps(steps: list[dict]) -> list[dict]:
    """Drop exact duplicates (same text and state); the refs stay as they were."""
    seen, out = set(), []
    for step in steps:
        key = (step.get("state"), re.sub(r"\s+", " ", step.get("text", "")).strip())
        if key not in seen:
            seen.add(key)
            out.append(step)
    return out


def material(new: list[dict], context: list[dict], steps: list[dict], rejected: list) -> dict:
    """``new``: [{"id", "role", "time"?, "text"}] in order; ``context``: [{"role", "text"}]."""
    real = [m for m in context if not is_noise(m["role"], m["text"])]
    return {
        "STEPS": clean_steps(list(steps)),
        "rejected": list(rejected),
        "CONTEXT": [{"role": m["role"], "text": m["text"]}
                    for m in (real[-CONTEXT_KEEP:] if CONTEXT_KEEP else [])],
        "NEW": [
            {"id": m["id"], "role": m["role"], **({"time": m["time"]} if "time" in m else {}),
             "pieces": [{"id": f"{m['id']}.{j}", "text": t}
                        for j, t in enumerate(pieces_of(m["role"], m["text"]), 1)]}
            for m in new if not is_noise(m["role"], m["text"])
        ],
    }


_HAN = re.compile(r"[一-鿿]")
_KANA = re.compile(r"[぀-ヿ]")
_HANGUL = re.compile(r"[가-힯]")
_WORD = re.compile(r"[A-Za-z]+")
_ENGLISH = {"the", "a", "an", "and", "to", "of", "in", "is", "it", "i", "you", "that", "for",
            "my", "what", "this", "with", "on", "be", "are", "was", "do", "have", "can"}


def language_of(texts: list[str]) -> str | None:
    """The language of the person's messages, by script; Latin script counts as
    English only when common English words show it. None when nothing tells."""
    text = "\n".join(texts)
    han, kana, hangul = (len(p.findall(text)) for p in (_HAN, _KANA, _HANGUL))
    words = [w.lower() for w in _WORD.findall(text)]
    cjk = han + kana + hangul
    if cjk and cjk >= len(words):  # one CJK character weighs about one Latin word
        return "Korean" if hangul > han + kana else "Japanese" if kana else "Chinese"
    if words and sum(w in _ENGLISH for w in words) >= 0.1 * len(words):
        return "English"
    return None


def language(material: dict) -> str | None:
    """Decided from the person's NEW messages, else the person's CONTEXT, else all of NEW."""
    users = [p["text"] for m in material["NEW"] if m["role"] == "user" for p in m["pieces"]]
    users = users or [m["text"] for m in material["CONTEXT"] if m["role"] == "user"]
    return language_of(users or [p["text"] for m in material["NEW"] for p in m["pieces"]])


def trailing_run(material: dict, scene: str) -> str | None:
    """Work: the first message of the assistant run that ends the window unanswered."""
    if scene != "work":
        return None
    first = None
    for m in reversed(material["NEW"]):
        if m["role"] != "assistant":
            break
        first = m["id"]
    return first


# --- prompt ------------------------------------------------------------------

# Extraction prompt v2 of the 2026-10-17 trial; changed since: the language of
# "says" is written in by code, and the output is compact with piece ranges.
RULES = """You are the semantic compiler of a long-term memory system.

You receive NEW messages to compile, earlier CONTEXT of the same conversation
(read-only) and STEPS: earlier memory blocks on record, each with a ref (N1,
N2 ...). Every NEW message comes cut into numbered pieces (m3.1, m3.2 ...).

Read NEW together with CONTEXT and STEPS as a whole first. Then go through the
NEW messages in order and write down what each one contributed, as entries.
Code builds the memory blocks and their links from your entries; you do not
group messages or build any structure yourself.

Entries
- One entry is one thing a message contributes: something done or found, a
  result, a proposal, a question, an answer, a decision, a correction, a
  request, a judgement, a statement about oneself or the world. A message that
  says several different things gives several entries, numbered or not; pieces
  that only make sense together stay in one entry. Every piece of the user's
  messages belongs to some entry.
- Messages without lasting content (greetings, filler) give no entry.

For each entry write:
- "msg": the message id; "pieces": the pieces of that message it is drawn from.
- "says": what this message itself contributes, understood, in one or two short
  plain third-person sentences ("The user ...", "The assistant ...")
  in {language}. Say who, what was done, found, decided, asked or proposed, and the
  result that matters, with the modality (fact, decision, plan, intention,
  question, proposal, correction, refusal, reported claim). The original
  message stays on record and is read back with the block, so do not retell it
  or copy its details; keep a concrete specific (a name, number, date, file)
  only when the point itself is that specific. For an assistant's report, say
  what it did or found and what came out, not how. Resolve a relative time
  against the message time when unambiguous and write it as YYYY-MM-DD or
  YYYY-MM-DD HH:MM. Write the message as it stood when it was said: if it is
  mistaken, keep the mistake; never add what later messages correct, decide or
  reveal.
- "happened": true when what it is about had already happened when it was said
  (a decision taken, work reported done or failed, something seen); false when
  it is still going on or ahead (a plan, an intention, a proposal, work still
  running). Asking about something does not make it going on.
- "did": for an assistant's entry, true when it reports work the assistant
  actually did or what came out of it; false for proposals, explanations,
  questions and acknowledgements. Leave it out for the user's entries.
- "uses": the earlier things this entry was built on. Name each that applies:
  - a feedback or an answer: the part of the earlier message it responds to
    (the part of the report it judges, decides on or corrects; the question it
    answers);
  - work reported done: the request or decision that work carries out. Every
    entry of such a report names it, not only the first;
  - any earlier statement it corrects, confirms, changes or comes back to,
    however far back.
  Each item is {"to": a piece of an earlier NEW message ("m3.4") or a STEPS ref
  ("N12"), "quote": 5 to 30 characters copied exactly from that target}. It may
  be several; it is empty when the entry builds on nothing on record. Coming
  next in the conversation, happening at the same time or sharing words does
  not make one thing built on another.
- "settles": STEPS refs of steps that were still going on and that this entry
  shows are now settled (done, decided, dropped). Usually empty.

Return only compact JSON (no indentation, no line breaks), no prose:
{"entries":[{"msg":"m3","pieces":["m3.1-m3.4","m3.7"],"says":"...","happened":true,"did":true,"uses":[{"to":"N2","quote":"..."}],"settles":[]}]}
"pieces" may give consecutive pieces of the message as a range ("m3.1-m3.4").
"""


def prompt(material: dict) -> str:
    lang = language(material)
    said_in = (f"{lang}, the language of the conversation" if lang
               else "the language of the conversation")
    return (RULES.replace("{language}", said_in)
            + "\nINPUT:\n" + json.dumps(material, ensure_ascii=False, indent=1))


# --- assembly ----------------------------------------------------------------

_PIECE = re.compile(r"^(m\d+)\.(\d+)$")
_RANGE = re.compile(r"^(m\d+)\.(\d+)\s*[-–~]\s*(?:m\d+\.)?(\d+)$")
_HEAD = re.compile(r"^(#{1,6}\s|\*\*|\d+[.)、．]\s*)")


def _norm(s: str) -> str:
    return re.sub(r"[\W_]+", "", s or "", flags=re.UNICODE).lower()


def quote_score(quote: str, text: str) -> float:
    """Share of the quote found as one contiguous run in the target (normalized)."""
    q, t = _norm(quote), _norm(text)
    if len(q) < 2 or not t:
        return 0.0
    if q in t:
        return 1.0
    m = difflib.SequenceMatcher(None, q, t, autojunk=False).find_longest_match(0, len(q), 0, len(t))
    return m.size / len(q)


def _expand(raw: object, msg_id: str, valid: dict) -> list[str]:
    """Piece ids of one message; a range "m3.2-m3.9" stands for every piece in it."""
    out: list[str] = []
    for p in [raw] if isinstance(raw, str) else raw if isinstance(raw, list) else []:
        p = str(p).strip()
        m = _RANGE.match(p)
        if m:
            out += [f"{m.group(1)}.{k}" for k in range(int(m.group(2)), int(m.group(3)) + 1)]
        else:
            out.append(p)
    return [p for p in dict.fromkeys(out) if p in valid and p.startswith(msg_id + ".")]


def _pnum(piece: str) -> int:
    return int(_PIECE.match(piece).group(2))


def _leaves(entry: dict, use: dict) -> bool:
    """A use pointing outside the entry's own message."""
    return not (use["to"] == entry["msg"] or use["to"].startswith(entry["msg"] + "."))


class _Assembly:
    def __init__(self, material: dict, scene: str, share_report_links: bool) -> None:
        self.scene = scene
        self.share_report_links = share_report_links
        self.new = material["NEW"]
        self.order = {m["id"]: k for k, m in enumerate(self.new)}
        self.role = {m["id"]: m["role"] for m in self.new}
        self.piece_text = {p["id"]: p["text"] for m in self.new for p in m["pieces"]}
        self.msg_pieces = {m["id"]: [p["id"] for p in m["pieces"]] for m in self.new}
        self.step_text = {s["ref"]: s.get("text", "") for s in material.get("STEPS", [])}
        self.audit: dict = {"bad_entries": [], "dropped": [], "unresolved": [], "redirected": [],
                            "not_earlier": [], "dangling": [], "carried": [], "inherited": [],
                            "shared_report_links": []}
        self.entries: list[dict] = []
        self.by_piece: dict[str, list[int]] = {}
        self.groups: list[dict] = []
        # a dropped entry whose links pass on: (entry, group indices, "message" | "frame")
        self.inherit: list[tuple[int, list[int], str]] = []

    def pos(self, piece: str) -> tuple[int, int]:
        return self.order[piece.split(".")[0]], _pnum(piece)

    def read(self, raw_entries: list) -> None:
        for e in raw_entries:
            mid = str(e.get("msg", "")).strip() if isinstance(e, dict) else ""
            if mid not in self.order:
                self.audit["bad_entries"].append({"entry": str(e)[:200],
                                                  "why": "unknown or noise message"})
                continue
            pieces = _expand(e.get("pieces"), mid, self.piece_text)
            if not pieces:
                pieces = list(self.msg_pieces[mid])
                self.audit["bad_entries"].append({"msg": mid,
                                                  "why": "no valid pieces, whole message used"})
            uses = []
            for u in e.get("uses") if isinstance(e.get("uses"), list) else []:
                if isinstance(u, str):
                    u = {"to": u, "quote": ""}
                if not isinstance(u, dict):
                    continue
                to = str(u.get("to", "")).strip()
                if to:
                    uses.append({"to": to, "quote": str(u.get("quote", "")).strip()})
            settles = e.get("settles") if isinstance(e.get("settles"), list) else []
            self.entries.append({
                "msg": mid, "role": self.role[mid], "pieces": pieces,
                "says": str(e.get("says", "")).strip(),
                "happened": e.get("happened", True) not in (False, None, "false"),
                "did": e.get("did", False) in (True, "true"), "uses": uses,
                "settles": [str(s).strip() for s in settles],
            })
        self.entries.sort(key=lambda e: self.pos(e["pieces"][0]))
        for k, e in enumerate(self.entries):
            e["k"] = k
            for p in e["pieces"]:
                self.by_piece.setdefault(p, []).append(k)

    def in_new(self, to: str) -> list[str]:
        """A use target inside NEW -> piece ids (a bare message id or a slip -> whole message)."""
        if to in self.piece_text:
            return [to]
        m = re.match(r"^(m\d+)(?:\.\d+)?$", to)
        if m and m.group(1) in self.msg_pieces:
            return list(self.msg_pieces[m.group(1)])
        return []

    def frames(self, carry: bool) -> tuple[list[dict], str | None]:
        lead = "assistant" if self.scene == "work" else "user"
        frames: list[dict] = []
        cur = None
        for m in self.new:
            if m["role"] == lead:
                if cur is None or cur["follow"]:
                    cur = {"lead": [], "follow": []}
                    frames.append(cur)
                cur["lead"].append(m["id"])
            else:
                if cur is None:
                    cur = {"lead": [], "follow": []}
                    frames.append(cur)
                cur["follow"].append(m["id"])
        if (carry and self.scene == "work" and frames and frames[-1]["lead"]
                and not frames[-1]["follow"]):
            self.audit["carried"] = frames[-1]["lead"]
            return frames[:-1], frames[-1]["lead"][0]
        return frames, None

    def sections(self, msg: str) -> list[list[str]]:
        out, cur = [], []
        for p in self.msg_pieces[msg]:
            if cur and _HEAD.match(self.piece_text[p]):
                out.append(cur)
                cur = []
            cur.append(p)
        if cur:
            out.append(cur)
        return out

    def chat_frame(self, u_msgs: list[str], a_msgs: list[str]) -> None:
        E = self.entries
        a_ent = [e["k"] for e in E if e["msg"] in a_msgs]
        u_ent = [e["k"] for e in E if e["msg"] in u_msgs]
        answers: dict[frozenset, list[int]] = {}  # the user entries an answer uses
        for k in a_ent:
            used = {t for u in E[k]["uses"] for p in self.in_new(u["to"])
                    for t in self.by_piece.get(p, []) if t in u_ent}
            if used:
                answers.setdefault(frozenset(used), []).append(k)
        taken: set[int] = set()
        for used, ks in answers.items():
            self.groups.append({"entries": sorted(used) + ks})
            taken |= used
        for k in a_ent:
            if not any(k in ks for ks in answers.values()):
                self.groups.append({"entries": [k]})
        for k in u_ent:
            if k not in taken:
                self.groups.append({"entries": [k]})

    def work_frame(self, a_msgs: list[str], u_msgs: list[str]) -> None:
        E = self.entries
        a_ent = [e["k"] for e in E if e["msg"] in a_msgs]
        u_ent = [e["k"] for e in E if e["msg"] in u_msgs]
        a_set = {p for m in a_msgs for p in self.msg_pieces[m]}
        used_by: dict[int, set[str]] = {}  # feedback entry -> the report pieces it uses
        for k in u_ent:
            used = {p for u in E[k]["uses"] for p in self.in_new(u["to"]) if p in a_set}
            if used:
                used_by[k] = used
        section = {p: tuple(s) for m in a_msgs for s in self.sections(m) for p in s}
        feedback: dict[frozenset, list[int]] = {}  # sections pointed at -> feedback entries
        for k, used in used_by.items():
            feedback.setdefault(frozenset(section[p] for p in used), []).append(k)
        frame_groups: list[dict] = []
        separate: list[int] = []
        if len(feedback) == 1:
            # one feedback takes the whole run as its situation, except reports in the
            # run that carry out something else (a late report of other work): those
            # were not answered and stand on their own
            (_, ks), = feedback.items()
            used_k = {t for k in ks for p in used_by[k] for t in self.by_piece.get(p, [])}
            frame_msgs = set(a_msgs) | set(u_msgs)

            def upstream(k: int) -> set[str]:
                return {u["to"].split(".")[0] for u in E[k]["uses"]} - frame_msgs
            base = set().union(*[upstream(k) for k in used_k]) if used_k else set()
            used_msgs = {E[k]["msg"] for k in used_k}
            sep: set[int] = set()
            if base:
                for k in a_ent:  # in order: a report over several messages follows its start
                    if k in used_k or E[k]["msg"] in used_msgs:
                        continue
                    if (upstream(k) and not upstream(k) & base) or any(
                            t in sep for u in E[k]["uses"] for p in self.in_new(u["to"])
                            for t in self.by_piece.get(p, [])):
                        sep.add(k)
            separate = sorted(sep)
            situation = [k for k in a_ent if k not in sep]
            pieces = {p for k in situation for p in E[k]["pieces"]}
            frame_groups.append({"pieces": pieces, "entries": sorted(situation) + sorted(ks)})
            covered = set(pieces)
        else:
            covered = set()
            for secs, ks in feedback.items():
                pieces = {p for s in secs for p in s}
                ents = [k for k in a_ent if set(E[k]["pieces"]) & pieces]
                frame_groups.append({"pieces": pieces, "entries": ents + sorted(ks)})
                covered |= pieces
        left = [k for k in a_ent if not set(E[k]["pieces"]) & covered]
        if feedback:
            did = [k for k in left if E[k]["did"] or k in separate]
        else:
            # nobody answered this run: what carries out a request of the person or
            # reports work done is the report block (the "did" flag alone is not reliable)
            did = [k for k in left if E[k]["did"] or any(
                self.role.get(u["to"].split(".")[0]) == "user" for u in E[k]["uses"])]
        if did:
            frame_groups.append({"pieces": {p for k in did for p in E[k]["pieces"]},
                                 "entries": did})
        first = len(self.groups)
        self.groups += frame_groups
        fb_idx = [first + i for i in range(len(frame_groups) - (1 if did else 0))]
        for k in left:
            if k in did:
                continue
            self.audit["dropped"].append({"msg": E[k]["msg"], "pieces": E[k]["pieces"],
                                          "says": E[k]["says"][:120]})
            same = [g for g in fb_idx
                    if any(p.split(".")[0] == E[k]["msg"] for p in self.groups[g]["pieces"])]
            if E[k]["uses"] and (same or fb_idx):
                self.inherit.append((k, same or fb_idx, "message" if same else "frame"))
        for k in u_ent:
            if k not in used_by:
                self.groups.append({"entries": [k]})

    def run(self, carry: bool) -> dict:
        E = self.entries
        frames, carry_from = self.frames(carry)
        for f in frames:
            if self.scene == "work":
                self.work_frame(f["lead"], f["follow"])
            else:
                self.chat_frame(f["lead"], f["follow"])
        for g in self.groups:
            if "pieces" not in g:
                g["pieces"] = {p for k in g["entries"] for p in E[k]["pieces"]}
            else:  # a situation keeps its section pieces; the person's entries add theirs
                g["pieces"] = set(g["pieces"]) | {
                    p for k in g["entries"] if E[k]["role"] != "assistant" or self.scene == "chat"
                    for p in E[k]["pieces"]}
        idx = sorted(range(len(self.groups)),
                     key=lambda i: min(self.pos(p) for p in self.groups[i]["pieces"]))
        remap = {old: new for new, old in enumerate(idx)}
        groups = [self.groups[i] for i in idx]
        inherit = [(k, [remap[g] for g in gs], via) for k, gs, via in self.inherit]
        for k, gs, via in inherit:
            self.audit["inherited"].append({"msg": E[k]["msg"], "pieces": E[k]["pieces"],
                                            "to": [f"b{g + 1}" for g in gs], "via": via})
        sb_of_piece: dict[str, list[int]] = {}
        for b, g in enumerate(groups):
            for p in g["pieces"]:
                sb_of_piece.setdefault(p, []).append(b)
        blocks = [self.block(b, g, sb_of_piece, inherit) for b, g in enumerate(groups)]
        users = {p for m in self.new if m["role"] == "user" and m["id"] not in self.audit["carried"]
                 for p in self.msg_pieces[m["id"]]}
        covered = {p for bl in blocks for p in bl["pieces"]} & users
        self.audit["user_coverage"] = round(len(covered) / len(users), 3) if users else 1.0
        self.audit["user_missing"] = sorted(users - covered)
        return {"blocks": blocks, "carry_from": carry_from, "audit": self.audit, "entries": E}

    def link_target(self, b: int, to: str, quote: str, sb_of_piece: dict
                    ) -> tuple[list[str] | None, bool, str]:
        """-> (targets, quote found, via) for one use from block b; None = nothing to
        link to; [] = inside the same block (pairing, not a link)."""
        if to in self.step_text:
            targets, score = [to], quote_score(quote, self.step_text[to])
        else:
            pcs = self.in_new(to)
            tsb = sorted({t for p in pcs for t in sb_of_piece.get(p, [])})
            if not tsb:
                return None, False, to
            msg = pcs[0].split(".")[0]
            score = quote_score(quote, "\n".join(self.piece_text[p] for p in self.msg_pieces[msg]))
            if len(tsb) > 1 and quote:
                # several blocks hold this message: keep the one(s) holding the quoted piece
                best: dict[int, float] = {}
                for p in self.msg_pieces[msg]:
                    s = quote_score(quote, self.piece_text[p])
                    if s >= 0.5:
                        for t in sb_of_piece.get(p, []):
                            best[t] = max(best.get(t, 0), s)
                if best:
                    top = max(best.values())
                    tsb = sorted(t for t, s in best.items() if s == top)
            targets = [f"b{t + 1}" for t in tsb if t != b]
            if not targets:
                return [], True, to
        if score >= 0.5:
            return targets, True, to
        if len(_norm(quote)) >= 6:
            # the quote clearly belongs elsewhere: content anchor over the cited id
            hits = [ref for ref, text in {**self.step_text, **self.piece_text}.items()
                    if ref != to and (ref in self.step_text or ref in sb_of_piece)
                    and quote_score(quote, text) >= 0.8]
            if len(hits) == 1:
                h = hits[0]
                tg = [h] if h in self.step_text else [f"b{t + 1}" for t in sb_of_piece[h] if t != b]
                self.audit["redirected"].append({"from": f"b{b + 1}", "cited": to, "to": tg,
                                                 "quote": quote})
                return tg, True, h
        return targets, False, to

    def block(self, b: int, g: dict, sb_of_piece: dict, inherit: list) -> dict:
        E = self.entries
        uses = [(E[k]["msg"], u) for k in g["entries"] for u in E[k]["uses"]]
        # links of a dropped part pass on only when they leave its own message
        uses += [(E[k]["msg"], u) for k, gs, _ in inherit if b in gs for u in E[k]["uses"]
                 if _leaves(E[k], u)]
        if self.share_report_links and self.scene == "work":  # reports are a work-scene notion
            # adopted by the user 2026-10-17: a part of an assistant report that names no
            # source of its own takes the sources the same report names elsewhere (the
            # work it reports was carried out once). Only the model's own links for that
            # message are shared; nothing links to "the block before".
            for m in sorted({E[k]["msg"] for k in g["entries"] if E[k]["role"] == "assistant"}):
                own = [k for k in g["entries"] if E[k]["msg"] == m]
                if any(_leaves(E[k], u) for k in own for u in E[k]["uses"]):
                    continue
                extra = [(m, u) for k in range(len(E)) if E[k]["msg"] == m and k not in own
                         for u in E[k]["uses"] if _leaves(E[k], u)]
                if extra:
                    self.audit["shared_report_links"].append(
                        {"block": f"b{b + 1}", "msg": m, "to": [u["to"] for _, u in extra]})
                    uses += extra
        after: list[str] = []
        for src, u in uses:
            tg, ok, via = self.link_target(b, u["to"], u["quote"], sb_of_piece)
            if tg is None:
                self.audit["dangling"].append({"from": src, "to": u["to"]})
                continue
            for t in tg:  # empty: inside the same block, a pairing and not a link
                if t.startswith("b") and int(t[1:]) - 1 >= b:
                    self.audit["not_earlier"].append({"from": f"b{b + 1}", "to": t, "via": via})
                elif not ok:
                    self.audit["unresolved"].append({"block": f"b{b + 1}", "to": t,
                                                     "via": u["to"], "quote": u["quote"]})
                elif t not in after:
                    after.append(t)
        settles: list[str] = []
        for k in g["entries"]:
            for s in E[k]["settles"]:
                if s in self.step_text and s not in settles:
                    settles.append(s)
        parts = []
        for k in sorted(g["entries"], key=lambda k: self.pos(E[k]["pieces"][0])):
            own = set(E[k]["pieces"]) & g["pieces"]
            # a block holds the understanding, not the text: when only some sections of a
            # report belong here, the report's "says" still stands for it, and the sections
            # themselves are read back from the sources (user 2026-10-20: SBs are a
            # projection, not a notebook)
            parts.append(E[k]["says"])
        last = sorted(g["entries"], key=lambda k: self.pos(E[k]["pieces"][-1]))[-1]
        return {"content": "\n".join(parts),
                "sources": sorted({p.split(".")[0] for p in g["pieces"]}, key=self.order.get),
                "pieces": sorted(g["pieces"], key=self.pos), "after": after,
                "happened": E[last]["happened"], "settles": settles, "entries": g["entries"]}


def assemble(material: dict, raw_entries: list, scene: str = "work", *, carry: bool = False,
             share_report_links: bool = True) -> dict:
    """Entries -> blocks in the points format plus "pieces" and "entries" for the
    journal; ``carry``: a trailing unanswered assistant run (work) is left out and
    its first message returned as ``carry_from``."""
    assembly = _Assembly(material, scene, share_report_links)
    assembly.read(raw_entries)
    return assembly.run(carry)
