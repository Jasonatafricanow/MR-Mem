"""Work scene, two-step compile (user 2026-10-23/24; design 4.1, 4.2, 4.6).

The work scene is centred on the person: the person decides, the agent carries out. An SB is
"what happened -> what the person did":

  step 1 (one call per window) writes the SBs: one per point the person makes (a message with
         several independent points gives several), each with the agent messages it responds to,
         the situation in plain words (the agent's own wording, self-assessment and pet phrases
         washed out) and the person's move. An SB is a projection, not a notebook (user
         2026-10-22/24): the
         situation in one or two sentences, the move as what the person settled; the details stay
         in the raw text. A report nobody answered is no SB of its own: when the person answers
         it, it goes into that point's situation. No earlier steps are shown: writing and linking
         are separate.
  step 2 (one call per window) places them: each SB gets candidates retrieved with its own text
         (runtime), and the model says which of them it builds on - the decision being carried
         out, or the person's own earlier decision or judgement brought back, changed or reversed,
         however far back. Sharing a concept is not building on.

The chat scene is centred on the dialogue (the interaction is the content) and does not use this
module. The prompt follows the person's language, decided once per user ("zh" or "en"); the rules
are
the same in both, only the wording, the examples and the output language differ.
Strings never quote with ASCII double quotes inside the JSON (models break the JSON with them)."""

from __future__ import annotations

import json
import re
from collections import Counter

ARROW = {"zh": " → 人：", "en": " -> Person: "}
NO_MOVE = {"zh": "（无回应）", "en": "(no response)"}
_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z]+")


def person_language(texts: list[str]) -> str:
    """ "zh" or "en" from the person's own messages only (the agent's process talk is often in
    another language: a mixed session follows the person). Chinese characters are counted
    against English words: code names in a Chinese sentence do not make it English."""
    cjk = sum(len(_CJK.findall(t)) for t in texts)
    words = sum(len(_WORD.findall(t)) for t in texts)
    return "zh" if cjk and cjk >= words else "en"


def content_of(situation: str, move: str, lang: str) -> str:
    """A move the person raises on their own has no situation: the content starts at the arrow."""
    if not move:
        return situation
    return f"{situation}{ARROW[lang]}{move}" if situation else f"{ARROW[lang].lstrip()}{move}"


def split_content(content: str) -> tuple[str, str]:
    """(situation, move) of an SB written by this module; move is "" for content without the
    arrow."""
    for arrow in ARROW.values():
        if content.startswith(arrow.lstrip()):
            return "", content[len(arrow.lstrip()) :]
        if arrow in content:
            situation, move = content.split(arrow, 1)
            return situation, move
    return content, ""


# --- step 1 ---------------------------------------------------------------------

STEP1 = {
    "zh": (
        "你是长期记忆系统的语义编译器。下面是人和编程 Agent 的工作对话。工作场景以人为主：人是"
        "决策主体，内容是项目的真实进度和问题。把 NEW 编成语义块（SB）：每个 SB 是“发生的情况 "
        "→ 人的动作”。先把 NEW 连同 CONTEXT 整体读懂，再写。\n"
        "\n"
        "输出\n"
        "moves：人的每一点各一项。人一条消息里有几处各自独立的意思（有没有编号都一样），就是几"
        "项。\n"
        "  - msg：人这条消息的 M 编号。\n"
        "  - about：这一点针对的 Agent 消息（M 编号，通常是人开口前那一两条汇报里和这一点有关"
        "的部分）；人主动提出的新需求可以为空。\n"
        "  - situation：只写 about 里 Agent 那边带来的、人这一点所回应的那部分：出了什么问题、"
        "得到了什么结果、提了什么方案或问题。一句话，最多 60 字。Agent 列了好几点、人只回应其"
        "中一两点，就只写那一两点；人只是同意或让它继续，就只写它提议做什么。点出关键的具体对"
        "象（哪个功能、哪个文件、起决定作用的那个数字），单独读得懂，不写“这个问题”“该方案”这"
        "种指代。不写 Agent 的执行过程、逐项清单和汇报里的枝节，不沿用它的措辞和自我评价；细节"
        "留在原文里，SB 是理解不是转述。人说的话不写进 situation；about 为空（人主动提出）时 s"
        "ituation 留空。\n"
        "  - move：只写人在这一点上定下了什么：决定了什么、纠正成什么、否定了什么、要求做什么"
        "、问了什么。一两句，最多 80 字，写出他定下的具体内容（比如选定的键、格式、取舍、标准"
        "），保留他的关键词，不整段抄原话，不重复 situation 已写的内容。人同意或让 Agent 照做"
        "时，写出他采纳的具体内容（比如「同意导出改用 CSV、按日期分文件」），不只写「同意」「"
        "让它继续」。\n"
        "  - happened：说的时候这件事是否已经发生。人的决定、纠正、反馈本身是已发生（true）；"
        "人提出的以后要做的计划是 false。\n"
        "几个会话交错在一个窗口里时，消息前标着会话（会话A、会话B……）：不同会话是不同的事，人"
        "的一点只回应同一会话里的 Agent 消息，about 只列同一会话的，situation 也只写那个会话里"
        "的事。\n"
        "没人回应的 Agent 汇报不单独成 SB：人后来回应它，就写进那一点的 situation，about 列上"
        "它；人始终没回应，不编。\n"
        "situation 和 move 都用第三人称写：人写作「人」，Agent 写作「Agent」（原文里 Agent 对"
        "人说的「你」指人，「我」指 Agent），不用「你」「我」。\n"
        "内容里说到的相对时间，按消息时间换成 YYYY-MM-DD 或 YYYY-MM-DD HH:MM。\n"
        "\n"
        "用中文写。只输出紧凑 JSON，字符串里不要用英文双引号（引用用「」）：\n"
        '{{"moves":[{{"msg":"M5","about":["M3","M4"],"situation":"...","move'
        '":"...","happened":true}}]}}\n'
        "\n"
        "CONTEXT（之前的消息，只供理解，不编译）：\n"
        "{context}\n"
        "\n"
        "NEW：\n"
        "{new}"
    ),
    "en": (
        "You are the semantic compiler of a long-term memory system. Below is a work conversat"
        "ion between a person and a coding agent. A work conversation is centred on the person"
        ": the person decides, and the content is the real progress of the project and its pro"
        'blems. Compile NEW into semantic blocks (SBs): each SB is "what happened -> what the'
        ' person did". Read NEW together with CONTEXT as a whole first, then write.\n'
        "\n"
        "Output\n"
        "moves: one item per point the person makes. A message with several independent points"
        " (numbered or not) gives several items.\n"
        "  - msg: the M id of the person's message.\n"
        "  - about: the agent messages this point responds to (M ids; usually the part of the "
        "report just before the person spoke that this point is about); empty for a new reques"
        "t the person raises on their own.\n"
        "  - situation: only the part of what the agent brought in about that this point of th"
        "e person responds to: what went wrong, what came out, what it proposed or asked. One "
        "sentence, at most 40 words. When the agent lists several points and the person answer"
        "s one or two, write only those; when the person just agrees or lets it go on, write o"
        "nly what it proposed to do. Name the key specific object (which feature, which file, "
        'the number that decides it) so the SB reads on its own, never "this issue" or "the'
        " proposal\". Leave out the agent's process, item-by-item lists and the side details o"
        "f its report, and do not carry over its wording or self-assessment; the details stay "
        "in the raw text: an SB is understanding, not a retelling. What the person said does n"
        "ot go into the situation; when about is empty (the person raises it on their own) the"
        " situation is empty.\n"
        "  - move: only what the person settled on it: what they decided, corrected it to, ref"
        "used, asked to be done, or asked. One or two sentences, at most 50 words, giving the "
        "specific content they settled (the key, format, trade-off or standard they chose), ke"
        "eping their key words; do not copy the whole message or repeat what the situation alr"
        "eady says. When the person agrees or lets the agent go ahead, write the specific cont"
        'ent they adopted (e.g. "agrees to switch the export to CSV, one file per day"), not'
        ' just "agrees" or "lets it go on".\n'
        "  - happened: whether it had already happened when said. The person's decision, corre"
        "ction or feedback itself has (true); a plan the person sets for later has not (false)"
        ".\n"
        "When several sessions interleave in one window, each message is marked with its sessi"
        "on (session A, session B ...): different sessions are different work; a point of the "
        "person responds only to agent messages of the same session, about lists only those, a"
        "nd the situation is about that session only.\n"
        "A report of the agent that nobody answered is no SB of its own: when the person answe"
        "rs it later, it goes into that point's situation, with it in about; if the person nev"
        "er answers it, it is not compiled.\n"
        'Write situation and move in the third person: "the person" and "the agent" (in th'
        'e messages, the agent\'s "you" is the person and its "I" is the agent); never "yo'
        'u" or "I".\n'
        "Write a relative time in the content as YYYY-MM-DD or YYYY-MM-DD HH:MM, resolved agai"
        "nst the message time.\n"
        "\n"
        "Write in English. Return only compact JSON, no double quotes inside strings:\n"
        '{{"moves":[{{"msg":"M5","about":["M3","M4"],"situation":"...","move'
        '":"...","happened":true}}]}}\n'
        "\n"
        "CONTEXT (earlier messages, for understanding only, not compiled):\n"
        "{context}\n"
        "\n"
        "NEW:\n"
        "{new}"
    ),
}

WHO = {"zh": {"user": "人", "assistant": "Agent"}, "en": {"user": "Person", "assistant": "Agent"}}
SESSION = {"zh": "会话", "en": "session "}


def paragraphs(text: str) -> list[str]:
    """A message's paragraphs as step 1 numbers them and the readout cuts them: the blocks between
    blank lines, a fenced code block whole and on its own. A message of one is one paragraph."""
    out, cur, fence = [], [], False
    for line in text.replace("\r", "").split("\n"):
        opens = line.lstrip().startswith("```") and not fence
        if opens and cur:  # code starts a paragraph of its own
            out.append("\n".join(cur))
            cur = []
        if line.lstrip().startswith("```"):
            fence = not fence
        if not line.strip() and not fence:
            if cur:
                out.append("\n".join(cur))
                cur = []
            continue
        cur.append(line)
        if (
            line.lstrip().startswith("```") and not fence and not opens
        ):  # code closed: its paragraph ends
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return out if len(out) >= 2 else [text.strip()]


def is_code(paragraph: str) -> bool:
    return paragraph.lstrip().startswith("```")


def code_lines(paragraph: str) -> list[str]:
    """The lines inside a code paragraph's fences, as step 1 numbers them (from 1)."""
    body = paragraph.split("\n")[1:]
    return body[:-1] if body and body[-1].strip().startswith("```") else body


def piece_spans(text: str) -> list[tuple[int, int]]:
    """Where each paragraph of paragraphs(text) sits in text: (start, end) character offsets."""
    out, pos = [], 0
    for p in paragraphs(text):
        i = text.find(p, pos)
        i = pos if i < 0 else i
        out.append((i, i + len(p)))
        pos = i + len(p)
    return out


def item_spans(text: str, items) -> list[list[int]]:
    """Paragraph numbers k and (k, first line, last line) of a code paragraph, as character
    spans."""
    spans, out = piece_spans(text), []
    for item in sorted(items, key=span_key):
        k = item if isinstance(item, int) else item[0]
        if not 1 <= k <= len(spans):
            continue
        s, e = spans[k - 1]
        if isinstance(item, int):
            out.append([s, e])
            continue
        para = text[s:e]
        lines = code_lines(para) if is_code(para) else para.split("\n")
        pos, at = (para.find("\n") + 1 if is_code(para) else 0), []
        for line in lines:
            at.append(pos)
            pos += len(line) + 1
        lo, hi = max(item[1], 1), min(item[2], len(lines))
        if lo <= hi:
            out.append([s + at[lo - 1], s + at[hi - 1] + len(lines[hi - 1])])
    return out


PLAIN = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"'}


def _fold(text: str) -> tuple[str, list[int]]:
    """Lowercased, whitespace runs as one space, quotes made plain; with each kept character's
    place."""
    out, where, space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not space and out:
                out.append(" ")
                where.append(i)
            space = True
            continue
        out.append(PLAIN.get(ch, ch).lower())
        where.append(i)
        space = False
    return "".join(out), where


def anchor(text: str, first: str, last: str) -> list[int] | None:
    """The span from the words ``first`` to the words ``last`` as step 1 copied them, or None;
    the shortest such span when the words come more than once."""
    a, b = _fold(str(first or "").strip())[0], _fold(str(last or "").strip())[0]
    if not a:
        return None
    t, where = _fold(text)
    best, i = None, t.find(a)
    if i < 0:
        return None
    while i >= 0:
        j = t.find(b, i) if b else -1
        end = j + len(b) if j >= 0 else i + len(a)
        if j >= 0 or not b:
            if best is None or end - i < best[1] - best[0]:
                best = (i, end)
        i = t.find(a, i + 1)
    if best is None:  # the last words not found after any: the first words alone
        i = t.find(a)
        best = (i, i + len(a))
    return [where[best[0]], where[best[1] - 1] + 1]


def _numbered(lang: str, label: str, message: dict, tag: str = "") -> str:
    """A NEW message; a long one in numbered paragraphs (M3.1, M3.2 ...) that turns may point to."""
    paras = paragraphs(message["text"])
    if len(paras) < 2:
        return _line(lang, label, message, None, tag)
    body = []
    for k, p in enumerate(paras, 1):
        if is_code(p):  # numbered lines, so a part can point to the lines it is about
            body.append(
                f"({label}.{k}) {p.split(chr(10))[0]}\n"
                + "\n".join(f"  {j}| {line}" for j, line in enumerate(code_lines(p), 1))
                + "\n  ```"
            )
        else:
            body.append(f"({label}.{k}) {p}")
    return _line(lang, label, {**message, "text": ""}, 0, tag).rstrip() + "\n" + "\n".join(body)


def _line(lang: str, label: str, message: dict, limit: int | None, tag: str = "") -> str:
    who = WHO[lang].get(message["role"], message["role"])
    when = f" {message['time']}" if message.get("time") else ""
    return f"[{label}{when}{tag} {who}] {message['text'][:limit]}"


def step1_prompt(lang: str, new: list[dict], context: list[dict]) -> str:
    """new / context: {"role", "text", "time"?, "session"?}; NEW messages are M1, M2 ... in order.
    Sessions are marked A, B ... (order of first appearance) only when more than one is present."""
    order = list(dict.fromkeys(m.get("session") for m in [*context, *new]))
    tag = (
        (lambda m: f" {SESSION[lang]}{chr(65 + order.index(m.get('session')) % 26)}")
        if len(order) > 1
        else (lambda m: "")
    )
    return STEP1[lang].format(
        context="\n".join(_line(lang, "", m, 600, tag(m)).replace("[ ", "[", 1) for m in context)
        or "-",
        new="\n".join(_line(lang, f"M{i + 1}", m, None, tag(m)) for i, m in enumerate(new)),
    )


def parse_step1(data: dict, roles: list[str], sessions: list | None = None) -> list[dict]:
    """SBs in the order their last message was said: {"sources": [message indexes], "situation",
    "move", "happened"}. Unknown ids are dropped; a move whose msg is not the person's is dropped;
    about keeps only earlier agent messages of the person's own session (``sessions`` per
    message)."""
    if not isinstance(data, dict) or not isinstance(data.get("moves"), list):
        raise ValueError("'moves' must be a list")
    ids = {f"M{i + 1}": i for i in range(len(roles))}
    out = []
    for m in data["moves"]:
        if not isinstance(m, dict) or ids.get(str(m.get("msg", "")).strip()) is None:
            continue
        i = ids[str(m["msg"]).strip()]
        if roles[i] != "user" or not str(m.get("move", "")).strip():
            continue
        about = {
            ids[a]
            for a in m.get("about") or []
            if isinstance(a, str)
            and a.strip() in ids
            and roles[ids[a.strip()]] != "user"
            and ids[a.strip()] < i
            and (sessions is None or sessions[ids[a.strip()]] == sessions[i])
        }
        out.append(
            {
                "sources": sorted(about | {i}),
                "situation": str(m.get("situation", "")).strip(),
                "move": str(m["move"]).strip(),
                "happened": m.get("happened") is not False,
            }
        )
    out.sort(key=lambda b: (b["sources"][-1], b["sources"][0]))
    return out


def check_step1(data: dict, roles: list[str], sessions: list | None = None) -> None:
    """A step-1 answer that leaves most of the person's messages without an SB is a broken answer
    (seen with deepseek-v4-flash: now and then one item and done), not a reading: ask again."""
    sbs = parse_step1(data, roles, sessions)
    people = sum(r == "user" for r in roles)
    covered = {b["sources"][-1] for b in sbs if b["move"]}
    if people >= 2 and len(covered) * 2 < people:
        raise ValueError(f"SBs for {len(covered)} of the person's {people} messages")


def tail_start(roles: list[str], scene: str = "work") -> int | None:
    """Where the unfinished end of a window starts (it goes to the next window). Work: an unanswered
    agent run, which the person's answer will close; chat: the person's last words before the
    assistant answers (with no assistant in the window, as between people, nothing waits)."""
    if scene == "chat":
        if not roles or roles[-1] != "user" or "assistant" not in roles:
            return None
        start = len(roles) - 1
        while start > 0 and roles[start - 1] == "user":
            start -= 1
        return start
    last_user = max((i for i, r in enumerate(roles) if r == "user"), default=None)
    if last_user is None or last_user == len(roles) - 1:
        return None
    return last_user + 1


# --- step 1, slim4 (trial, 2026-10-26) ---------------------------------------------
# The SB written as in the bbz_diag slim4 prompt: one SB per stretch whose "what was done, how it
# came out" has one answer, its turns spanning the agent's messages too, the summary being what was
# done, the result and the person's decision or reservation. Only the interface differs from slim4:
# M ids, a CONTEXT, no links (step 2 places). Chinese and English (the person's language); the work
# and the chat scene alike.

STEP1_SLIM4 = {
    "zh": (
        '你是 SemanticBlock（SB）编译器。把对话编译成若干 SB，每个 SB 记录"一件事"。\n'
        "\n"
        "## 切分规则\n"
        '1. 先通读全文再切。一个 SB 是"这一段做了什么、结果怎样"只有一个答案的一段。切点落在'
        "这个答案变了的地方：做的事变了、结果变了，或出现了转折（被纠正、被否决、出了意外、得"
        "出结论）。\n"
        '   - 同一个大目标下的不同进展，只要各有各的结果，就是不同的 SB。例如"出门→打车→没赶'
        '上航班"是三个 SB，不是一个。\n'
        "   - 一个请求和对它的执行、汇报、答复，如果只得出同一个结果，是一个 SB。\n"
        "   - 后一段要靠前一段的背景才讲得通，这不是合并的理由。\n"
        '2. 没有自己结果的内容，不单独成 SB：只说"现在去做 X"的过程播报、一句没人接的话、没'
        "展开的术语，归入它所属或所引出的那个 SB（不单独写出，细节留在原文）。\n"
        "3. 解释一个词或概念、或由此引出的旁枝：有来有回（至少一问一答，或接着聊了几句）、有自"
        "己的结果，就单独成 SB；只提了一句、没人接，归入当前 SB，不单独写出。\n"
        '4. 不按表面话题词、轮次或"一问一答"配对机械切分；看起来像新话题、其实在佐证当前这件'
        "事的，归入当前 SB。\n"
        '5. 消解相对时间和指代："今天/昨天/这个/它"写成具体日期或对象。回忆中的"今天"指回'
        "忆那天，不是说话那天。每条消息有时间可用。\n"
        "6. 某字段在原文里没有体现就填 null，不要编。\n"
        "\n"
        "## unresolved\n"
        "7. 待定 = 这件事已经推进出了结果，下一步只能由人拍板，而且到对话结束人都没有回应。填"
        "人该拍板的那条消息的 M 编号，否则填 null。\n"
        '   判断看"下一步在谁手里"，不看措辞：要等实验、运行、数据或别的事情发生才能定的，不'
        "是在等人，不算待定；只是提了个想法、问了个问题、还没有结果的，不算待定；后面已经回答"
        "、决定或做了的，不算待定。\n"
        "\n"
        "## 每个 SB 的字段\n"
        "- id\n"
        '- turns: 这件事覆盖的 NEW 消息范围，如 "M3-M7"（只有一条写 "M3"）；长消息按段落编'
        '了号（M3.1、M3.2……），只有其中几段属于这件事时写到段落，如 "M3.2-M3.5"\n'
        "- summary: 一段短摘要，只写这一件事的三样东西：做了什么、结果怎样、人在其中的决定或保"
        "留（没有就不写这一样）。细节、数字、原话都留在原文，不要写进来。摘要不是复述。\n"
        "- unresolved: 见规则 7\n"
        "\n"
        "## 输出\n"
        '只输出合法 JSON 数组，不要其他文字，字符串里不要用英文双引号（引用用「」）：[{{"id"'
        ':"b1","turns":"M1-M4","summary":"...","unresolved":null}}]\n'
        "\n"
        "CONTEXT（之前的消息，只供理解，不编译）：\n"
        "{context}\n"
        "\n"
        "NEW：\n"
        "{new}"
    ),
    "en": (
        "You are a SemanticBlock (SB) compiler. Compile the conversation into SBs; each SB rec"
        'ords "one matter".\n'
        "\n"
        "## Cutting\n"
        '1. Read it all before cutting. An SB is a stretch where "what was done or said, and '
        'how it came out" has a single answer. Cut where that answer changes: what is being d'
        "one changes, the result changes, or there is a turn (corrected, refused, something un"
        "expected, a conclusion reached).\n"
        "   - Different steps toward one larger goal are different SBs when each has its own r"
        'esult. E.g. "left home -> took a taxi -> missed the flight" is three SBs, not one.'
        "\n"
        "   - A request and its handling, report and answer are one SB when they reach a singl"
        "e result.\n"
        "   - That a later stretch needs an earlier one as background is not a reason to merge"
        " them.\n"
        '2. Content with no result of its own is not an SB of its own: a "now I will do X" p'
        "rogress note, a remark nobody picks up, an unexplained term go into the SB they belon"
        "g to or lead to (not written out; the details stay in the raw text).\n"
        "3. Explaining a word or concept, or a side branch it leads to: with a real exchange ("
        "at least a question and an answer, or a few more lines) and a result of its own, it i"
        "s its own SB; mentioned once and not picked up, it goes into the current SB, not writ"
        "ten out.\n"
        "4. Do not cut mechanically by surface topic words, by turns or by question-answer pai"
        "rs; what looks like a new topic but supports the current matter goes into the current"
        " SB.\n"
        '5. Resolve relative time and reference: write "today / yesterday / this / it" as th'
        'e concrete date or object. In a recollection, "today" is the day recalled, not the '
        "day it is said. Every message has its time.\n"
        "6. A field the text does not show is null; do not invent.\n"
        "\n"
        "## unresolved\n"
        "7. Pending = the matter has come to a result and the next step can only be taken by t"
        "he person, who has not responded by the end of the conversation. Give the M id of the"
        " message where the person should decide, else null.\n"
        "   Judge by whose hands the next step is in, not by wording: what waits for an experi"
        "ment, a run, data or some other event is not waiting for the person; an idea floated "
        "or a question asked with no result yet is not pending; what is later answered, decide"
        "d or done is not pending.\n"
        "\n"
        "## Fields of each SB\n"
        "- id\n"
        '- turns: the NEW messages this matter covers, e.g. "M3-M7" ("M3" for one); a long'
        " message is shown in numbered paragraphs (M3.1, M3.2 ...): when only some of them bel"
        'ong to this matter, give those paragraphs, e.g. "M3.2-M3.5"\n'
        "- summary: a short summary of only three things about this matter: what was done or s"
        "aid, how it came out, and the person's decision or reservation in it (leave out what "
        "is not there). When several people talk, say who said what. Details, numbers and exac"
        "t words stay in the raw text; do not put them here. A summary is not a retelling.\n"
        "- unresolved: see rule 7\n"
        "\n"
        "## Output\n"
        'Return only a valid JSON array, nothing else, no double quotes inside strings: [{{"i'
        'd":"b1","turns":"M1-M4","summary":"...","unresolved":null}}]\n'
        "\n"
        "CONTEXT (earlier messages, for understanding only, not compiled):\n"
        "{context}\n"
        "\n"
        "NEW:\n"
        "{new}"
    ),
}


SUMMARY = {
    "zh": (
        "- summary: 一段短摘要，只写这一件事的三样东西：做了什么、结果怎样、"
        "人在其中的决定或保留（没有就不写这一样）。细节、数字、原话都留在原文，"
        "不要写进来。摘要不是复述。"
    ),
    "en": (
        "- summary: a short summary of only three things about this matter: what was done or s"
        "aid, how it came out, and the person's decision or reservation in it (leave out what "
        "is not there). When several people talk, say who said what. Details, numbers and exac"
        "t words stay in the raw text; do not put them here. A summary is not a retelling."
    ),
}
# The chat scene (SB freeze 2026-09-25; user 2026-10-27): an SB is a semantic point, one thing
# someone meant at that time, cut by meaning and not by message; who meant it and how firmly is
# written in its text, not split into fields. Its status is happened, not yet or unknown.
STEP1_CHAT = {
    "zh": (
        "你是 SemanticBlock（SB）编译器。把下面的聊天编译成 SB，并写出它们之间谁在谁的基础上推"
        "进。\n"
        "\n"
        "SB 是一件事：某一方在那个时点说出的一个完整意思，离开原文也不会被误读，并且对应原文里"
        "的一段话。人说的话决定这件事是什么；助手的回答是接在人这件事后面的步骤。\n"
        "\n"
        "## 怎么切\n"
        '1. 按意思切，不按句子、条目、段落切。人的一条消息里说了几件不同的事，就是几个 SB（"'
        '我测了是 150ms，怎么更快？"是一次测量和一个提问）；人交代的已经做了的、测到的、现在'
        "的状态，和他要的是不同的意思，即使是为了引出请求才说的，也各自成 SB；人提出一个需求、"
        "又对它提了几条意思不同的要求，需求和各条要求分开。同一类东西连着罗列（一串想用的库、"
        "一张待办清单、一组例子）是一个意思，是一个 SB，具体有哪些留在原文里；罗列里某一项人另"
        "外说了什么（单独的决定、事实、理由），它才单独成 SB。一件事要人的几条消息才完整，就是"
        "一个 SB。\n"
        "2. 助手的回复按它带来的新意思来切：对某件事给出的方案、做法、结论、解释，意思不同的各"
        "是一个 SB；同一个意思展开成几步、几条、几段代码，仍是一个 SB。助手复述、汇总已经说过"
        "的东西（列出人还有哪些待办、回顾前面的方案）不是新的事，不为其中每一项再立 SB，整段复"
        "述至多是一个 SB。\n"
        "3. 寒暄、过程播报这类没有意思的话不成 SB。\n"
        "\n"
        "## 怎么写\n"
        "4. text：一句话写这件事，写明是谁（人……／助手……），按说话者的本意，相对时间和指代换成"
        "具体的；代码留在原文。人的 SB 保留别人可能问到的值（名称、版本、日期、数字）。助手的 "
        'SB 只写它给了什么、为人的哪件事给的（"助手给了一份和 HR 谈福利的通话话术"），不列出'
        "里面的条目和值：事情是人推进的，助手给的东西是人决策的参考，具体内容留在原文，靠索引"
        "取。\n"
        "5. 原文说了的写清，没说的不补；不写后来才发生的事。\n"
        "\n"
        "## 推进\n"
        "6. 在本窗口里某个 SB 的基础上推进的 SB，写进那个 SB 的 next：大需求里的具体事写在大需"
        "求下面；助手的回答写在它回答的那件事下面；接着做下去的一步写在它接着的那一步下面。一"
        "个 SB 下面有几个 next，就是从它分出几条。不在本窗口任何 SB 基础上的，放在最外层。\n"
        "\n"
        "## 状态\n"
        "7. when：从说这句话的那一刻看，这件事在时间轴上的位置（它在对话里的位置，不是任何钟表"
        '时间）。"before"：那一刻之前已经发生或已经成立（做完的事、事实、做出的决定、提出的'
        '请求、成立的状态）。"after"：在那之后；原文给了日期就写上（"after 2024-03-31"）。'
        "一个安排和它安排的事位置不同：安排（人的计划、项目排期、截止日、给出的指令）说的时候"
        "已经存在，是 before；它安排的事（计划执行、冲刺结束、按期完成、照指令去做）是 after，"
        "分成不同的 SB。\n"
        "8. status 跟着 when 定：\n"
        '   - before：发生了或成立填 "confirmed"；没发生填 "negated"。\n'
        '   - after：已经安排好的（人定了或坚定要做，或排期、截止日定了它）填 "likely"；取消'
        '、拒绝、决定不做的填 "negated"；只是提议或推荐、人没接的填 "unknown"。\n'
        '   - 两边都没有证据：填 "unknown"。\n'
        "   likely 的事以后由新消息确认或否定，这里不判断。\n"
        '   助手的 SB 说的是它给出的那件事，不是"助手说了"：助手说了，不等于那件事发生了或成'
        "立了。计划的一项、推荐的做法、让人去用的代码，是 after·unknown；它陈述的事实、解释、"
        "判断和预期，是人被告知的内容，说过去或现在的是 before、说以后的是 after，人确认、照做"
        "或报告结果之前都是 unknown。\n"
        "\n"
        "## standing\n"
        '9. 人给出的、从此以后都要遵守的指令或偏好（"以后都……""每次……的时候……""我更喜欢……'
        '"），standing 填 true；其余填 false。\n'
        "\n"
        "## turns\n"
        "10. turns 指向原话：消息按段落编了号（M3.1、M3.2……），代码段的行也编了号。SB 指向它那"
        '段话所在的段落或行（"M3.2"、"M4.3:5-18"），整条消息都是这件事才写 "M3"；并照原'
        "文抄下那段话开头的几个词（from）和结尾的几个词（to），让代码找到它的确切位置；指向代"
        "码行时可以不抄。\n"
        "\n"
        "## 在哪里停\n"
        "NEW 不必在一次回答里编完：一次回答最多大约 {cap} 个 SB。按顺序编，每条消息都照规则切"
        "足。下一条消息会让你超出这个数时，就停在它前面，停在一个此前所有 SB 都已完整收口的消"
        "息之后；后面的消息在下一次回答里编，这些作为上下文。不要为了编到最后而缩短、合并或漏"
        "掉 SB。\n"
        "\n"
        "## 输出\n"
        "先写 ANALYSIS：用平常的话按顺序过一遍消息：人的每条消息说了哪几件事，一件件点名；助手"
        "对每件事回答了哪几块。然后另起一行写 JSON:，后面只跟合法 JSON 数组，字符串里不要用英"
        "文双引号（引用用「」）：\n"
        '[{{"turns":"M1","from":"...","to":"...","text":"人……","when":"befo'
        're","status":"confirmed","standing":false}},\n'
        ' {{"turns":"M1","from":"...","to":"...","text":"人请求……","when":"'
        'before","status":"confirmed","standing":false,"next":[\n'
        '   {{"turns":"M1","from":"...","to":"...","text":"人要……","when":"'
        'after","status":"likely","standing":false,"next":[\n'
        '     {{"turns":"M2.3-M2.5","from":"...","to":"...","text":"助手建议……'
        '","when":"after","status":"unknown","standing":false}}]}},\n'
        '   {{"turns":"M2.4:3-12","text":"助手给出……","when":"before","status":'
        '"unknown","standing":false}}]}}]\n'
        "JSON 之后另起一行写 STOP: M<k>，k 是你编到的最后一条消息。\n"
        "\n"
        "CONTEXT（之前的消息，只供理解，不编译）：\n"
        "{context}\n"
        "\n"
        "NEW：\n"
        "{new}"
    ),
    "en": (
        "You are a SemanticBlock (SB) compiler. Compile the chat below into SBs and write whic"
        "h builds on which.\n"
        "\n"
        "An SB is one thing: one complete meaning said by one side at that time, which cannot "
        "be misread without the raw text, and which sits in a stretch of the words. What the p"
        "erson says defines what a thing is; the assistant's answers are steps that follow the"
        " person's thing.\n"
        "\n"
        "## Cutting\n"
        "1. Cut by meaning, not by sentences, items or paragraphs. A person's message that say"
        's several different things gives several SBs ("I measured 150ms; how can I make it f'
        'aster?" is a measurement and a question); what the person says is already done, meas'
        "ured or the current state means something other than what they want, even when they s"
        "ay it to lead up to the request, so each is an SB of its own; when the person brings "
        "up a need and several requirements for it that mean different things, the need and ea"
        "ch requirement are SBs. Things of one kind listed together (a string of libraries the"
        "y want, a to-do list, a set of examples) are one meaning and one SB; which ones they "
        "are stays in the raw text. An item of such a list is an SB of its own only when the p"
        "erson says something more about it (a decision, a fact, a reason of its own). One thi"
        "ng that takes several of the person's messages to be complete is one SB.\n"
        "2. The assistant's reply is cut by the new meanings it brings: for a thing, the plans"
        ", measures, conclusions and explanations it gives that mean different things are SBs;"
        " one meaning spread over several steps, items or pieces of code is still one SB. When"
        " the assistant restates or sums up what was already said (listing the person's remain"
        "ing to-dos, going over an earlier plan), that is not a new thing: no SB for each item"
        "; the restatement is at most one SB.\n"
        "3. Small talk and progress notes that carry no meaning are not SBs.\n"
        "\n"
        "## Writing an SB\n"
        "4. text: the thing in one sentence, saying who (the person ... / the assistant ...), "
        "as the speaker meant it, with relative times and references made concrete; code stays"
        " in the raw text. An SB of the person keeps the values someone may ask for (names, ve"
        "rsions, dates, numbers). An SB of the assistant says only what it gave and for which "
        'matter of the person ("The assistant gives a script for the benefits call with HR")'
        ", without listing its items or values: the person moves the matter on, what the assis"
        "tant gives is what the person decides on, and its details stay in the raw text, reach"
        "ed by the index.\n"
        "5. Write what the raw text says, add nothing it does not; do not write what happened "
        "later.\n"
        "\n"
        "## Building on\n"
        "6. An SB that builds on an SB of this window goes in that SB's next: a concrete thing"
        " of a larger need under the need; an answer under the thing it answers; a step carrie"
        "d on under the step it carries on. Several next under one SB are the branches from it"
        ". An SB that builds on no SB of this window stays at the top level.\n"
        "\n"
        "## status\n"
        "7. when: where the thing sits on the timeline, seen from the moment it was said (its "
        'place in the conversation, not any clock). "before": it had already happened, or it'
        " already held, by then (something done, a fact, a decision made, a request made, a st"
        'ate that holds). "after": it comes later; add its date when the raw text gives one '
        '("after 2024-03-31"). An arrangement and the thing it arranges sit at different pla'
        "ces: the arrangement (the person's plan, a project schedule, a deadline, an instructi"
        "on given) already exists when it is said, so it is before; the thing it arranges (the"
        " plan carried out, the sprint ending, the deadline met, the instruction followed) is "
        "after; they are separate SBs.\n"
        "8. status follows from when:\n"
        '   - when before: "confirmed" if it happened or holds; "negated" if it did not ha'
        "ppen.\n"
        '   - when after: "likely" if it is arranged: the person set it or firmly intends it'
        ', or a schedule or deadline fixes it; "negated" if it was cancelled, refused or dec'
        'ided against; "unknown" if it was only proposed or recommended and the person did n'
        "ot take it up.\n"
        '   - no evidence either way: "unknown".\n'
        "   Later messages confirm or negate what is likely; that is not judged here.\n"
        "   An SB of the assistant is about the thing it gives, not that the assistant said it"
        ": the assistant saying it does not make it happen or hold. An item of a plan, a recom"
        "mended measure, code for the person to use is after and unknown; a fact, explanation,"
        " judgement or expectation it states is what the person was told, before when it is ab"
        "out the past or present and after when it is about what will be, and unknown until th"
        "e person confirms it, takes it up or reports how it went.\n"
        "\n"
        "## standing\n"
        "9. An instruction or preference the person gives that is meant to hold from now on ("
        '"always ...", "whenever ..., do ...", "I prefer ..."): standing is true; otherw'
        "ise false.\n"
        "\n"
        "## turns\n"
        "10. turns point to the words: messages are shown in numbered paragraphs (M3.1, M3.2 ."
        "..), and the lines of a code paragraph are numbered too. An SB points to the paragrap"
        'hs or lines its words are in ("M3.2", "M4.3:5-18"), "M3" only when the whole me'
        "ssage is this thing; and copies, exactly as written, the first few words (from) and t"
        "he last few words (to) of its words, so the code can find exactly where they are; not"
        " needed when it points to code lines.\n"
        "\n"
        "## Where to stop\n"
        "You need not compile all of NEW in one answer: one answer holds about {cap} SBs at mo"
        "st. Compile it in order, each message as fully as the rules ask. When the next messag"
        "e would take you past that, stop before it, after a message where every SB so far is "
        "complete; the messages after it are compiled in the next answer, with these as contex"
        "t. Do not shorten, merge or leave out SBs to reach the end.\n"
        "\n"
        "## Output\n"
        "First write ANALYSIS: in plain words, go through the messages in order: for each mess"
        "age of the person, the things it says, each named; for each thing, the pieces the ass"
        "istant answers with. Then write a line JSON: followed by only a valid JSON array, no "
        "double quotes inside strings:\n"
        '[{{"turns":"M1","from":"...","to":"...","text":"The person ...","whe'
        'n":"before","status":"confirmed","standing":false}},\n'
        ' {{"turns":"M1","from":"...","to":"...","text":"The person asks ...",'
        '"when":"before","status":"confirmed","standing":false,"next":[\n'
        '   {{"turns":"M1","from":"...","to":"...","text":"The person wants ...'
        '","when":"after","status":"likely","standing":false,"next":[\n'
        '     {{"turns":"M2.3-M2.5","from":"...","to":"...","text":"The assista'
        'nt recommends ...","when":"after","status":"unknown","standing":false}}]}}'
        ",\n"
        '   {{"turns":"M2.4:3-12","text":"The assistant gives ...","when":"before"'
        ',"status":"unknown","standing":false}}]}}]\n'
        "After the JSON write a line STOP: M<k>, the last message you compiled.\n"
        "\n"
        "CONTEXT (earlier messages, for understanding only, not compiled):\n"
        "{context}\n"
        "\n"
        "NEW:\n"
        "{new}"
    ),
}


# The chat scene between people (no assistant in the window, as in LoCoMo): what one tells the
# other and the back-and-forth about it is one thing, and both sides are people, named. The JSON is
# the chat step 1's, so parse_step1_chat reads it (2026-10-31: with STEP1_CHAT every short turn,
# greetings too, became an SB of "the person", names lost).
STEP1_PEOPLE = {
    "zh": (
        "你是 SemanticBlock（SB）编译器。把下面几个人之间的聊天编译成 SB，并写出它们之间谁在谁的"
        "基础上推进。\n"
        "\n"
        "SB 是一件事：一个人讲出来的一件事（一段经历、一个消息、一个打算、一个看法或口味），连同"
        "围绕它的来回：对方的追问、他的补充和细节、对方的回应。这一段来回合起来是一个完整意思，"
        "离开原文也不会被误读，对应原文里连续的几条消息。\n"
        "\n"
        "## 怎么切\n"
        "1. 按话题切，不按消息切：一件事常常要几条、十几条消息才讲完，就是一个 SB。话题换了（换"
        "了一件事，或换成另一个人讲他自己的事），才是新的 SB。同一段经历里分量不同的几面（做了"
        "什么、为什么、结果怎样、他怎么想）可以各成 SB，但不要一句一个。\n"
        "2. 对方只是回应（夸奖、恭喜、附和、表示感兴趣、追问）不单独成 SB，算在这件事里；对方借"
        "着这个话题讲了他自己的事或看法，那是他的 SB。\n"
        "3. 寒暄、道别、没有内容的客套不成 SB。\n"
        "\n"
        "## 怎么写\n"
        "4. text：一两句话写这件事，用名字写明是谁、对谁说的（原文没有名字就写说话者 A／B），按"
        "说话者的本意；相对时间（上周、昨天、去年）按消息时间换成具体日期或月份；保留别人可能问"
        "到的值（人名、地名、物名、数字、日期）。\n"
        "5. 原文说了的写清，没说的不补；不写后来才发生的事。\n"
        "\n"
        "## 推进\n"
        "6. 接着本窗口里某件事说下去的 SB（同一件事后来的进展、对它的回应引出的新事）写进那个 SB "
        "的 next；不接着本窗口任何 SB 的放在最外层。\n"
        "\n"
        "## 状态\n"
        '7. when：从说这件事的那一刻看，它在时间轴上的位置。"before"：那时已经发生或成立。'
        '"after"：在那之后（打算、安排）；原文给了日期就写上。\n'
        '8. status：before 时发生了或成立填 "confirmed"，没发生填 "negated"；after 时定下要做'
        '的填 "likely"，取消或不做的填 "negated"，只是想想的填 "unknown"；没有证据填 "unknown"。\n'
        "9. standing：一个人说自己长期如此的偏好或规矩（我从不……、我一直……、我对……过敏）填 true；"
        "建议、打算、经历都填 false。\n"
        "\n"
        "## turns\n"
        '10. turns 写这件事所在的消息范围（"M3-M9"，只有一条就写 "M3"）；from 照原文抄下第一条'
        "里这件事开头的几个词，to 抄下最后一条里它结尾的几个词。\n"
        "\n"
        "## 在哪里停\n"
        "一次回答最多大约 {cap} 个 SB。按顺序编；下一件事会让你超出这个数时，就停在它前面；后面"
        "的消息在下一次回答里编。\n"
        "\n"
        "## 输出\n"
        "先写 ANALYSIS：按顺序一行一件事，写它的消息范围和是谁讲的什么（不要逐条复述消息）。然后"
        "另起一行写 JSON:，后面只跟合法 JSON 数组，字符串里不要用英文双引号（引用用「」）：\n"
        '[{{"turns":"M3-M9","from":"...","to":"...","text":"甲告诉乙……","when":"before",'
        '"status":"confirmed","standing":false,"next":[\n'
        '   {{"turns":"M40-M44","from":"...","to":"...","text":"甲后来……","when":"before",'
        '"status":"confirmed","standing":false}}]}}]\n'
        "JSON 之后另起一行写 STOP: M<k>，k 是你编到的最后一条消息。\n"
        "\n"
        "CONTEXT（之前的消息，只供理解，不编译）：\n"
        "{context}\n"
        "\n"
        "NEW：\n"
        "{new}"
    ),
    "en": (
        "You are a SemanticBlock (SB) compiler. Compile the chat between people below into SBs a"
        "nd write which builds on which.\n"
        "\n"
        "An SB is one thing: a thing one person tells (an experience, a piece of news, a plan, a"
        "n opinion or a taste) together with the back-and-forth about it: the other's questions,"
        " the teller's further details, the other's response. That stretch together is one comp"
        "lete meaning, which cannot be misread without the raw text, and sits in a run of consec"
        "utive messages.\n"
        "\n"
        "## Cutting\n"
        "1. Cut by topic, not by message: a thing often takes several or a dozen messages to tel"
        "l, and it is one SB. A new SB starts when the topic changes (another thing, or the othe"
        "r person telling their own thing). Facets of one experience that weigh differently (wha"
        "t was done, why, how it came out, what they think of it) may be SBs of their own, but n"
        "ot one per sentence.\n"
        "2. The other's mere response (praise, congratulations, agreement, interest, a follow-up"
        " question) is no SB of its own; it belongs to the thing. When the other uses the topic "
        "to tell their own thing or view, that is their SB.\n"
        "3. Greetings, goodbyes and empty pleasantries are not SBs.\n"
        "\n"
        "## Writing an SB\n"
        "4. text: the thing in one or two sentences, naming who told whom (speaker A / B when th"
        "e words give no names), as the speaker meant it; relative times (last week, yesterday, "
        "last year) made concrete from the message time; keep the values someone may ask for (n"
        "ames of people, places and things, numbers, dates).\n"
        "5. Write what the raw text says, add nothing it does not; do not write what happened la"
        "ter.\n"
        "\n"
        "## Building on\n"
        "6. An SB that carries on a thing of this window (later news of the same thing, a new th"
        "ing its response brings up) goes in that SB's next; one that carries on no SB of this "
        "window stays at the top level.\n"
        "\n"
        "## status\n"
        '7. when: where the thing sits on the timeline, seen from when it was told. "before": i'
        't had happened or held by then. "after": it comes later (a plan, an arrangement); add '
        "its date when the raw text gives one.\n"
        '8. status: when before, "confirmed" if it happened or holds, "negated" if it did not; '
        'when after, "likely" if it is settled, "negated" if called off, "unknown" if only th'
        'ought of; "unknown" with no evidence either way.\n'
        "9. standing: true for a lasting preference or rule a person states about themselves (I nev"
        "er ..., I always ..., I am allergic to ...); advice, plans and stories are false.\n"
        "\n"
        "## turns\n"
        '10. turns: the run of messages the thing is in ("M3-M9", or "M3" for one message); fr'
        "om copies, exactly as written, the first few words of the thing in its first message, "
        "to the last few words in its last message.\n"
        "\n"
        "## Where to stop\n"
        "One answer holds about {cap} SBs at most. Compile in order; when the next thing would t"
        "ake you past that, stop before it; the messages after it are compiled in the next answ"
        "er.\n"
        "\n"
        "## Output\n"
        "First write ANALYSIS: in order, one line per thing, its run of messages and who told wh"
        "at (do not restate the messages one by one). Then write a line JSON: followed by only "
        "a valid JSON array, no double quotes inside strings:\n"
        '[{{"turns":"M3-M9","from":"...","to":"...","text":"Nate tells Joanna ...","when":"bef'
        'ore","status":"confirmed","standing":false,"next":[\n'
        '   {{"turns":"M40-M44","from":"...","to":"...","text":"Nate later ...","when":"before"'
        ',"status":"confirmed","standing":false}}]}}]\n'
        "After the JSON write a line STOP: M<k>, the last message you compiled.\n"
        "\n"
        "CONTEXT (earlier messages, for understanding only, not compiled):\n"
        "{context}\n"
        "\n"
        "NEW:\n"
        "{new}"
    ),
}

# The chat scene's step 1 by the thing talked about (user 2026-10-31): an SB is the thing being
# talked about, over as many messages as the talk stays on it ("I had breakfast - what did you
# have - a sandwich - he prefers noodles" is one SB); what exactly was said stays in the raw text,
# which the raw search finds, so the SB holds what the raw search cannot give: what is being
# talked about and what came of it. The Line's stretches carry how one thing led to the next.
# Who talks and what each message is (own words, pasted material, others' talk) the model reads
# from the words. The JSON is the chat step 1's, read by parse_step1_chat.
STEP1_TOPIC = {
    "zh": (
        "你是 SemanticBlock（SB）编译器。把下面这段对话编译成 SB。"
        "\n"
        "\n"
        "先看清这段对话：是谁在说（人和助手，还是几个人之间；原文里有名字就用名字），每条消息是"
        "什么——某人自己的话，贴进来的材料（文档、规则、代码、日志、数据、别人的聊天记录），还是"
        "没有内容的寒暄。\n"
        "\n"
        "## SB 是什么\n"
        "1. SB 是正在聊的一件事：某人提起一件事，以及围绕它的来回——追问、回答、补充、反应——只"
        "要话题还在这件事上，几条还是上百条消息都是这一个 SB。话题转到另一件事，才开始新的 SB；"
        "一条消息中途转了话题，新 SB 就从那里开始。\n"
        "2. SB 写的是理解，用句子主干写：主语（谁）+ 谓语（做了什么：询问、讲述、解释、建议、决"
        "定、给出……）+ 宾语（哪件事，写出具体名称，让问到这件事的问题能找到它）。具体内容不写："
        "吃了什么、谁喜欢什么、数字、步骤、代码、原文里写着的日期，都留在原文里，靠索引取。\n"
        "3. 人给出的、从此以后都要遵守的指令或长期偏好（以后都……、每次……、我更喜欢……），单独"
        "成一个 SB，standing 填 true。\n"
        "4. 贴进来的材料不是谁的话：在它被贴来所为的那件事的 SB 里提一句贴了什么，不为它的内容立 "
        "SB。没有内容的寒暄不成 SB。\n"
        "\n"
        "## 怎么写\n"
        '5. text：一到两句话，按第 2 条的主干写。主语：人和助手的对话写"人""助手"，不用人设或'
        "系统提示里的名字；几个人之间写他们的名字（没有名字写说话者 A／B）。原文说了的写清，没说"
        "的不补，不写后来才发生的事。\n"
        "6. 时间只解析一层：消息带着它说出口的时间（消息时间）。这件事本身发生的时间如果是相对说"
        '法（三年之前、上周、明天），以消息时间为基准写出它约在何时，如"约 2023 年（三年之前）我'
        '说过十年前去过冰岛"；这件事内容里再套着的时间（十年前）原样留着，不往下推算。\n'
        "7. when 和 status 说这件事到这个 SB 结束时的状态：已经发生或成立填 before、confirmed；"
        "没发生 before、negated；安排好了的 after、likely（有日期写上，如 after 2024-03-31）；只"
        "是提议或推荐、没被接受的 after、unknown；取消或不做的 after、negated。助手说了不等于那件"
        "事发生了。\n"
        '8. turns 写这件事所在的消息范围（"M3-M20"，只有一条写 "M3"）。只有这件事从某条消息中'
        "途开始或在中途结束时，才写 from（照原文抄下它开头的几个词）或 to（结尾的几个词），"
        "否则不写。\n"
        "\n"
        "## 在哪里停\n"
        "一次回答最多大约 {cap} 个 SB。按顺序编；下一件事会让你超出这个数时，就停在它前面；后面"
        "的消息在下一次回答里编。\n"
        "\n"
        "## 输出\n"
        "先写 ANALYSIS：一行说是谁在说、有哪几类消息；然后按顺序一行一件事，写成「M3-M9：在聊什"
        "么」（不加方括号，不要逐条复述消息）。然后另起一行写 JSON:，后面只跟合法 JSON 数组，"
        "字符串里不要用"
        "英文双引号（引用用「」）：\n"
        '[{{"turns":"M1-M9","text":"人询问……，助手给出……","when":"before",'
        '"status":"confirmed","standing":false}},\n'
        ' {{"turns":"M10-M14","text":"……","when":"after","status":"likely",'
        '"standing":false}}]\n'
        "JSON 之后另起一行写 STOP: M<k>，k 是你编到的最后一条消息。\n"
        "\n"
        "CONTEXT（之前的消息，只供理解，不编译）：\n"
        "{context}\n"
        "\n"
        "NEW：\n"
        "{new}"
    ),
    "en": (
        "You are a SemanticBlock (SB) compiler. Compile the conversation below into SBs."
        "\n"
        "\n"
        "First see what the conversation is: who is talking (a person and an assistant, or seve"
        "ral people; use their names when the words give them), and what each message is: some"
        "one's own words, material pasted in (a document, rules, code, a log, data, a transcrip"
        "t of other people's talk), or small talk with nothing in it.\n"
        "\n"
        "## What an SB is\n"
        "1. An SB is one thing being talked about: someone brings up a matter, and the back-and"
        "-forth on it (questions, answers, further details, reactions), for as long as the talk"
        " stays on it, over a few messages or a hundred, is this one SB. A new SB starts only w"
        "hen the talk moves on to another thing; a message that turns to another thing midway s"
        "tarts the new SB there.\n"
        "2. An SB holds understanding, written as the core of a sentence: subject (who) + verb "
        "(what they did: asks, tells, explains, suggests, decides, gives ...) + object (which m"
        "atter, named concretely so that a question about it finds it). The contents are left "
        "out: what was eaten, who likes what, numbers, steps, code, dates written in the words "
        "stay in the raw text, which the index reaches.\n"
        "3. An instruction or lasting preference the person gives that is meant to hold from n"
        'ow on ("always ...", "whenever ..., do ...", "I prefer ...") is an SB of its own, '
        "standing true.\n"
        "4. Pasted material is nobody's words: the SB of the thing it was pasted for mentions w"
        "hat was pasted; its contents get no SBs. Small talk with nothing in it is no SB.\n"
        "\n"
        "## Writing an SB\n"
        "5. text: one or two sentences with the core of rule 2. Subject: in a chat with an ass"
        'istant write "The person" and "the assistant", never a name from a persona or a syste'
        "m prompt; between people use their names (speaker A / B when the words give none). Wr"
        "ite what the raw text says, add nothing it does not, nothing that happened later.\n"
        "6. Time, one level only: each message carries when it was said (its message time). Wh"
        "en the time of the thing itself is said relatively (three years ago, last week, tomor"
        'row), write about when it was from the message time, as "about 2023 (three years ag'
        'o) I said I had been to Iceland ten years before"; a time inside the thing (ten year'
        "s before) stays as said, not worked out further.\n"
        "7. when and status: where the thing stands when this SB ends: it happened or holds, be"
        "fore and confirmed; it did not, before and negated; it is arranged, after and likely ("
        'add its date, "after 2024-03-31"); only proposed or recommended and not taken up, af'
        "ter and unknown; called off, after and negated. The assistant saying it does not make "
        "it happen.\n"
        '8. turns: the run of messages the thing is in ("M3-M20", or "M3" for one). Only when '
        "it starts or ends midway through a message, from copies exactly the first few words of "
        "it, to the last few; otherwise leave them out.\n"
        "\n"
        "## Where to stop\n"
        "One answer holds about {cap} SBs at most. Compile in order; when the next thing would "
        "take you past that, stop before it; the messages after it are compiled in the next an"
        "swer.\n"
        "\n"
        "## Output\n"
        "First write ANALYSIS: one line on who is talking and what kinds of messages there are;"
        ' then in order one line per thing, as "M3-M9: what it is about" (no brackets; do not '
        "restate the messages one by one). Then write a line JSON: followed by only a valid JSON "
        "array, no double quotes inside strings:\n"
        '[{{"turns":"M1-M9","text":"The person asks ..., the assistant gives ...",'
        '"when":"before","status":"confirmed","standing":false}},\n'
        ' {{"turns":"M10-M14","text":"...","when":"after","status":"likely",'
        '"standing":false}}]\n'
        "After the JSON write a line STOP: M<k>, the last message you compiled.\n"
        "\n"
        "CONTEXT (earlier messages, for understanding only, not compiled):\n"
        "{context}\n"
        "\n"
        "NEW:\n"
        "{new}"
    ),
}

# The chat scene's step 1: about how many SBs one answer holds. Past its usual length the model
# compresses (a 92-sentence message taken by one SB, dropped words) instead of stopping, so it is
# told to stop where its SBs close; the rest opens the next window. A property of the compiler
# model (DeepSeek v4 answers run to about 7k tokens), not of the data.
ANSWER_SBS = 60
_STOP = re.compile(r"STOP:\s*M(\d+)")


def stop_at(text: str, n: int) -> int:
    """How many NEW messages the answer compiled (its STOP line; all of them when it has none)."""
    found = _STOP.findall(text)
    return min(max(int(found[-1]), 1), n) if found else n


def compiled_before(sbs: list[dict], done: int) -> list[dict]:
    """The SBs whose words all come before NEW message ``done`` (the answer stopped there); an SB
    built on one left out builds on nothing here."""
    keep = [b for b in sbs if all(i < done for i in b["sources"])]
    index = {id(b): n for n, b in enumerate(keep)}
    return [
        {**b, "under": index.get(id(sbs[b["under"]])) if b.get("under") is not None else None}
        for b in keep
    ]


# a speaker's name opening a message: one to three capitalised words, or a few CJK characters
_SPEAKER = re.compile(r"^\s*([A-Z][\w'.-]*(?: [A-Z][\w'.-]*){0,2}|[一-鿿]{1,6})[:：]\s")


def between_people(messages: list[dict]) -> bool:
    """A chat between people: no assistant, and the words of most messages open with the name of
    one of at least two speakers who each speak more than once ("Nate: ..."). One person's pasted
    notes, or a machine log whose lines open with a key, are not."""
    if not messages or any(m.get("role") == "assistant" for m in messages):
        return False
    names = Counter(n.group(1) for m in messages if (n := _SPEAKER.match(m.get("text") or "")))
    speakers = [name for name, k in names.items() if k >= 2]
    return len(speakers) >= 2 and sum(names[s] for s in speakers) * 2 >= len(messages)


def step1_prompt_slim4(
    lang: str, new: list[dict], context: list[dict], scene: str = "work", grain: str = "meaning"
) -> str:
    """As step1_prompt: NEW messages are M1, M2 ... in order. The chat scene has its own
    templates (STEP1_CHAT; STEP1_PEOPLE for a chat between people; STEP1_TOPIC, by the thing
    talked about, with ``grain="topic"``)."""
    order = list(dict.fromkeys(m.get("session") for m in [*context, *new]))
    tag = (
        (lambda m: f" {SESSION[lang]}{chr(65 + order.index(m.get('session')) % 26)}")
        if len(order) > 1
        else (lambda m: "")
    )
    template = (
        STEP1_TOPIC[lang]
        if scene == "chat" and grain == "topic"
        else (STEP1_PEOPLE if between_people([*context, *new]) else STEP1_CHAT)[lang]
        if scene == "chat"
        else STEP1_SLIM4[lang]
    )
    return template.format(
        cap=ANSWER_SBS,
        context="\n".join(_line(lang, "", m, 600, tag(m)).replace("[ ", "[", 1) for m in context)
        or "-",
        new="\n".join(_numbered(lang, f"M{i + 1}", m, tag(m)) for i, m in enumerate(new)),
    )


_RANGE = re.compile(r"M(\d+)(?:\s*[-–~～至]\s*M?(\d+))?")
_LINES = re.compile(r"M(\d+)\.(\d+):(\d+)(?:\s*[-–~～至]\s*(\d+))?")
_TURN = re.compile(r"M(\d+)(?:\.(\d+))?(?:\s*[-–~～至]\s*(M?)(\d+)(?:\.(\d+))?)?")


def span_key(item) -> tuple:
    """Order of span items: a paragraph number k, or (k, first line, last line) of a code
    paragraph."""
    return (item, 0) if isinstance(item, int) else (item[0], item[1])


def _turns(value: str, n: int, paras: list[int]) -> tuple[set[int], dict[int, set]]:
    """Messages (0-based) a turns value covers, and for those it covers only in part, the
    paragraphs (1-based) or (paragraph, first line, last line) of a code paragraph: "M3", "M3-M7",
    "M3.2", "M3.2-M3.5", "M3.2-5", "M3.2-M5.1", "M3.4:5-18"."""
    whole: set[int] = set()
    part: dict[int, set] = {}
    for a, k, lo, hi in _LINES.findall(value):
        i, k, lo = int(a), int(k), int(lo)
        if 1 <= i <= n and 1 <= k <= paras[i - 1]:
            part.setdefault(i - 1, set()).add((k, lo, max(lo, int(hi or lo))))
    value = _LINES.sub(" ", value)
    for a, ap, bm, b, bp in _TURN.findall(value):
        if b and ap and not bm and not bp:  # M3.2-5: paragraphs of one message
            lo, hi, lo_p, hi_p = int(a), int(a), int(ap), int(b)
        else:
            lo, hi = int(a), int(b or a)
            lo_p, hi_p = (
                (int(ap) if ap else 0),
                (int(bp) if bp else (int(ap) if ap and not b else 0)),
            )
            if lo > hi:
                lo, hi, lo_p, hi_p = hi, lo, 0, 0
        for i in range(max(lo, 1), min(hi, n) + 1):
            count = paras[i - 1]
            start = lo_p if i == lo and lo_p else 1
            end = min(hi_p if i == hi and hi_p else count, count)
            if start <= 1 and end >= count:
                whole.add(i - 1)
            elif start <= end:
                part.setdefault(i - 1, set()).update(range(start, end + 1))
    return whole | set(part), {i: p for i, p in part.items() if i not in whole}


def parse_step1_slim4(data: object, roles: list[str], paras: list[int] | None = None) -> list[dict]:
    """SBs as parse_step1 gives them: {"sources", "situation" (the summary, or the chat scene's
    text), "move" (""), "happened", "unresolved"}. The work scene does not tell plans apart
    (happened is True); in the chat scene only status "happened" is happened, "not yet" and
    "unknown" stay in progress. Sources are every message in the turns span; spans out of range
    are clipped, an SB with no valid span is dropped. ``paras``: the paragraph count of each
    message (paragraphs()); "spans" gives, for the messages an SB covers only in part, the
    paragraphs it points to."""
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of SBs")
    n = len(roles)
    paras = paras or [1] * n
    out = []
    for b in data:
        if not isinstance(b, dict):
            continue
        text = str(b.get("summary") or b.get("text") or "").strip()
        if not text:
            continue
        sources, spans = _turns(str(b.get("turns") or ""), n, paras)
        if not sources:
            continue
        u = _RANGE.fullmatch(str(b.get("unresolved") or "").strip())
        out.append(
            {
                "sources": sorted(sources),
                "spans": {i: sorted(p, key=span_key) for i, p in spans.items()},
                "situation": text,
                "move": "",
                "happened": "status" not in b or str(b["status"]).strip().lower() == "happened",
                "standing": b.get("standing") is True or str(b.get("standing")).lower() == "true",
                "unresolved": int(u.group(1)) - 1 if u and 0 < int(u.group(1)) <= n else None,
            }
        )
    out.sort(key=lambda b: (b["sources"][-1], b["sources"][0]))
    return out


STATUS = ("confirmed", "likely", "negated", "unknown")


def extract_after_json(text: str) -> list:
    """The array after the last "JSON:" line, else after the last ```json fence (the ANALYSIS
    before it may hold brackets)."""
    for mark in ("JSON:", "```json"):
        if mark in text:
            return extract_json_array(text[text.rfind(mark) :])
    return extract_json_array(text)


def _stamp(x: dict) -> str:
    status = str(x.get("status") or "unknown").strip().lower()
    when = str(x.get("when") or "").strip()
    return f"[{status if status in STATUS else 'unknown'}{'; ' + when if when else ''}]"


def _fragments(texts, idxs, items: dict, first, last) -> tuple[dict, int]:
    """{message: [[start, end], ...] or None (whole)} for the messages idxs: the copied words when
    they are found, else the numbered paragraphs or lines; and 1 when copied words were not
    found."""
    found, out = False, {}
    for i in sorted(idxs):
        if first and (span := anchor(texts[i], first, last)) is not None:
            out[i], found = [span], True
        elif i in items:
            out[i] = item_spans(texts[i], items[i])
        else:
            out[i] = None
    if found:  # the words were found: other messages only where numbers point into them
        out = {i: p for i, p in out.items() if p is not None}
    return out, int(bool(first) and not found)


def parse_step1_chat(
    data: object, roles: list[str], paras: list[int] | None = None, texts: list[str] | None = None
) -> list[dict]:
    """SBs of the chat step 1 (STEP1_CHAT) as parse_step1_slim4 gives them, nested ones (next)
    unfolded: "situation" is the text and its [status; when]; "under" the SB of this window it
    builds on (after sorting), None for none. Sources: its turns; "pointers": the fragment of the
    words of each message it comes from (character spans when texts are given), None for whole."""
    if isinstance(data, dict):
        data = next((v for v in data.values() if isinstance(v, list)), None)
    if not isinstance(data, list):
        raise ValueError("expected a JSON array of SBs")
    n = len(roles)
    paras = paras or [1] * n
    flat: list[tuple[dict, int | None]] = []  # (SB, position of the SB it builds on)

    def unfold(items, parent: int | None) -> None:
        for x in items if isinstance(items, list) else []:
            if isinstance(x, dict):
                flat.append((x, parent))
                unfold(x.get("next"), len(flat) - 1)

    unfold(data, None)
    kept: dict[int, dict] = {}
    out = []
    for at, (b, _) in enumerate(flat):
        text = str(b.get("text") or b.get("summary") or "").strip()
        sources, own = _turns(str(b.get("turns") or ""), n, paras)
        if not text or not sources:
            continue
        missed = 0
        if texts:
            mine, missed = _fragments(texts, sources, own, b.get("from"), b.get("to"))
        else:
            mine = {i: sorted(own[i], key=span_key) if i in own else None for i in sources}
        status = str(b.get("status") or "").strip().lower()
        if status == "confirmed" and all(roles[i] == "assistant" for i in sources):
            # what the person is told does not hold by being said (design 1.2): only the person's
            # later words settle it
            b, status = {**b, "status": "unknown"}, "unknown"
        out.append(
            {
                "sources": sorted(sources),
                "spans": {i: sorted(p, key=span_key) for i, p in own.items()},
                "situation": f"{text} {_stamp(b)}",
                "move": "",
                "pointers": {
                    "own": mine,
                    "parts": [],
                    **({"chars": True, "missed": missed} if texts else {}),
                },
                "happened": status == "confirmed",
                "standing": b.get("standing") is True or str(b.get("standing")).lower() == "true",
                "unresolved": None,
            }
        )
        kept[at] = out[-1]
    out.sort(key=lambda b: (b["sources"][-1], b["sources"][0]))
    index = {id(b): i for i, b in enumerate(out)}
    for at, sb in kept.items():
        parent = flat[at][1]
        while parent is not None and parent not in kept:  # an SB dropped: the one it built on
            parent = flat[parent][1]
        sb["under"] = None if parent is None else index[id(kept[parent])]
    if texts:  # an SB's own words: its fragment without the words of what branches from it there
        for sb in out:
            for kid in (k for k in out if k.get("under") is not None and out[k["under"]] is sb):
                for i, spans in kid["pointers"]["own"].items():
                    mine = sb["pointers"]["own"]
                    if i not in mine or spans is None:
                        continue
                    own = [[0, len(texts[i])]] if mine[i] is None else mine[i]
                    left = _without(own, spans, texts[i])
                    if left:
                        mine[i] = left
        _own_code(out, roles, texts)
    return out


def _own_code(sbs: list[dict], roles: list[str], texts: list[str]) -> None:
    """Code the person pastes stays in the raw text (rule 4), so the raw index must reach it: a
    code paragraph of the person's message that no SB points to goes with the SB of that message
    whose words come right before it (the one that brings it in), else the first after it."""
    for i, (role, text) in enumerate(zip(roles, texts, strict=False)):
        mine = [b for b in sbs if b["pointers"]["own"].get(i)]  # SBs with fragments in this message
        if role != "user" or not mine:
            continue
        for (s, e), p in zip(piece_spans(text), paragraphs(text), strict=False):
            if not is_code(p) or any(
                a < e and s < z for b in mine for a, z in b["pointers"]["own"][i]
            ):
                continue
            ends = {
                id(b): max((z for _, z in b["pointers"]["own"][i] if z <= s), default=-1)
                for b in mine
            }
            to = (
                max(mine, key=lambda b: ends[id(b)])
                if max(ends.values()) >= 0
                else min(mine, key=lambda b: b["pointers"]["own"][i][0][0])
            )
            to["pointers"]["own"][i] = sorted([*to["pointers"]["own"][i], [s, e]])


def _without(spans: list, cut: list, text: str) -> list[list[int]]:
    """The character spans with the spans ``cut`` taken out, pieces holding only blanks or
    punctuation dropped."""
    out = []
    for s, e in spans:
        pieces = [[s, e]]
        for c, d in cut:
            pieces = [
                p
                for a, b in pieces
                for p in ([[a, min(b, c)]] if a < c else []) + ([[max(a, d), b]] if b > d else [])
            ]
        out += [p for p in pieces if any(ch.isalnum() for ch in text[p[0] : p[1]])]
    return out


def uncovered(
    sbs: list[dict], roles: list[str], texts: list[str], words: int = 8
) -> list[tuple[int, str]]:
    """The stretches of the person's words that no SB fragment covers, of at least ``words``
    words: (message, the words)."""
    out = []
    for i, (role, text) in enumerate(zip(roles, texts, strict=False)):
        if role != "user":
            continue
        cov = [False] * len(text)
        for b in sbs:
            if i not in b["pointers"]["own"]:
                continue
            p = b["pointers"]["own"][i]
            for s, e in [[0, len(text)]] if p is None else p:
                cov[s:e] = [True] * (max(0, min(e, len(text)) - s))
        start = None
        for k, done in enumerate([*cov, True]):
            if not done and start is None:
                start = k
            elif done and start is not None:
                stretch = text[start:k].strip()
                if len(stretch.split()) >= words:
                    out.append((i, stretch))
                start = None
    return out


def check_step1_chat(
    data: object,
    roles: list[str],
    paras: list[int] | None = None,
    texts: list[str] | None = None,
    done: int | None = None,
) -> None:
    """A broken answer (seen: one item and done) leaves most of the person's messages without an SB
    and never reaches the last message before its stop: ask again. Only the messages before the
    answer's stop (``done``) count: the rest open the next window. Material the person pasted (a
    rulebook or a thread, around their question) may have no SB while the question and the answer
    have one; an empty answer means nothing here to compile (a machine log): the raw text stays,
    nothing is asked again (2026-10-31, CL-Bench)."""
    if data == []:
        return
    sbs = parse_step1_chat(data, roles, paras, texts)
    people = [i for i, r in enumerate(roles[:done]) if r == "user"]
    covered = {i for b in sbs for i in b["sources"]}
    if (
        len(people) >= 2
        and sum(i in covered for i in people) * 2 < len(people)
        and len(roles[:done]) - 1 not in covered
    ):
        raise ValueError(
            f"SBs cover {sum(i in covered for i in people)} of the person's {len(people)} messages"
        )


# one "key": "value" per line, the value copied from the raw text (code holds quotes the model
# leaves bare)
_FIELD = re.compile(r'^(\s*"[A-Za-z_]+":\s*")(.*)("\s*,?\s*)$')


def _escape_bare_quotes(text: str) -> str:
    def fix(line: str) -> str:
        m = _FIELD.match(line)
        return m[1] + re.sub(r'(?<!\\)"', r'\\"', m[2]) + m[3] if m else line

    return "\n".join(map(fix, text.split("\n")))


def extract_json_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("no JSON array in the output")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return json.loads(_escape_bare_quotes(text[start : end + 1]))


def check_step1_slim4(data: object, roles: list[str]) -> None:
    """Most of the person's messages left outside every SB is a broken answer: ask again."""
    sbs = parse_step1_slim4(data, roles)
    people = [i for i, r in enumerate(roles) if r == "user"]
    covered = {i for b in sbs for i in b["sources"]}
    if len(people) >= 2 and sum(i in covered for i in people) * 2 < len(people):
        raise ValueError(
            f"SBs cover {sum(i in covered for i in people)} of the person's {len(people)} messages"
        )


# --- step 2 ---------------------------------------------------------------------

STEP2 = {
    "zh": (
        "你在给长期记忆系统的语义块（SB）连承接。下面是人和编程 Agent 工作对话里一个窗口的 SB"
        "（s 编号，每个是“发生的情况 → 人的动作”，后面列出它的候选），以及候选详情：更早的步骤"
        "（N 编号，标着当时是已发生还是进行中）。\n"
        "\n"
        "对每个 SB 判断它是在哪些步骤的基础上推进的（after）：\n"
        "- 这一步里 Agent 在执行的，是人在哪一步做的决定或要求，就接那一步；\n"
        "- 人在这一步里把自己早先的决定或判断拿回来用、修改或推翻，就接那一步，不管隔多远（比"
        "如“开头要求的格式还是不对”接回开头定格式的那一步；“不是说好用测试库吗”接回说好的那一"
        "步）；\n"
        "- 人对 Agent 一次汇报的反馈，接这次汇报所执行的那一步。\n"
        "可以接它列出的 N 候选，也可以接本窗口里排在它前面的 s。可以有几个，也可以没有。用到同"
        "一个概念、话题相同、用词相近、时间挨着，都不是承接；Agent 的叙述里提到某一步也不算，"
        "看人这一步自己做了什么。\n"
        "如果这个 SB 让某个还在进行中的步骤有了结果（做完、定下、放弃），写进 settles。\n"
        "\n"
        '只输出紧凑 JSON，字符串里不要用英文双引号：{{"links":[{{"sb":"s3","to":"N12'
        '","why":"一句话"}}],"settles":[{{"sb":"s3","step":"N7"}}]}}\n'
        "\n"
        "本窗口的 SB：\n"
        "{sbs}\n"
        "\n"
        "候选详情：\n"
        "{cands}"
    ),
    "en": (
        "You are linking semantic blocks (SBs) of a long-term memory system. Below are the SBs"
        " of one window of a work conversation between a person and a coding agent (s ids, eac"
        'h "what happened -> what the person did", followed by its candidates), and the cand'
        "idate details: earlier steps (N ids, marked as happened or in progress at the time)."
        "\n"
        "\n"
        "For each SB, decide which steps it builds on (links):\n"
        "- the step where the person made the decision or request that the agent is carrying o"
        "ut in this step;\n"
        "- a step whose decision or judgement the person brings back here, changes or reverses"
        ', however far back (e.g. "the format I asked for at the start is still wrong" goes '
        "back to the step where the format was set; \"didn't we agree to use the test database"
        '?" goes back to that agreement);\n'
        "- for the person's feedback on a report, the step that report carries out.\n"
        "It may link to its listed N candidates or to an s earlier in this window; several, or"
        " none. Sharing a concept, a topic or words, or coming right next in time, is not buil"
        "ding on; neither is the agent's narrative mentioning a step: look at what the person "
        "did in this step.\n"
        "When an SB shows that a step still in progress has come to a result (done, decided, d"
        "ropped), put it in settles.\n"
        "\n"
        'Return only compact JSON, no double quotes inside strings: {{"links":[{{"sb":"s3'
        '","to":"N12","why":"one sentence"}}],"settles":[{{"sb":"s3","step":'
        '"N7"}}]}}\n'
        "\n"
        "SBs of this window:\n"
        "{sbs}\n"
        "\n"
        "Candidate details:\n"
        "{cands}"
    ),
}
# The chat scene (the interaction is the content; slim4 SBs): building on is the same matter going
# on.
# The chat scene's step 2 shows the earlier steps under the person's matter they follow (design 4.1:
# what the person says defines a matter, the assistant's answers are steps after it): asked for
# steps
# one by one, the model linked every one that looked related and joined unrelated matters.
STEP2_CHAT = {
    "zh": (
        "你在给长期记忆系统的语义块（SB）连承接。下面是一段聊天里一个窗口的 SB（s 编号，每个是"
        '一件事，后面列出给它的早先的事），以及早先的事：每件是人提出的一件事（标"人"的 N）'
        "，它后面跟着的步骤缩进列在下面（N 编号，人的或助手的）；每步标着当时是已发生还是进行"
        "中。\n"
        "\n"
        "对每个 SB 判断它接着推进的是哪件早先的事：这件事接着发展（计划做了、变了、取消了，状"
        "况有了新消息，做的时候碰到了问题）、对其中说过的作了更新、修正或推翻、回到这件事上，"
        "不管隔多远。连到它接着往下走的那一步（这件事本身，或它下面的某一步）。如果它接着的是"
        "本窗口更早的 SB（它是在那里开头的那件事的一部分），就连那个 s。话题相同、用词相近、时"
        "间挨着都不是承接。一个 SB 接着推进几件事，就是把它们合到了一起，这时在 why 里写明合的"
        "是什么。不接着推进任何早先的事就不连。\n"
        "人的 SB 报告了某个还在进行中的步骤有了结果（做了、定了、放弃了；助手给的东西，人确认"
        "了、照着做了或用上了），写进 settles。问起它、再要一次、或者只是助手这么说，都不是结"
        '果。标着"只看 settles"的 SB 在本窗口里已经有位置，只给它们写 settles，不连承接。\n'
        "\n"
        '只输出紧凑 JSON，字符串里不要用英文双引号：{{"links":[{{"sb":"s3","to":"N12'
        '","why":"一句话"}}],"settles":[{{"sb":"s3","step":"N7"}}]}}\n'
        "\n"
        "本窗口的 SB：\n"
        "{sbs}\n"
        "\n"
        "早先的事：\n"
        "{cands}"
    ),
    "en": (
        "You are linking semantic blocks (SBs) of a long-term memory system. Below are the SBs"
        " of one window of a conversation (s ids, each one thing, followed by the earlier matt"
        "ers listed for it), and the earlier matters: each is a thing the person brought up (a"
        "n N marked the person), with the steps that followed it indented under it (N ids, by "
        "the person or the assistant); each step is marked happened or in progress at the time"
        ".\n"
        "\n"
        "For each SB, decide which earlier matter it carries on: the same matter going on (a p"
        "lan carried out, changed or cancelled; news about a situation; a problem met while do"
        "ing it), an update, correction or reversal of something said in it, or a return to it"
        ", however far back. Link it to the step of that matter it goes on from (the matter it"
        "self or a step under it). When it goes on from an earlier SB of this window instead ("
        "it is part of a thing begun there), link it to that s. Sharing a topic or words, or c"
        "oming right next in time, is not carrying on. An SB that carries on several matters b"
        "rings them together: then say in why what it brings together. None if it carries on n"
        "o earlier matter.\n"
        "When an SB of the person reports that a step still in progress came to a result (it w"
        "as done, decided or dropped; for what the assistant gave: the person confirmed it, fo"
        "llowed it or put it to use), put it in settles. Asking about it, asking for it again,"
        " or the assistant saying so is not a result. SBs marked (settles only) already have t"
        "heir place in this window: give only settles for them, no links.\n"
        "\n"
        'Return only compact JSON, no double quotes inside strings: {{"links":[{{"sb":"s3'
        '","to":"N12","why":"one sentence"}}],"settles":[{{"sb":"s3","step":'
        '"N7"}}]}}\n'
        "\n"
        "SBs of this window:\n"
        "{sbs}\n"
        "\n"
        "Earlier matters:\n"
        "{cands}"
    ),
}
# Threads, their own call after step 2 (2026-10-30): asked inside the linking call they made its
# links worse (holdout, history 2: about 63 -> 54 on the same readout).
THREADS = {
    "zh": (
        "下面是一段对话里一个窗口的 SB（s 编号），以及早先的段（T 编号，带当前的总结）。\n"
        "\n"
        '段：一条线上的一段，是人要达成的一个目标，它能单独说"做成了、没做成、放弃了"（给天'
        "气 App 加城市自动补全、控制 API 调用次数、部署到 GitHub Pages）。为这个目标问怎么做、"
        '真去做、碰到的子问题和改进，都是这一段里的条目。整个项目（"做天气 App"）不是一段，'
        "是这些段所在的线；人转去追另一个这样的目标，就开新的一段。\n"
        "\n"
        "早先的事（N 编号）是这些 SB 可能接着的具体步骤，每一步标着它在哪一段；SB 后面列着它的"
        "候选。判断 SB 属于哪一段，要对照这些具体步骤，不只看段的总结。给每个 SB 写它属于哪一"
        '段：早先的某个 T；本窗口里开了一段的某个 s；或者它自己开一段，写 "new"（只写在新段'
        "的第一个 SB 上，同一段后面的 SB 写那个 s）。开新段时，如果这个新目标是从某一段的结果"
        "引出来的，在 from 里写那一段（T 或 s），在 why 里写一句怎么引出来的；不是这样引出的就"
        "不写 from。\n"
        "\n"
        '只输出紧凑 JSON，字符串里不要用英文双引号：{{"threads":[{{"sb":"s1","thread":'
        '"T2"}},{{"sb":"s2","thread":"new","from":"T2","why":"一句话"}},{{"'
        'sb":"s3","thread":"s2"}}]}}\n'
        "\n"
        "本窗口的 SB：\n"
        "{sbs}\n"
        "\n"
        "早先的段：\n"
        "{threads}\n"
        "\n"
        "早先的事：\n"
        "{cands}"
    ),
    "en": (
        "Below are the SBs of one window of a conversation (s ids), and the earlier threads (T"
        " ids, with their summary as it stands).\n"
        "\n"
        "Threads: a thread is one stretch of a Line: one goal the person pursues that can on i"
        "ts own be done, not done or dropped (adding city autocomplete to the weather app, kee"
        "ping the API calls under the limits, deploying to GitHub Pages). Asking how, doing it"
        ", the sub-problems met and the improvements for that goal are entries of one thread. "
        'A whole project ("building the weather app") is not a thread but the Line its threa'
        "ds are on; when the person turns to another such goal, a new thread begins.\n"
        "\n"
        "The earlier matters (N ids) are the concrete steps these SBs may go on from, each mar"
        "ked with its thread; each SB lists its candidates. Judge which thread an SB is in aga"
        "inst these concrete steps, not against the summaries alone. For each SB say which thr"
        'ead it belongs to: an earlier T; an s of this window that began one; or "new" when '
        "it begins one itself (only on the first SB of a new thread; its later SBs give that s"
        "). For a new thread, when the new goal came out of a thread's result, give that threa"
        "d (T or s) in from and in why one sentence on how it led there; no from when it did n"
        "ot.\n"
        "\n"
        'Return only compact JSON, no double quotes inside strings: {{"threads":[{{"sb":"'
        's1","thread":"T2"}},{{"sb":"s2","thread":"new","from":"T2","why":'
        '"one sentence"}},{{"sb":"s3","thread":"s2"}}]}}\n'
        "\n"
        "SBs of this window:\n"
        "{sbs}\n"
        "\n"
        "Earlier threads:\n"
        "{threads}\n"
        "\n"
        "Earlier matters:\n"
        "{cands}"
    ),
}


def threads_prompt(lang: str, sbs: list[dict], threads: list[dict], cands: list[dict] = ()) -> str:
    """sbs: {"text", "time", "cands": [N indexes]}; cands: the concrete earlier steps step 2 was
    shown, {"text", "thread": T index or None}, so a thread is judged against steps, not
    summaries."""
    word = {"zh": "候选", "en": "candidates"}[lang]
    lines = [
        f"s{i + 1} [{x.get('time', '')}] {x['text']}"
        + (f"\n   {word}: " + " ".join(f"N{j + 1}" for j in x["cands"]) if x.get("cands") else "")
        for i, x in enumerate(sbs)
    ]
    shown = [f"T{t + 1} {x['summary']}" for t, x in enumerate(threads)]

    def tag(c: dict) -> str:  # the thread a step is in, when shown
        return f" T{c['thread'] + 1}" if c.get("thread") is not None else ""

    steps = [f"N{j + 1}{tag(c)} {c['text'].split(chr(10))[0]}" for j, c in enumerate(cands)]
    return THREADS[lang].format(
        sbs="\n".join(lines), threads="\n".join(shown) or "-", cands="\n".join(steps) or "-"
    )


def parse_threads(data: dict, n_sbs: int, n_threads: int) -> dict:
    """{"thread": {i: ("T", t) | ("s", k)}, "from": {i: (("T", t) | ("s", k), why)}} for the
    window's SBs."""
    if not isinstance(data, dict) or not isinstance(data.get("threads", []), list):
        raise ValueError("'threads' must be a list")

    def num(x: object, prefix: str, limit: int) -> int | None:
        v = str(x or "").strip()
        return (
            int(v[1:]) - 1
            if v[:1] == prefix and v[1:].isdigit() and 0 < int(v[1:]) <= limit
            else None
        )

    thread: dict[int, tuple] = {}
    came: dict[int, tuple] = {}
    for x in data.get("threads") or []:
        if not isinstance(x, dict) or (i := num(x.get("sb"), "s", n_sbs)) is None:
            continue
        to = str(x.get("thread", "")).strip()
        if (t := num(to, "T", n_threads)) is not None:
            thread[i] = ("T", t)
        elif (k := num(to, "s", n_sbs)) is not None and k <= i:
            thread[i] = ("s", k)
        elif to.lower() == "new":
            thread[i] = ("s", i)
        if thread.get(i) == ("s", i):
            frm = str(x.get("from", "")).strip()
            ref = (
                ("T", t)
                if (t := num(frm, "T", n_threads)) is not None
                else ("s", k)
                if (k := num(frm, "s", n_sbs)) is not None and k < i
                else None
            )
            if ref is not None:
                came[i] = (ref, str(x.get("why", "")).strip()[:300])
    return {"thread": thread, "from": came}


# A thread's summary, its own step after step 2 (2026-10-30): with summaries written inside the
# linking call, window by window, they grew into logs; asked on its own, plainly, a model writes a
# summary of the matter.
SUMMARIZE = {
    "zh": (
        "下面是对话里的几件事。每件给出它之前的总结（如果有），以及之后新的经过（SB）。重写每"
        "件事的总结：用几句话说这件事是为了什么、谁做了什么、最后怎样（交付了什么，或者留了什"
        '么要人决定）。助手给的东西只说它是什么、为了什么（"助手给了一个带队列和重试的调用计'
        '数器"），不逐项列出里面的内容；具体内容在原文里，靠索引去取。\n'
        "\n"
        '只输出紧凑 JSON，字符串里不要用英文双引号：{{"summaries":[{{"thread":"T1","sum'
        'mary":"几句话"}}]}}\n'
        "\n"
        "{threads}"
    ),
    "en": (
        "Below are some matters of a conversation. Each gives its summary so far (if any) and "
        "what happened since (SBs). Rewrite each matter's summary: in a few sentences, what it"
        " was for, who did what, and how it ended up (what was delivered, or what is left for "
        "the person to decide). Say what the assistant gave by what it is and what it is for ("
        '"the assistant gave a call tracker with a queue and retries"), never listing what i'
        "s in it; that is in the raw text, reached by the index.\n"
        "\n"
        'Return only compact JSON, no double quotes inside strings: {{"summaries":[{{"threa'
        'd":"T1","summary":"a few sentences"}}]}}\n'
        "\n"
        "{threads}"
    ),
}


def summarize_prompt(lang: str, threads: list[dict]) -> str:
    """threads: {"before": its summary so far or "", "new": the texts of its SBs since}; T1.. in
    order."""
    so_far = {"zh": "之前的总结", "en": "Summary so far"}[lang]
    since = {"zh": "之后的经过", "en": "Since then"}[lang]
    blocks = [
        f"T{t + 1}\n{so_far}: {x['before'] or '-'}\n{since}:\n"
        + "\n".join(f"- {n}" for n in x["new"])
        for t, x in enumerate(threads)
    ]
    return SUMMARIZE[lang].format(threads="\n\n".join(blocks))


def parse_summaries(data: dict, n_threads: int) -> dict[int, str]:
    if not isinstance(data, dict) or not isinstance(data.get("summaries", []), list):
        raise ValueError("'summaries' must be a list")
    out = {}
    for x in data["summaries"]:
        t = str(x.get("thread", "")).strip() if isinstance(x, dict) else ""
        if (
            t[:1] == "T"
            and t[1:].isdigit()
            and 0 < int(t[1:]) <= n_threads
            and str(x.get("summary", "")).strip()
        ):
            out[int(t[1:]) - 1] = str(x["summary"]).strip()[:4000]
    return out


# --- the stretch layer (docs/LCE_FROZEN.md, 2026-10-31) ---------------------------------------
# After a window's SBs, one call: the SBs in order cut into stretches (a stretch is the part of a
# Line that has grown and settled: the semantic summary of a stretch of Path A), each closed one
# summarized once - cause, who, when, what, consequence - which helps retrieval and the local
# projection; where the talk took up an earlier stretch again; and which steps still in progress
# these SBs settle (the terminator is the memory side's own: never shown to the answer model).
STRETCH = {
    "zh": (
        "下面是一段对话按顺序编出的 SB，每个带着它所在消息的时间范围：开头的 o1、o2……是上次还没"
        "收口的那一段，后面的 s1、s2……是这次新编的。把它们按顺序划成段，给已经收口的段写摘要。\n"
        "\n"
        "## 段是什么\n"
        "1. 段是连续的几个 SB：话题在一条路上往前走，一件事引出下一件（聊早饭 → 饮食习惯 → 偏"
        "好 → 想吃蛋挞）。话题转到另一条路上（聊的领域变了，或做的事变了：从排进度到写代码），"
        "或者开了新的会话、隔了很久，段就到此为止，下一段开始。\n"
        "2. 段按顺序覆盖所有 SB，不跳过、不重叠。\n"
        "3. o1、o2……已经是同一段的开头，不拆开，第一段从 o1 开始；照第 1 条看这一段到哪里为止，"
        "话题接着往下走就延伸到后面的 s。没有 o 时，第一段从 s1 开始。\n"
        "4. closed：话题已经离开这一段（后面接着别的段，或会话结束了），填 true。最后一段话题还没"
        "走完的填 false，等它收口时再写摘要。\n"
        "5. summary 只给 closed 的段写（含 o 的段，摘要连 o 一起写），一到三句"
        "话，按顺序写：前因（因为什么聊起来的；对话是突然开始的就写无）、谁、何时（这一段消息的"
        "时间范围）、做了什么、后果（结果怎样，话题因此转到了哪里）。这是比 SB 高一层的结构性总"
        "结，讲这一段怎么变过来的：不写具体内容和单纯的数据（吃了什么、起点、目标、参数、测量值、"
        "版本、步骤、代码、清单）。只有一种数值要写：它就是结果本身（分数提升了 30%、覆盖率提到 "
        "78%），写最后的结果，不写起点和目标。时间照 SB 里的写法，不往下推算。\n"
        "6. from：某一段重新聊起 EARLIER 里以前某一段的事，写出那一段的编号（T2），前因里写明接"
        "着它。\n"
        "7. settles：本次某个 SB 是人报告 IN PROGRESS 里某件还在进行的事有了结果（做完了、定了、"
        '取消了），写 {{"s5": ["N2"]}}。只有人的报告算，助手这么说不算。\n'
        "\n"
        "只输出 JSON：\n"
        '{{"stretches":[{{"sbs":"s1-s4","from":null,"closed":true,'
        '"summary":"前因……，谁……，何时……，做了……，后果……"}},'
        '{{"sbs":"s5-s7","from":"T2","closed":false,"summary":""}}],'
        '"settles":{{"s6":["N1"]}}}}\n'
        "\n"
        "EARLIER（以前的段）：\n"
        "{earlier}\n"
        "\n"
        "IN PROGRESS（还在进行的事）：\n"
        "{progress}\n"
        "\n"
        "SBS：\n"
        "{sbs}"
    ),
    "en": (
        "Below are the SBs compiled from a conversation, in order, each with the time range of "
        "its messages: the first ones, o1, o2 ..., are the stretch left open last time; s1, "
        "s2 ... are the new ones. Cut them, in order, into stretches, and summarize the closed "
        "ones.\n"
        "\n"
        "## What a stretch is\n"
        "1. A stretch is a run of consecutive SBs where the talk goes on along one course, one "
        "thing leading to the next (breakfast -> eating habits -> preferences -> wanting egg "
        "tarts). Where the talk turns to another course (another field, or another thing being "
        "done: from planning to writing code), or a new session starts, or after a long pause, "
        "the stretch ends and the next one begins.\n"
        "2. Stretches cover every SB, in order, with no gap and no overlap.\n"
        "3. o1, o2 ... are already the start of one stretch, not split: the first stretch begins "
        "at o1; by rule 1, see where that stretch ends: when the talk goes on along its course "
        "it runs on into the s's. With no o's, the first stretch begins at s1.\n"
        "4. closed: true when the talk has left the stretch (another stretch follows, or the "
        "session ended); false for a last stretch whose course is not run yet: it is summarized "
        "when it closes.\n"
        "5. summary, only for a closed stretch (one with o's covers the o's too): one "
        "to three sentences, in this order: cause (what started it; none when the talk just "
        "began), who, when (the time range of its messages), what they did, consequence (what "
        "came of it, and where the talk went because of it). A structural summary one level "
        "above the SBs, of how this stretch changed things: no contents and no mere data (what "
        "was eaten, starting points, targets, parameters, measurements, versions, steps, code, "
        "lists). One kind of figure is written: one that is the result itself (the score rose "
        "30%, coverage reached 78%), the final result, not the start or the target. Times as "
        "the SBs give them, not worked out further.\n"
        "6. from: when a stretch takes up again what an earlier stretch in EARLIER was about, "
        "name it (T2); its cause says it carries that on.\n"
        "7. settles: an SB here in which the person reports how something IN PROGRESS came out "
        '(done, decided, called off): {{"s5": ["N2"]}}. Only the person\'s report counts, not '
        "the assistant saying so.\n"
        "\n"
        "Output only JSON:\n"
        '{{"stretches":[{{"sbs":"s1-s4","from":null,"closed":true,'
        '"summary":"Cause ..., who ..., when ..., what ..., consequence ..."}},'
        '{{"sbs":"s5-s7","from":"T2","closed":false,"summary":""}}],'
        '"settles":{{"s6":["N1"]}}}}\n'
        "\n"
        "EARLIER (earlier stretches):\n"
        "{earlier}\n"
        "\n"
        "IN PROGRESS (things still going on):\n"
        "{progress}\n"
        "\n"
        "SBS:\n"
        "{sbs}"
    ),
}


def stretch_prompt(
    lang: str, sbs: list[dict], open_sbs: list[dict], earlier: list[str], progress: list[str]
) -> str:
    """sbs: {"text", "time"} of the window (s1 ...); open_sbs: the same of the stretch left open
    before (o1 ..., cut with the window's SBs as one sequence); earlier: summaries of earlier
    stretches (T1 ...); progress: steps still in progress (N1 ...)."""

    def lines(prefix: str, items: list[dict]) -> list[str]:
        return [
            f"{prefix}{n + 1} [{s['time']}] {s['text']}"
            if s.get("time")
            else f"{prefix}{n + 1} {s['text']}"
            for n, s in enumerate(items)
        ]

    return STRETCH[lang].format(
        earlier="\n".join(f"T{n + 1}: {t}" for n, t in enumerate(earlier)) or "-",
        progress="\n".join(f"N{n + 1}: {t}" for n, t in enumerate(progress)) or "-",
        sbs="\n".join([*lines("o", open_sbs), *lines("s", sbs)]),
    )


_SRANGE = re.compile(r"([os])(\d+)(?:\s*[-–~～至]\s*([os])?(\d+))?")


def parse_stretches(
    data: object, n_sbs: int, n_earlier: int, n_progress: int, n_open: int = 0
) -> dict:
    """{"stretches": [{"sbs": [i ...], "joins_open", "from": k | None, "closed", "summary"}],
    "open_summary": str, "settles": {i: [k ...]}} (0-based over the window's SBs). The open
    stretch's o1..ok and the window's s1..sn are cut as one sequence: covered in order, no gap,
    no overlap, the o's together in the first stretch. The first stretch running on into the s's
    carries the open stretch on (joins_open); ending at ok, the open stretch has closed there,
    and its summary is "open_summary" (2026-10-31: with the open stretch outside the sequence and
    only "carry it on or not" asked, one left behind was never summarized). A closed stretch
    needs a summary."""
    if not isinstance(data, dict) or not isinstance(data.get("stretches"), list):
        raise ValueError("'stretches' must be a list")

    def at(prefix: str, n: str) -> int:
        k = int(n) - 1
        if prefix == "o":
            if not 0 <= k < n_open:
                raise ValueError(f"o{n} is not one of o1-o{n_open}")
            return k
        return n_open + k

    def name(i: int) -> str:
        return f"o{i + 1}" if i < n_open else f"s{i - n_open + 1}"

    total, out, nxt, open_summary = n_open + n_sbs, [], 0, ""
    for x in data["stretches"]:
        if not isinstance(x, dict):
            raise ValueError("a stretch must be an object")
        m = _SRANGE.fullmatch(str(x.get("sbs", "")).strip())
        if not m:
            raise ValueError(f"stretch sbs not a range: {x.get('sbs')!r}")
        a = at(m.group(1), m.group(2))
        b = at(m.group(3) or m.group(1), m.group(4) or m.group(2))
        if a != nxt or b < a or b >= total:
            raise ValueError(
                f"stretches must cover {name(0)}-{name(total - 1)} in order;"
                f" got {name(a)}-{name(b)}"
            )
        if a == 0 and n_open and b < n_open - 1:
            raise ValueError(f"o1-o{n_open} are one stretch, not to be split")
        nxt = b + 1
        frm = str(x.get("from") or "").strip()
        k = int(frm[1:]) - 1 if frm[:1] == "T" and frm[1:].isdigit() else None
        closed = bool(x.get("closed")) or b < n_open  # followed by the s's: it has closed
        summary = str(x.get("summary") or "").strip()[:2000]
        if closed and not summary:
            raise ValueError(f"closed stretch {name(a)}-{name(b)} has no summary")
        if b < n_open:  # the open stretch alone: it closed before the window's SBs
            open_summary = summary
            continue
        out.append(
            {
                "sbs": [i - n_open for i in range(max(a, n_open), b + 1)],
                "joins_open": bool(n_open) and a == 0,
                "from": k if k is not None and 0 <= k < n_earlier else None,
                "closed": closed,
                "summary": summary if closed else "",
            }
        )
    if nxt != total:
        raise ValueError(f"stretches cover {name(0)}-{name(nxt - 1)}, not to {name(total - 1)}")
    settles: dict[int, list[int]] = {}
    for key, refs in (data.get("settles") or {}).items():
        k = str(key).strip()
        if k[:1] == "s" and k[1:].isdigit() and 0 < int(k[1:]) <= n_sbs:
            got = [
                int(r[1:]) - 1
                for r in (refs if isinstance(refs, list) else [refs])
                if str(r)[:1] == "N" and str(r)[1:].isdigit() and 0 < int(str(r)[1:]) <= n_progress
            ]
            if got:
                settles[int(k[1:]) - 1] = got
    return {"stretches": out, "open_summary": open_summary, "settles": settles}


STATE = {
    "zh": {"happened": "已发生", "in progress": "进行中"},
    "en": {"happened": "happened", "in progress": "in progress"},
}
WHO2 = {
    "zh": {"person": "人", "assistant": "助手"},
    "en": {"person": "the person", "assistant": "the assistant"},
}


def step2_prompt(
    lang: str, sbs: list[dict], cands: list[dict], per_sb: list[list[int]], scene: str = "work"
) -> str:
    """sbs: {"text", "time"}; cands: {"text", "time", "state"} (N1.. in this order);
    per_sb[i]: the candidate indexes listed under SB i. In the chat scene a candidate may carry
    "matter" (the index of the person's candidate it follows, itself for one of the person) and
    "who"; candidates are then shown under their matter, and an SB lists the matters of its own."""

    def detail(j: int, c: dict) -> str:
        state = STATE[lang].get(c.get("state") or "", "") if c.get("state") else ""
        who = f" ({WHO2[lang][c['who']]})" if c.get("who") in WHO2[lang] else ""
        return f"N{j + 1}{who} [{c.get('time', '')}{', ' + state if state else ''}] {c['text']}"

    grouped = scene == "chat" and any("matter" in c for c in cands)
    matter = [c.get("matter", j) if grouped else j for j, c in enumerate(cands)]
    word = (
        ("早先的事" if lang == "zh" else "matters")
        if grouped
        else ("候选" if lang == "zh" else "candidates")
    )
    only = {"zh": "（只看 settles）", "en": " (settles only)"}[lang]
    lines = [
        f"s{i + 1}{only if s.get('settles_only') else ''} [{s.get('time', '')}] "
        f"{s['text']}\n   {word}: "
        + " ".join(f"N{j + 1}" for j in dict.fromkeys(matter[j] for j in per_sb[i]))
        for i, s in enumerate(sbs)
    ]
    if grouped:
        details = []
        for m in dict.fromkeys(matter):
            details.append(detail(m, cands[m]))
            details += [
                "   " + detail(j, c) for j, c in enumerate(cands) if matter[j] == m and j != m
            ]
    else:
        details = [detail(j, c) for j, c in enumerate(cands)]
    if scene != "chat":
        return STEP2[lang].format(sbs="\n".join(lines), cands="\n".join(details) or "-")
    return STEP2_CHAT[lang].format(sbs="\n".join(lines), cands="\n".join(details) or "-")


def parse_step2(data: dict, n_sbs: int, n_cands: int, n_threads: int = 0) -> dict:
    """{"after": {i: [("N", j) | ("s", k)]}, "why": {(i, ref): str}, "settles": {i: [j]}} with
    i, k SB indexes (k < i) and j candidate indexes; anything else is dropped."""
    if not isinstance(data, dict) or not isinstance(data.get("links", []), list):
        raise ValueError("'links' must be a list")

    def num(x: object, prefix: str, limit: int) -> int | None:
        s = str(x or "").strip()
        return (
            int(s[1:]) - 1
            if s[:1] == prefix and s[1:].isdigit() and 0 < int(s[1:]) <= limit
            else None
        )

    after: dict[int, list] = {}
    why: dict = {}
    for link in data.get("links") or []:
        if not isinstance(link, dict) or (i := num(link.get("sb"), "s", n_sbs)) is None:
            continue
        to = str(link.get("to", "")).strip()
        if (j := num(to, "N", n_cands)) is not None:
            ref = ("N", j)
        elif (k := num(to, "s", n_sbs)) is not None and k < i:
            ref = ("s", k)
        else:
            continue
        if ref not in after.setdefault(i, []):
            after[i].append(ref)
            why[(i, ref)] = str(link.get("why", ""))[:300]
    settles: dict[int, list[int]] = {}
    for s in data.get("settles") or []:
        if (
            isinstance(s, dict)
            and (i := num(s.get("sb"), "s", n_sbs)) is not None
            and (j := num(s.get("step"), "N", n_cands)) is not None
        ):
            settles.setdefault(i, []).append(j)
    # threads: each SB's thread - ("T", t) an earlier one, ("s", k) the one SB k of the window
    # began (k == i: it begins one itself); summaries are keyed the same way
    thread: dict[int, tuple] = {}
    came: dict[int, tuple] = {}
    for x in data.get("threads") or []:
        if not isinstance(x, dict) or (i := num(x.get("sb"), "s", n_sbs)) is None:
            continue
        to = str(x.get("thread", "")).strip()
        if (t := num(to, "T", n_threads)) is not None:
            thread[i] = ("T", t)
        elif (k := num(to, "s", n_sbs)) is not None and k <= i:
            thread[i] = ("s", k)
        elif to.lower() == "new":
            thread[i] = ("s", i)
        if thread.get(i) == ("s", i):  # a new thread: the step whose result led to it, if any
            frm = str(x.get("from", "")).strip()
            ref = (
                ("N", j)
                if (j := num(frm, "N", n_cands)) is not None
                else ("s", k)
                if (k := num(frm, "s", n_sbs)) is not None and k < i
                else None
            )
            if ref is not None:
                came[i] = (ref, str(x.get("why", "")).strip()[:300])
    summaries: dict[tuple, str] = {}
    for x in data.get("summaries") or []:
        if not isinstance(x, dict) or not str(x.get("summary", "")).strip():
            continue
        to = str(x.get("thread", "")).strip()
        key = (
            ("T", t)
            if (t := num(to, "T", n_threads)) is not None
            else ("s", k)
            if (k := num(to, "s", n_sbs)) is not None
            else None
        )
        if key is not None:
            summaries[key] = str(x["summary"]).strip()[:4000]
    return {
        "after": after,
        "why": why,
        "settles": settles,
        "thread": thread,
        "summaries": summaries,
        "from": came,
    }


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the output")
    return json.loads(text[start : end + 1])
