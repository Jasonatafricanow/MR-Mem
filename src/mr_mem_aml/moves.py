"""Work scene, two-step compile (user 2026-10-23/24; design 4.1, 4.2, 4.6).

The work scene is centred on the person: the person decides, the agent carries out. An SB is
"what happened -> what the person did":

  step 1 (one call per window) writes the SBs: one per point the person makes (a message with
         several independent points gives several), each with the agent messages it responds to,
         the situation in plain words (the agent's own wording, self-assessment and pet phrases
         washed out) and the person's move. An SB is a projection, not a notebook (user 2026-10-22/24): the
         situation in one or two sentences, the move as what the person settled; the details stay
         in the raw text. A report nobody answered is no SB of its own: when the person answers
         it, it goes into that point's situation. No earlier steps are shown: writing and linking are separate.
  step 2 (one call per window) places them: each SB gets candidates retrieved with its own text
         (runtime), and the model says which of them it builds on - the decision being carried
         out, or the person's own earlier decision or judgement brought back, changed or reversed,
         however far back. Sharing a concept is not building on.

The chat scene is centred on the dialogue (the interaction is the content) and does not use this
module. The prompt follows the person's language, decided once per user ("zh" or "en"); the rules are
the same in both, only the wording, the examples and the output language differ.
Strings never quote with ASCII double quotes inside the JSON (models break the JSON with them)."""

from __future__ import annotations

import json
import re

ARROW = {"zh": " → 人：", "en": " -> Person: "}
NO_MOVE = {"zh": "（无回应）", "en": "(no response)"}
_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z]+")


def person_language(texts: list[str]) -> str:
    """"zh" or "en" from the person's own messages only (the agent's process talk is often in
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
    """(situation, move) of an SB written by this module; move is "" for content without the arrow."""
    for arrow in ARROW.values():
        if content.startswith(arrow.lstrip()):
            return "", content[len(arrow.lstrip()):]
        if arrow in content:
            situation, move = content.split(arrow, 1)
            return situation, move
    return content, ""


# --- step 1 ---------------------------------------------------------------------

STEP1 = {
"zh": """你是长期记忆系统的语义编译器。下面是人和编程 Agent 的工作对话。工作场景以人为主：人是决策主体，内容是项目的真实进度和问题。把 NEW 编成语义块（SB）：每个 SB 是“发生的情况 → 人的动作”。先把 NEW 连同 CONTEXT 整体读懂，再写。

输出
moves：人的每一点各一项。人一条消息里有几处各自独立的意思（有没有编号都一样），就是几项。
  - msg：人这条消息的 M 编号。
  - about：这一点针对的 Agent 消息（M 编号，通常是人开口前那一两条汇报里和这一点有关的部分）；人主动提出的新需求可以为空。
  - situation：只写 about 里 Agent 那边带来的、人这一点所回应的那部分：出了什么问题、得到了什么结果、提了什么方案或问题。一句话，最多 60 字。Agent 列了好几点、人只回应其中一两点，就只写那一两点；人只是同意或让它继续，就只写它提议做什么。点出关键的具体对象（哪个功能、哪个文件、起决定作用的那个数字），单独读得懂，不写“这个问题”“该方案”这种指代。不写 Agent 的执行过程、逐项清单和汇报里的枝节，不沿用它的措辞和自我评价；细节留在原文里，SB 是理解不是转述。人说的话不写进 situation；about 为空（人主动提出）时 situation 留空。
  - move：只写人在这一点上定下了什么：决定了什么、纠正成什么、否定了什么、要求做什么、问了什么。一两句，最多 80 字，写出他定下的具体内容（比如选定的键、格式、取舍、标准），保留他的关键词，不整段抄原话，不重复 situation 已写的内容。人同意或让 Agent 照做时，写出他采纳的具体内容（比如「同意导出改用 CSV、按日期分文件」），不只写「同意」「让它继续」。
  - happened：说的时候这件事是否已经发生。人的决定、纠正、反馈本身是已发生（true）；人提出的以后要做的计划是 false。
几个会话交错在一个窗口里时，消息前标着会话（会话A、会话B……）：不同会话是不同的事，人的一点只回应同一会话里的 Agent 消息，about 只列同一会话的，situation 也只写那个会话里的事。
没人回应的 Agent 汇报不单独成 SB：人后来回应它，就写进那一点的 situation，about 列上它；人始终没回应，不编。
situation 和 move 都用第三人称写：人写作「人」，Agent 写作「Agent」（原文里 Agent 对人说的「你」指人，「我」指 Agent），不用「你」「我」。
内容里说到的相对时间，按消息时间换成 YYYY-MM-DD 或 YYYY-MM-DD HH:MM。

用中文写。只输出紧凑 JSON，字符串里不要用英文双引号（引用用「」）：
{{"moves":[{{"msg":"M5","about":["M3","M4"],"situation":"...","move":"...","happened":true}}]}}

CONTEXT（之前的消息，只供理解，不编译）：
{context}

NEW：
{new}""",
"en": """You are the semantic compiler of a long-term memory system. Below is a work conversation between a person and a coding agent. A work conversation is centred on the person: the person decides, and the content is the real progress of the project and its problems. Compile NEW into semantic blocks (SBs): each SB is "what happened -> what the person did". Read NEW together with CONTEXT as a whole first, then write.

Output
moves: one item per point the person makes. A message with several independent points (numbered or not) gives several items.
  - msg: the M id of the person's message.
  - about: the agent messages this point responds to (M ids; usually the part of the report just before the person spoke that this point is about); empty for a new request the person raises on their own.
  - situation: only the part of what the agent brought in about that this point of the person responds to: what went wrong, what came out, what it proposed or asked. One sentence, at most 40 words. When the agent lists several points and the person answers one or two, write only those; when the person just agrees or lets it go on, write only what it proposed to do. Name the key specific object (which feature, which file, the number that decides it) so the SB reads on its own, never "this issue" or "the proposal". Leave out the agent's process, item-by-item lists and the side details of its report, and do not carry over its wording or self-assessment; the details stay in the raw text: an SB is understanding, not a retelling. What the person said does not go into the situation; when about is empty (the person raises it on their own) the situation is empty.
  - move: only what the person settled on it: what they decided, corrected it to, refused, asked to be done, or asked. One or two sentences, at most 50 words, giving the specific content they settled (the key, format, trade-off or standard they chose), keeping their key words; do not copy the whole message or repeat what the situation already says. When the person agrees or lets the agent go ahead, write the specific content they adopted (e.g. "agrees to switch the export to CSV, one file per day"), not just "agrees" or "lets it go on".
  - happened: whether it had already happened when said. The person's decision, correction or feedback itself has (true); a plan the person sets for later has not (false).
When several sessions interleave in one window, each message is marked with its session (session A, session B ...): different sessions are different work; a point of the person responds only to agent messages of the same session, about lists only those, and the situation is about that session only.
A report of the agent that nobody answered is no SB of its own: when the person answers it later, it goes into that point's situation, with it in about; if the person never answers it, it is not compiled.
Write situation and move in the third person: "the person" and "the agent" (in the messages, the agent's "you" is the person and its "I" is the agent); never "you" or "I".
Write a relative time in the content as YYYY-MM-DD or YYYY-MM-DD HH:MM, resolved against the message time.

Write in English. Return only compact JSON, no double quotes inside strings:
{{"moves":[{{"msg":"M5","about":["M3","M4"],"situation":"...","move":"...","happened":true}}]}}

CONTEXT (earlier messages, for understanding only, not compiled):
{context}

NEW:
{new}"""}

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
        if line.lstrip().startswith("```") and not fence and not opens:  # code closed: its paragraph ends
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
    """Paragraph numbers k and (k, first line, last line) of a code paragraph, as character spans."""
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
    """Lowercased, whitespace runs as one space, quotes made plain; with each kept character's place."""
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
            body.append(f"({label}.{k}) {p.split(chr(10))[0]}\n"
                        + "\n".join(f"  {j}| {line}" for j, line in enumerate(code_lines(p), 1)) + "\n  ```")
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
    tag = (lambda m: f" {SESSION[lang]}{chr(65 + order.index(m.get('session')) % 26)}") if len(order) > 1 else (lambda m: "")
    return STEP1[lang].format(
        context="\n".join(_line(lang, "", m, 600, tag(m)).replace("[ ", "[", 1) for m in context) or "-",
        new="\n".join(_line(lang, f"M{i + 1}", m, None, tag(m)) for i, m in enumerate(new)))


def parse_step1(data: dict, roles: list[str], sessions: list | None = None) -> list[dict]:
    """SBs in the order their last message was said: {"sources": [message indexes], "situation",
    "move", "happened"}. Unknown ids are dropped; a move whose msg is not the person's is dropped;
    about keeps only earlier agent messages of the person's own session (``sessions`` per message)."""
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
        about = {ids[a] for a in m.get("about") or [] if isinstance(a, str) and a.strip() in ids
                 and roles[ids[a.strip()]] != "user" and ids[a.strip()] < i
                 and (sessions is None or sessions[ids[a.strip()]] == sessions[i])}
        out.append({"sources": sorted(about | {i}), "situation": str(m.get("situation", "")).strip(),
                    "move": str(m["move"]).strip(), "happened": m.get("happened") is not False})
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

STEP1_SLIM4 = {"zh": """你是 SemanticBlock（SB）编译器。把对话编译成若干 SB，每个 SB 记录"一件事"。

## 切分规则
1. 先通读全文再切。一个 SB 是"这一段做了什么、结果怎样"只有一个答案的一段。切点落在这个答案变了的地方：做的事变了、结果变了，或出现了转折（被纠正、被否决、出了意外、得出结论）。
   - 同一个大目标下的不同进展，只要各有各的结果，就是不同的 SB。例如"出门→打车→没赶上航班"是三个 SB，不是一个。
   - 一个请求和对它的执行、汇报、答复，如果只得出同一个结果，是一个 SB。
   - 后一段要靠前一段的背景才讲得通，这不是合并的理由。
2. 没有自己结果的内容，不单独成 SB：只说"现在去做 X"的过程播报、一句没人接的话、没展开的术语，归入它所属或所引出的那个 SB（不单独写出，细节留在原文）。
3. 解释一个词或概念、或由此引出的旁枝：有来有回（至少一问一答，或接着聊了几句）、有自己的结果，就单独成 SB；只提了一句、没人接，归入当前 SB，不单独写出。
4. 不按表面话题词、轮次或"一问一答"配对机械切分；看起来像新话题、其实在佐证当前这件事的，归入当前 SB。
5. 消解相对时间和指代："今天/昨天/这个/它"写成具体日期或对象。回忆中的"今天"指回忆那天，不是说话那天。每条消息有时间可用。
6. 某字段在原文里没有体现就填 null，不要编。

## unresolved
7. 待定 = 这件事已经推进出了结果，下一步只能由人拍板，而且到对话结束人都没有回应。填人该拍板的那条消息的 M 编号，否则填 null。
   判断看"下一步在谁手里"，不看措辞：要等实验、运行、数据或别的事情发生才能定的，不是在等人，不算待定；只是提了个想法、问了个问题、还没有结果的，不算待定；后面已经回答、决定或做了的，不算待定。

## 每个 SB 的字段
- id
- turns: 这件事覆盖的 NEW 消息范围，如 "M3-M7"（只有一条写 "M3"）；长消息按段落编了号（M3.1、M3.2……），只有其中几段属于这件事时写到段落，如 "M3.2-M3.5"
- summary: 一段短摘要，只写这一件事的三样东西：做了什么、结果怎样、人在其中的决定或保留（没有就不写这一样）。细节、数字、原话都留在原文，不要写进来。摘要不是复述。
- unresolved: 见规则 7

## 输出
只输出合法 JSON 数组，不要其他文字，字符串里不要用英文双引号（引用用「」）：[{{"id":"b1","turns":"M1-M4","summary":"...","unresolved":null}}]

CONTEXT（之前的消息，只供理解，不编译）：
{context}

NEW：
{new}""",
"en": """You are a SemanticBlock (SB) compiler. Compile the conversation into SBs; each SB records "one matter".

## Cutting
1. Read it all before cutting. An SB is a stretch where "what was done or said, and how it came out" has a single answer. Cut where that answer changes: what is being done changes, the result changes, or there is a turn (corrected, refused, something unexpected, a conclusion reached).
   - Different steps toward one larger goal are different SBs when each has its own result. E.g. "left home -> took a taxi -> missed the flight" is three SBs, not one.
   - A request and its handling, report and answer are one SB when they reach a single result.
   - That a later stretch needs an earlier one as background is not a reason to merge them.
2. Content with no result of its own is not an SB of its own: a "now I will do X" progress note, a remark nobody picks up, an unexplained term go into the SB they belong to or lead to (not written out; the details stay in the raw text).
3. Explaining a word or concept, or a side branch it leads to: with a real exchange (at least a question and an answer, or a few more lines) and a result of its own, it is its own SB; mentioned once and not picked up, it goes into the current SB, not written out.
4. Do not cut mechanically by surface topic words, by turns or by question-answer pairs; what looks like a new topic but supports the current matter goes into the current SB.
5. Resolve relative time and reference: write "today / yesterday / this / it" as the concrete date or object. In a recollection, "today" is the day recalled, not the day it is said. Every message has its time.
6. A field the text does not show is null; do not invent.

## unresolved
7. Pending = the matter has come to a result and the next step can only be taken by the person, who has not responded by the end of the conversation. Give the M id of the message where the person should decide, else null.
   Judge by whose hands the next step is in, not by wording: what waits for an experiment, a run, data or some other event is not waiting for the person; an idea floated or a question asked with no result yet is not pending; what is later answered, decided or done is not pending.

## Fields of each SB
- id
- turns: the NEW messages this matter covers, e.g. "M3-M7" ("M3" for one); a long message is shown in numbered paragraphs (M3.1, M3.2 ...): when only some of them belong to this matter, give those paragraphs, e.g. "M3.2-M3.5"
- summary: a short summary of only three things about this matter: what was done or said, how it came out, and the person's decision or reservation in it (leave out what is not there). When several people talk, say who said what. Details, numbers and exact words stay in the raw text; do not put them here. A summary is not a retelling.
- unresolved: see rule 7

## Output
Return only a valid JSON array, nothing else, no double quotes inside strings: [{{"id":"b1","turns":"M1-M4","summary":"...","unresolved":null}}]

CONTEXT (earlier messages, for understanding only, not compiled):
{context}

NEW:
{new}"""}


SUMMARY = {
    "zh": "- summary: 一段短摘要，只写这一件事的三样东西：做了什么、结果怎样、人在其中的决定或保留（没有就不写这一样）。细节、数字、原话都留在原文，不要写进来。摘要不是复述。",
    "en": "- summary: a short summary of only three things about this matter: what was done or said, how it came out, and the person's decision or reservation in it (leave out what is not there). When several people talk, say who said what. Details, numbers and exact words stay in the raw text; do not put them here. A summary is not a retelling.",
}
# The chat scene (SB freeze 2026-09-25; user 2026-10-27): an SB is a semantic point, one thing
# someone meant at that time, cut by meaning and not by message; who meant it and how firmly is
# written in its text, not split into fields. Its status is happened, not yet or unknown.
STEP1_CHAT = {"zh": """你是 SemanticBlock（SB）编译器。把下面的聊天编译成 SB，并写出它们之间谁在谁的基础上推进。

SB 是一件事：某一方在那个时点说出的一个完整意思，离开原文也不会被误读，并且对应原文里的一段话。人说的话决定这件事是什么；助手的回答是接在人这件事后面的步骤。

## 怎么切
1. 按意思切，不按句子、条目、段落切。人的一条消息里说了几件不同的事，就是几个 SB（"我测了是 150ms，怎么更快？"是一次测量和一个提问）；人交代的已经做了的、测到的、现在的状态，和他要的是不同的意思，即使是为了引出请求才说的，也各自成 SB；人提出一个需求、又对它提了几条意思不同的要求，需求和各条要求分开。同一类东西连着罗列（一串想用的库、一张待办清单、一组例子）是一个意思，是一个 SB，具体有哪些留在原文里；罗列里某一项人另外说了什么（单独的决定、事实、理由），它才单独成 SB。一件事要人的几条消息才完整，就是一个 SB。
2. 助手的回复按它带来的新意思来切：对某件事给出的方案、做法、结论、解释，意思不同的各是一个 SB；同一个意思展开成几步、几条、几段代码，仍是一个 SB。助手复述、汇总已经说过的东西（列出人还有哪些待办、回顾前面的方案）不是新的事，不为其中每一项再立 SB，整段复述至多是一个 SB。
3. 寒暄、过程播报这类没有意思的话不成 SB。

## 怎么写
4. text：一句话写这件事，写明是谁（人……／助手……），按说话者的本意，相对时间和指代换成具体的；代码留在原文。人的 SB 保留别人可能问到的值（名称、版本、日期、数字）。助手的 SB 只写它给了什么、为人的哪件事给的（"助手给了一份和 HR 谈福利的通话话术"），不列出里面的条目和值：事情是人推进的，助手给的东西是人决策的参考，具体内容留在原文，靠索引取。
5. 原文说了的写清，没说的不补；不写后来才发生的事。

## 推进
6. 在本窗口里某个 SB 的基础上推进的 SB，写进那个 SB 的 next：大需求里的具体事写在大需求下面；助手的回答写在它回答的那件事下面；接着做下去的一步写在它接着的那一步下面。一个 SB 下面有几个 next，就是从它分出几条。不在本窗口任何 SB 基础上的，放在最外层。

## 状态
7. when：从说这句话的那一刻看，这件事在时间轴上的位置（它在对话里的位置，不是任何钟表时间）。"before"：那一刻之前已经发生或已经成立（做完的事、事实、做出的决定、提出的请求、成立的状态）。"after"：在那之后；原文给了日期就写上（"after 2024-03-31"）。一个安排和它安排的事位置不同：安排（人的计划、项目排期、截止日、给出的指令）说的时候已经存在，是 before；它安排的事（计划执行、冲刺结束、按期完成、照指令去做）是 after，分成不同的 SB。
8. status 跟着 when 定：
   - before：发生了或成立填 "confirmed"；没发生填 "negated"。
   - after：已经安排好的（人定了或坚定要做，或排期、截止日定了它）填 "likely"；取消、拒绝、决定不做的填 "negated"；只是提议或推荐、人没接的填 "unknown"。
   - 两边都没有证据：填 "unknown"。
   likely 的事以后由新消息确认或否定，这里不判断。
   助手的 SB 说的是它给出的那件事，不是"助手说了"：助手说了，不等于那件事发生了或成立了。计划的一项、推荐的做法、让人去用的代码，是 after·unknown；它陈述的事实、解释、判断和预期，是人被告知的内容，说过去或现在的是 before、说以后的是 after，人确认、照做或报告结果之前都是 unknown。

## standing
9. 人给出的、从此以后都要遵守的指令或偏好（"以后都……""每次……的时候……""我更喜欢……"），standing 填 true；其余填 false。

## turns
10. turns 指向原话：消息按段落编了号（M3.1、M3.2……），代码段的行也编了号。SB 指向它那段话所在的段落或行（"M3.2"、"M4.3:5-18"），整条消息都是这件事才写 "M3"；并照原文抄下那段话开头的几个词（from）和结尾的几个词（to），让代码找到它的确切位置；指向代码行时可以不抄。

## 在哪里停
NEW 不必在一次回答里编完：一次回答最多大约 {cap} 个 SB。按顺序编，每条消息都照规则切足。下一条消息会让你超出这个数时，就停在它前面，停在一个此前所有 SB 都已完整收口的消息之后；后面的消息在下一次回答里编，这些作为上下文。不要为了编到最后而缩短、合并或漏掉 SB。

## 输出
先写 ANALYSIS：用平常的话按顺序过一遍消息：人的每条消息说了哪几件事，一件件点名；助手对每件事回答了哪几块。然后另起一行写 JSON:，后面只跟合法 JSON 数组，字符串里不要用英文双引号（引用用「」）：
[{{"turns":"M1","from":"...","to":"...","text":"人……","when":"before","status":"confirmed","standing":false}},
 {{"turns":"M1","from":"...","to":"...","text":"人请求……","when":"before","status":"confirmed","standing":false,"next":[
   {{"turns":"M1","from":"...","to":"...","text":"人要……","when":"after","status":"likely","standing":false,"next":[
     {{"turns":"M2.3-M2.5","from":"...","to":"...","text":"助手建议……","when":"after","status":"unknown","standing":false}}]}},
   {{"turns":"M2.4:3-12","text":"助手给出……","when":"before","status":"unknown","standing":false}}]}}]
JSON 之后另起一行写 STOP: M<k>，k 是你编到的最后一条消息。

CONTEXT（之前的消息，只供理解，不编译）：
{context}

NEW：
{new}""",
"en": """You are a SemanticBlock (SB) compiler. Compile the chat below into SBs and write which builds on which.

An SB is one thing: one complete meaning said by one side at that time, which cannot be misread without the raw text, and which sits in a stretch of the words. What the person says defines what a thing is; the assistant's answers are steps that follow the person's thing.

## Cutting
1. Cut by meaning, not by sentences, items or paragraphs. A person's message that says several different things gives several SBs ("I measured 150ms; how can I make it faster?" is a measurement and a question); what the person says is already done, measured or the current state means something other than what they want, even when they say it to lead up to the request, so each is an SB of its own; when the person brings up a need and several requirements for it that mean different things, the need and each requirement are SBs. Things of one kind listed together (a string of libraries they want, a to-do list, a set of examples) are one meaning and one SB; which ones they are stays in the raw text. An item of such a list is an SB of its own only when the person says something more about it (a decision, a fact, a reason of its own). One thing that takes several of the person's messages to be complete is one SB.
2. The assistant's reply is cut by the new meanings it brings: for a thing, the plans, measures, conclusions and explanations it gives that mean different things are SBs; one meaning spread over several steps, items or pieces of code is still one SB. When the assistant restates or sums up what was already said (listing the person's remaining to-dos, going over an earlier plan), that is not a new thing: no SB for each item; the restatement is at most one SB.
3. Small talk and progress notes that carry no meaning are not SBs.

## Writing an SB
4. text: the thing in one sentence, saying who (the person ... / the assistant ...), as the speaker meant it, with relative times and references made concrete; code stays in the raw text. An SB of the person keeps the values someone may ask for (names, versions, dates, numbers). An SB of the assistant says only what it gave and for which matter of the person ("The assistant gives a script for the benefits call with HR"), without listing its items or values: the person moves the matter on, what the assistant gives is what the person decides on, and its details stay in the raw text, reached by the index.
5. Write what the raw text says, add nothing it does not; do not write what happened later.

## Building on
6. An SB that builds on an SB of this window goes in that SB's next: a concrete thing of a larger need under the need; an answer under the thing it answers; a step carried on under the step it carries on. Several next under one SB are the branches from it. An SB that builds on no SB of this window stays at the top level.

## status
7. when: where the thing sits on the timeline, seen from the moment it was said (its place in the conversation, not any clock). "before": it had already happened, or it already held, by then (something done, a fact, a decision made, a request made, a state that holds). "after": it comes later; add its date when the raw text gives one ("after 2024-03-31"). An arrangement and the thing it arranges sit at different places: the arrangement (the person's plan, a project schedule, a deadline, an instruction given) already exists when it is said, so it is before; the thing it arranges (the plan carried out, the sprint ending, the deadline met, the instruction followed) is after; they are separate SBs.
8. status follows from when:
   - when before: "confirmed" if it happened or holds; "negated" if it did not happen.
   - when after: "likely" if it is arranged: the person set it or firmly intends it, or a schedule or deadline fixes it; "negated" if it was cancelled, refused or decided against; "unknown" if it was only proposed or recommended and the person did not take it up.
   - no evidence either way: "unknown".
   Later messages confirm or negate what is likely; that is not judged here.
   An SB of the assistant is about the thing it gives, not that the assistant said it: the assistant saying it does not make it happen or hold. An item of a plan, a recommended measure, code for the person to use is after and unknown; a fact, explanation, judgement or expectation it states is what the person was told, before when it is about the past or present and after when it is about what will be, and unknown until the person confirms it, takes it up or reports how it went.

## standing
9. An instruction or preference the person gives that is meant to hold from now on ("always ...", "whenever ..., do ...", "I prefer ..."): standing is true; otherwise false.

## turns
10. turns point to the words: messages are shown in numbered paragraphs (M3.1, M3.2 ...), and the lines of a code paragraph are numbered too. An SB points to the paragraphs or lines its words are in ("M3.2", "M4.3:5-18"), "M3" only when the whole message is this thing; and copies, exactly as written, the first few words (from) and the last few words (to) of its words, so the code can find exactly where they are; not needed when it points to code lines.

## Where to stop
You need not compile all of NEW in one answer: one answer holds about {cap} SBs at most. Compile it in order, each message as fully as the rules ask. When the next message would take you past that, stop before it, after a message where every SB so far is complete; the messages after it are compiled in the next answer, with these as context. Do not shorten, merge or leave out SBs to reach the end.

## Output
First write ANALYSIS: in plain words, go through the messages in order: for each message of the person, the things it says, each named; for each thing, the pieces the assistant answers with. Then write a line JSON: followed by only a valid JSON array, no double quotes inside strings:
[{{"turns":"M1","from":"...","to":"...","text":"The person ...","when":"before","status":"confirmed","standing":false}},
 {{"turns":"M1","from":"...","to":"...","text":"The person asks ...","when":"before","status":"confirmed","standing":false,"next":[
   {{"turns":"M1","from":"...","to":"...","text":"The person wants ...","when":"after","status":"likely","standing":false,"next":[
     {{"turns":"M2.3-M2.5","from":"...","to":"...","text":"The assistant recommends ...","when":"after","status":"unknown","standing":false}}]}},
   {{"turns":"M2.4:3-12","text":"The assistant gives ...","when":"before","status":"unknown","standing":false}}]}}]
After the JSON write a line STOP: M<k>, the last message you compiled.

CONTEXT (earlier messages, for understanding only, not compiled):
{context}

NEW:
{new}"""}


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
    return [{**b, "under": index.get(id(sbs[b["under"]])) if b.get("under") is not None else None} for b in keep]


def step1_prompt_slim4(lang: str, new: list[dict], context: list[dict], scene: str = "work") -> str:
    """As step1_prompt: NEW messages are M1, M2 ... in order. The chat scene has its own
    template (STEP1_CHAT)."""
    order = list(dict.fromkeys(m.get("session") for m in [*context, *new]))
    tag = (lambda m: f" {SESSION[lang]}{chr(65 + order.index(m.get('session')) % 26)}") if len(order) > 1 else (lambda m: "")
    template = STEP1_CHAT[lang] if scene == "chat" else STEP1_SLIM4[lang]
    return template.format(
        cap=ANSWER_SBS,
        context="\n".join(_line(lang, "", m, 600, tag(m)).replace("[ ", "[", 1) for m in context) or "-",
        new="\n".join(_numbered(lang, f"M{i + 1}", m, tag(m)) for i, m in enumerate(new)))


_RANGE = re.compile(r"M(\d+)(?:\s*[-–~～至]\s*M?(\d+))?")
_LINES = re.compile(r"M(\d+)\.(\d+):(\d+)(?:\s*[-–~～至]\s*(\d+))?")
_TURN = re.compile(r"M(\d+)(?:\.(\d+))?(?:\s*[-–~～至]\s*(M?)(\d+)(?:\.(\d+))?)?")


def span_key(item) -> tuple:
    """Order of span items: a paragraph number k, or (k, first line, last line) of a code paragraph."""
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
        if b and ap and not bm and not bp:      # M3.2-5: paragraphs of one message
            lo, hi, lo_p, hi_p = int(a), int(a), int(ap), int(b)
        else:
            lo, hi = int(a), int(b or a)
            lo_p, hi_p = (int(ap) if ap else 0), (int(bp) if bp else (int(ap) if ap and not b else 0))
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
        out.append({"sources": sorted(sources), "spans": {i: sorted(p, key=span_key) for i, p in spans.items()},
                    "situation": text, "move": "",
                    "happened": "status" not in b or str(b["status"]).strip().lower() == "happened",
                    "standing": b.get("standing") is True or str(b.get("standing")).lower() == "true",
                    "unresolved": int(u.group(1)) - 1 if u and 0 < int(u.group(1)) <= n else None})
    out.sort(key=lambda b: (b["sources"][-1], b["sources"][0]))
    return out


STATUS = ("confirmed", "likely", "negated", "unknown")


def extract_after_json(text: str) -> list:
    """The array after the last "JSON:" line (the ANALYSIS before it may hold brackets)."""
    return extract_json_array(text[text.rfind("JSON:"):] if "JSON:" in text else text)


def _stamp(x: dict) -> str:
    status = str(x.get("status") or "unknown").strip().lower()
    when = str(x.get("when") or "").strip()
    return f"[{status if status in STATUS else 'unknown'}{'; ' + when if when else ''}]"


def _fragments(texts, idxs, items: dict, first, last) -> tuple[dict, int]:
    """{message: [[start, end], ...] or None (whole)} for the messages idxs: the copied words when
    they are found, else the numbered paragraphs or lines; and 1 when copied words were not found."""
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


def parse_step1_chat(data: object, roles: list[str], paras: list[int] | None = None,
                     texts: list[str] | None = None) -> list[dict]:
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
        out.append({"sources": sorted(sources), "spans": {i: sorted(p, key=span_key) for i, p in own.items()},
                    "situation": f"{text} {_stamp(b)}", "move": "",
                    "pointers": {"own": mine, "parts": [], **({"chars": True, "missed": missed} if texts else {})},
                    "happened": status == "confirmed",
                    "standing": b.get("standing") is True or str(b.get("standing")).lower() == "true",
                    "unresolved": None})
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
    for i, (role, text) in enumerate(zip(roles, texts)):
        mine = [b for b in sbs if b["pointers"]["own"].get(i)]  # SBs with fragments in this message
        if role != "user" or not mine:
            continue
        for (s, e), p in zip(piece_spans(text), paragraphs(text)):
            if not is_code(p) or any(a < e and s < z for b in mine for a, z in b["pointers"]["own"][i]):
                continue
            ends = {id(b): max((z for _, z in b["pointers"]["own"][i] if z <= s), default=-1) for b in mine}
            to = (max(mine, key=lambda b: ends[id(b)]) if max(ends.values()) >= 0
                  else min(mine, key=lambda b: b["pointers"]["own"][i][0][0]))
            to["pointers"]["own"][i] = sorted([*to["pointers"]["own"][i], [s, e]])


def _without(spans: list, cut: list, text: str) -> list[list[int]]:
    """The character spans with the spans ``cut`` taken out, pieces holding only blanks or
    punctuation dropped."""
    out = []
    for s, e in spans:
        pieces = [[s, e]]
        for c, d in cut:
            pieces = [p for a, b in pieces for p in ([[a, min(b, c)]] if a < c else []) + ([[max(a, d), b]] if b > d else [])]
        out += [p for p in pieces if any(ch.isalnum() for ch in text[p[0]:p[1]])]
    return out


def uncovered(sbs: list[dict], roles: list[str], texts: list[str], words: int = 8) -> list[tuple[int, str]]:
    """The stretches of the person's words that no SB fragment covers, of at least ``words``
    words: (message, the words)."""
    out = []
    for i, (role, text) in enumerate(zip(roles, texts)):
        if role != "user":
            continue
        cov = [False] * len(text)
        for b in sbs:
            if i not in b["pointers"]["own"]:
                continue
            p = b["pointers"]["own"][i]
            for s, e in ([[0, len(text)]] if p is None else p):
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


def check_step1_chat(data: object, roles: list[str], paras: list[int] | None = None,
                     texts: list[str] | None = None) -> None:
    """Most of the person's messages left outside every SB is a broken answer: ask again."""
    sbs = parse_step1_chat(data, roles, paras, texts)
    people = [i for i, r in enumerate(roles) if r == "user"]
    covered = {i for b in sbs for i in b["sources"]}
    if len(people) >= 2 and sum(i in covered for i in people) * 2 < len(people):
        raise ValueError(f"SBs cover {sum(i in covered for i in people)} of the person's {len(people)} messages")


# one "key": "value" per line, the value copied from the raw text (code holds quotes the model leaves bare)
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
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return json.loads(_escape_bare_quotes(text[start:end + 1]))


def check_step1_slim4(data: object, roles: list[str]) -> None:
    """Most of the person's messages left outside every SB is a broken answer: ask again."""
    sbs = parse_step1_slim4(data, roles)
    people = [i for i, r in enumerate(roles) if r == "user"]
    covered = {i for b in sbs for i in b["sources"]}
    if len(people) >= 2 and sum(i in covered for i in people) * 2 < len(people):
        raise ValueError(f"SBs cover {sum(i in covered for i in people)} of the person's {len(people)} messages")


# --- step 2 ---------------------------------------------------------------------

STEP2 = {
"zh": """你在给长期记忆系统的语义块（SB）连承接。下面是人和编程 Agent 工作对话里一个窗口的 SB（s 编号，每个是“发生的情况 → 人的动作”，后面列出它的候选），以及候选详情：更早的步骤（N 编号，标着当时是已发生还是进行中）。

对每个 SB 判断它是在哪些步骤的基础上推进的（after）：
- 这一步里 Agent 在执行的，是人在哪一步做的决定或要求，就接那一步；
- 人在这一步里把自己早先的决定或判断拿回来用、修改或推翻，就接那一步，不管隔多远（比如“开头要求的格式还是不对”接回开头定格式的那一步；“不是说好用测试库吗”接回说好的那一步）；
- 人对 Agent 一次汇报的反馈，接这次汇报所执行的那一步。
可以接它列出的 N 候选，也可以接本窗口里排在它前面的 s。可以有几个，也可以没有。用到同一个概念、话题相同、用词相近、时间挨着，都不是承接；Agent 的叙述里提到某一步也不算，看人这一步自己做了什么。
如果这个 SB 让某个还在进行中的步骤有了结果（做完、定下、放弃），写进 settles。

只输出紧凑 JSON，字符串里不要用英文双引号：{{"links":[{{"sb":"s3","to":"N12","why":"一句话"}}],"settles":[{{"sb":"s3","step":"N7"}}]}}

本窗口的 SB：
{sbs}

候选详情：
{cands}""",
"en": """You are linking semantic blocks (SBs) of a long-term memory system. Below are the SBs of one window of a work conversation between a person and a coding agent (s ids, each "what happened -> what the person did", followed by its candidates), and the candidate details: earlier steps (N ids, marked as happened or in progress at the time).

For each SB, decide which steps it builds on (links):
- the step where the person made the decision or request that the agent is carrying out in this step;
- a step whose decision or judgement the person brings back here, changes or reverses, however far back (e.g. "the format I asked for at the start is still wrong" goes back to the step where the format was set; "didn't we agree to use the test database?" goes back to that agreement);
- for the person's feedback on a report, the step that report carries out.
It may link to its listed N candidates or to an s earlier in this window; several, or none. Sharing a concept, a topic or words, or coming right next in time, is not building on; neither is the agent's narrative mentioning a step: look at what the person did in this step.
When an SB shows that a step still in progress has come to a result (done, decided, dropped), put it in settles.

Return only compact JSON, no double quotes inside strings: {{"links":[{{"sb":"s3","to":"N12","why":"one sentence"}}],"settles":[{{"sb":"s3","step":"N7"}}]}}

SBs of this window:
{sbs}

Candidate details:
{cands}"""}
# The chat scene (the interaction is the content; slim4 SBs): building on is the same matter going on.
# The chat scene's step 2 shows the earlier steps under the person's matter they follow (design 4.1:
# what the person says defines a matter, the assistant's answers are steps after it): asked for steps
# one by one, the model linked every one that looked related and joined unrelated matters.
STEP2_CHAT = {
"zh": """你在给长期记忆系统的语义块（SB）连承接。下面是一段聊天里一个窗口的 SB（s 编号，每个是一件事，后面列出给它的早先的事），以及早先的事：每件是人提出的一件事（标"人"的 N），它后面跟着的步骤缩进列在下面（N 编号，人的或助手的）；每步标着当时是已发生还是进行中。

对每个 SB 判断它接着推进的是哪件早先的事：这件事接着发展（计划做了、变了、取消了，状况有了新消息，做的时候碰到了问题）、对其中说过的作了更新、修正或推翻、回到这件事上，不管隔多远。连到它接着往下走的那一步（这件事本身，或它下面的某一步）。如果它接着的是本窗口更早的 SB（它是在那里开头的那件事的一部分），就连那个 s。话题相同、用词相近、时间挨着都不是承接。一个 SB 接着推进几件事，就是把它们合到了一起，这时在 why 里写明合的是什么。不接着推进任何早先的事就不连。
人的 SB 报告了某个还在进行中的步骤有了结果（做了、定了、放弃了；助手给的东西，人确认了、照着做了或用上了），写进 settles。问起它、再要一次、或者只是助手这么说，都不是结果。标着"只看 settles"的 SB 在本窗口里已经有位置，只给它们写 settles，不连承接。

只输出紧凑 JSON，字符串里不要用英文双引号：{{"links":[{{"sb":"s3","to":"N12","why":"一句话"}}],"settles":[{{"sb":"s3","step":"N7"}}]}}

本窗口的 SB：
{sbs}

早先的事：
{cands}""",
"en": """You are linking semantic blocks (SBs) of a long-term memory system. Below are the SBs of one window of a conversation (s ids, each one thing, followed by the earlier matters listed for it), and the earlier matters: each is a thing the person brought up (an N marked the person), with the steps that followed it indented under it (N ids, by the person or the assistant); each step is marked happened or in progress at the time.

For each SB, decide which earlier matter it carries on: the same matter going on (a plan carried out, changed or cancelled; news about a situation; a problem met while doing it), an update, correction or reversal of something said in it, or a return to it, however far back. Link it to the step of that matter it goes on from (the matter itself or a step under it). When it goes on from an earlier SB of this window instead (it is part of a thing begun there), link it to that s. Sharing a topic or words, or coming right next in time, is not carrying on. An SB that carries on several matters brings them together: then say in why what it brings together. None if it carries on no earlier matter.
When an SB of the person reports that a step still in progress came to a result (it was done, decided or dropped; for what the assistant gave: the person confirmed it, followed it or put it to use), put it in settles. Asking about it, asking for it again, or the assistant saying so is not a result. SBs marked (settles only) already have their place in this window: give only settles for them, no links.

Return only compact JSON, no double quotes inside strings: {{"links":[{{"sb":"s3","to":"N12","why":"one sentence"}}],"settles":[{{"sb":"s3","step":"N7"}}]}}

SBs of this window:
{sbs}

Earlier matters:
{cands}"""}
STATE = {"zh": {"happened": "已发生", "in progress": "进行中"}, "en": {"happened": "happened", "in progress": "in progress"}}
WHO2 = {"zh": {"person": "人", "assistant": "助手"}, "en": {"person": "the person", "assistant": "the assistant"}}


def step2_prompt(lang: str, sbs: list[dict], cands: list[dict], per_sb: list[list[int]],
                 scene: str = "work") -> str:
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
    word = ("早先的事" if lang == "zh" else "matters") if grouped else ("候选" if lang == "zh" else "candidates")
    only = {"zh": "（只看 settles）", "en": " (settles only)"}[lang]
    lines = [f"s{i + 1}{only if s.get('settles_only') else ''} [{s.get('time', '')}] {s['text']}\n   {word}: "
             + " ".join(f"N{j + 1}" for j in dict.fromkeys(matter[j] for j in per_sb[i]))
             for i, s in enumerate(sbs)]
    if grouped:
        details = []
        for m in dict.fromkeys(matter):
            details.append(detail(m, cands[m]))
            details += ["   " + detail(j, c) for j, c in enumerate(cands) if matter[j] == m and j != m]
    else:
        details = [detail(j, c) for j, c in enumerate(cands)]
    return (STEP2_CHAT if scene == "chat" else STEP2)[lang].format(sbs="\n".join(lines),
                                                                   cands="\n".join(details) or "-")


def parse_step2(data: dict, n_sbs: int, n_cands: int) -> dict:
    """{"after": {i: [("N", j) | ("s", k)]}, "why": {(i, ref): str}, "settles": {i: [j]}} with
    i, k SB indexes (k < i) and j candidate indexes; anything else is dropped."""
    if not isinstance(data, dict) or not isinstance(data.get("links", []), list):
        raise ValueError("'links' must be a list")

    def num(x: object, prefix: str, limit: int) -> int | None:
        s = str(x or "").strip()
        return int(s[1:]) - 1 if s[:1] == prefix and s[1:].isdigit() and 0 < int(s[1:]) <= limit else None
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
        if isinstance(s, dict) and (i := num(s.get("sb"), "s", n_sbs)) is not None \
                and (j := num(s.get("step"), "N", n_cands)) is not None:
            settles.setdefault(i, []).append(j)
    return {"after": after, "why": why, "settles": settles}


def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the output")
    return json.loads(text[start:end + 1])
