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

ARROW = {"zh": " → 人：", "en": " -> Person: "}
NO_MOVE = {"zh": "（无回应）", "en": "(no response)"}
_CJK = re.compile(r"[一-鿿]")
_WORD = re.compile(r"[A-Za-z]+")


_NOTICE = re.compile(r"^\s*<task-notification>")


def is_noise(role: str, text: str) -> bool:
    """Background notices and the "No response requested." that follows them."""
    text = (text or "").strip()
    return role == "system" or text == "No response requested." or bool(_NOTICE.match(text))


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
        "6. 时间只解析一层：消息带着它说出口的时间（消息时间）。这件事本身说是在相对的时间发生或"
        "要发生（昨天、上周五、上周、下个月、三年之前），就以消息时间为基准，在这几个字旁边写上它"
        '对应的日期或月份："昨天（2022-08-21）赢了比赛""上周（2022-08-07 那周）收到一封信""约 '
        '2023 年（三年之前）我说过十年前去过冰岛"。只写这一个时间，即这件事本身的；这件事内容里'
        "再套着的时间（十年前）原样留着，不往下推算。\n"
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
        "en the thing itself is said to have happened or to be coming at a relative time (yeste"
        "rday, last Friday, last week, next month, three years ago), write the date or month it "
        'works out to from the message time beside the words: "won a tournament yesterday (202'
        '2-08-21)", "a letter last week (week of 2022-08-07)", "about 2023 (three years ago) I'
        ' said I had been to Iceland ten years before". Only that one time, the time of the th'
        "ing; a time inside the thing (ten years before) stays as said, not worked out further"
        ".\n"
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



# STEP1_TOPIC's rule 8 with block numbers (compile_input.blocks; docs/LCE_DESIGN.md 5.5): an SB is
# compiled from these words, so it says which blocks of a message it comes from
TOPIC_BLOCKS_RULE = {
    "zh": (
        '8. turns 写这件事来自原文的哪些块。消息没有分段编号的，写消息（"M3"、"M3-M20"）；'
        '消息分成了编号的段（M3.1、M3.2……）的，只写和这件事有关的那几段（"M3.2-M3.5"，或者 '
        '"M3.1,M3.4"），可以和别的消息连着写（"M3.2-M3.5,M4,M5.1"）；代码段可以写到行'
        '（"M3.4:5-18"）。只有整条消息的每一段都和这件事有关，才写整条。同一条消息的不同段可以'
        "属于不同的 SB。段的内容可能只显示了开头，按开头判断它讲什么。\n"
    ),
    "en": (
        '8. turns: the blocks of the raw text the thing comes from. A message without numbered '
        'paragraphs is written whole ("M3", "M3-M20"); a message shown in numbered paragraphs (M3'
        '.1, M3.2 ...) by only the paragraphs on this thing ("M3.2-M3.5", or "M3.1,M3.4"), which '
        'can run on with other messages ("M3.2-M3.5,M4,M5.1"); a code paragraph can go down to '
        'its lines ("M3.4:5-18"). Write a message whole only when every paragraph of it is on '
        "this thing. Different paragraphs of one message can belong to different SBs. A "
        "paragraph may show only its beginning: judge what it is about from that.\n"
    ),
}
_TOPIC_RULE_8 = re.compile(r"8\. turns.*?\n(?=\n## )", re.S)


def step1_prompt_topic(
    lang: str, new: list[dict], context: list[dict], blocks: bool = False
) -> str:
    """The chat scene's step 1 (STEP1_TOPIC): NEW messages are M1, M2 ... in order. ``blocks``:
    it asks for block numbers."""
    order = list(dict.fromkeys(m.get("session") for m in [*context, *new]))
    tag = (
        (lambda m: f" {SESSION[lang]}{chr(65 + order.index(m.get('session')) % 26)}")
        if len(order) > 1
        else (lambda m: "")
    )
    template = STEP1_TOPIC[lang]
    if blocks:
        template = _TOPIC_RULE_8.sub(lambda _: TOPIC_BLOCKS_RULE[lang], template)
    return template.format(
        cap=ANSWER_SBS,
        context="\n".join(_line(lang, "", m, 600, tag(m)).replace("[ ", "[", 1) for m in context)
        or "-",
        new="\n".join(_numbered(lang, f"M{i + 1}", m, tag(m)) for i, m in enumerate(new)),
    )


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



# --- step 2 ---------------------------------------------------------------------


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



def extract_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in the output")
    return json.loads(text[start : end + 1])
