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


def _line(lang: str, label: str, message: dict, limit: int, tag: str = "") -> str:
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
        new="\n".join(_line(lang, f"M{i + 1}", m, 4000, tag(m)) for i, m in enumerate(new)))


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


def tail_start(roles: list[str]) -> int | None:
    """Where an unanswered agent run at the end of the window starts (it goes to the next window,
    where the person's answer will close it); None when the window ends with the person."""
    last_user = max((i for i, r in enumerate(roles) if r == "user"), default=None)
    if last_user is None or last_user == len(roles) - 1:
        return None
    return last_user + 1


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
STATE = {"zh": {"happened": "已发生", "in progress": "进行中"}, "en": {"happened": "happened", "in progress": "in progress"}}


def step2_prompt(lang: str, sbs: list[dict], cands: list[dict], per_sb: list[list[int]]) -> str:
    """sbs: {"text", "time"}; cands: {"text", "time", "state"} (N1.. in this order);
    per_sb[i]: the candidate indexes listed under SB i."""
    word = "候选" if lang == "zh" else "candidates"
    lines = [f"s{i + 1} [{s.get('time', '')}] {s['text']}\n   {word}: " + " ".join(f"N{j + 1}" for j in per_sb[i])
             for i, s in enumerate(sbs)]
    details = [f"N{j + 1} [{c.get('time', '')}{', ' + STATE[lang].get(c.get('state') or '', '') if c.get('state') else ''}] {c['text']}"
               for j, c in enumerate(cands)]
    return STEP2[lang].format(sbs="\n".join(lines), cands="\n".join(details) or "-")


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
