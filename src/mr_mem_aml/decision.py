"""Readout-time bypass on a decision model (Bailian System One, ``decision-model-preview``).

Memory stays honest recall (conflicts are not a compile product): this is a per-question
judgement made at Search, written as one note in front of the readout that says only what the
messages found say, and decides nothing. Two rules, fixed before answering and measured on the
holdout BEAM 100k 60 questions, two passes (2026-10-31):

- contradiction: a question the model calls "yesno" (has the person done / had it), where one of
  the person's messages among the raw hits says it did (affirm >= 0.5) and another says it never
  did (deny >= 0.5), gets a note quoting both. 4 of 6 contradiction questions 0 -> 0.75-0.875,
  none other touched: 50.3 -> 55.4.
- absence: a question it calls "specific" (some detail of the person's own), where it judges the
  detail asked for is not stated by the person in the hits (only in the assistant's suggestions,
  or nowhere), gets a note saying so. 14 questions noted (4 abstention): +2.1.

Any failure (the endpoint down, rate limits past the retries) leaves the readout as it is.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

log = logging.getLogger(__name__)

KINDS = {
    "yesno": "问人自己有没有做过、有没有过某件事（回答是或否）",
    "specific": "问人自己的某个具体信息：是什么、有哪些、为什么、怎么做的",
    "other": "让助手给建议或写东西、总结、列举顺序、计算时间间隔、问最新状态等",
}
SAID = {
    "affirm": {
        "type": "noul",
        "instructions": "在这条消息里，说话的人（Person/User，不是 Assistant）是否明确说过："
        "问题所问的那件事已经做过、发生过或成立？"
        "人只是在提问、打算或请求帮助，不算；助手说的不算。",
    },
    "deny": {
        "type": "noul",
        "instructions": "在这条消息里，说话的人（Person/User，不是 Assistant）是否明确说过："
        "问题所问的那件事从没做过、没有发生或不成立？",
    },
}
WHERE = {
    "person": "问题要的那个具体信息，人（User）自己在消息里说过",
    "assistant_only": "这个具体信息只在助手（Assistant）的建议、方案或示例里出现过，"
    "人没有说过自己用了、做了或定了",
    "absent": "这些消息里根本没有问题要的那个具体信息（话题可能聊过，但没有这个信息）",
}
CONTRA = {
    "en": (
        "[Note from the memory search: on what the question asks, the person's own messages"
        " disagree]\n- One message of the person's says it did happen: {a}\n- Another says it"
        " never did: {d}"
    ),
    "zh": "【记忆检索的说明：关于问题所问的事，人自己的说法不一致】\n- 一条消息说做过：{a}\n"
    "- 另一条说从没做过：{d}",
}
ABSENT = {
    "en": (
        "[Note from the memory search: in the messages found, the person never states what the"
        " question asks for; {why}]"
    ),
    "zh": "【记忆检索的说明：找到的消息里，人自己从没说过问题要的这个信息；{why}】",
}
WHY = {
    "en": {
        "assistant_only": "it appears only in the assistant's suggestions or examples, not as"
        " something the person said they did, used or decided",
        "absent": "it does not appear in them",
    },
    "zh": {
        "assistant_only": "它只出现在助手的建议或示例里，人没有说过自己用了、做了或定了",
        "absent": "这些消息里没有它",
    },
}


@dataclass(frozen=True)
class Message:
    role: str  # "user" | "assistant"
    content: str


_SLOTS = threading.Semaphore(16)  # System One takes about 20 calls a second per workspace


class DecisionBypass:
    """``config``: {"base_url" or "base_url_env": an OpenAI-compatible base that serves System One
    (the Token Plan base, or a Bailian workspace's; not the public dashscope base), "api_key_env",
    "model", "timeout", "workers", "hits"}; the endpoint is that base + "/systemone"."""

    def __init__(self, config: dict) -> None:
        base = config.get("base_url") or os.environ.get(config.get("base_url_env", ""), "")
        if not base:
            raise ValueError("decision: no base_url (or its base_url_env is unset)")
        self.url = base.rstrip("/") + "/systemone"
        self.key_env = config.get("api_key_env", "DASHSCOPE_API_KEY")
        self.model = config.get("model", "decision-model-preview")
        self.timeout = float(config.get("timeout", 30))
        self.workers = int(config.get("workers", 16))
        self.hits = int(config.get("hits", 30))  # the raw hits judged, in search order

    def _ask(self, state: dict, questions: dict) -> dict:
        body = json.dumps({"model": self.model, "state": state, "questions": questions}).encode()
        key = os.environ.get(self.key_env, "")
        for attempt in range(5):
            request = urllib.request.Request(
                self.url,
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer " + key},
            )
            try:
                with _SLOTS, urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())["answers"]
            except urllib.error.HTTPError as exc:
                if exc.code != 429 and exc.code < 500:
                    raise
            except OSError:
                pass
            time.sleep(min(8, 2**attempt))
        raise RuntimeError("decision model: gave up after retries")

    def note(self, question: str, hits: list[Message], lang: str = "en") -> str | None:
        """The note for this question over its raw hits (search order), or None."""
        lang = lang if lang in CONTRA else "en"
        hits = hits[: self.hits]
        if not hits:
            return None
        try:
            kind = self._ask(
                {"question": question},
                {
                    "kind": {
                        "type": "choice",
                        "instructions": "这个问题属于哪一类？",
                        "criteria": KINDS,
                    }
                },
            )["kind"]["choice"]
            if kind == "yesno":
                return self._contradiction(question, hits, lang)
            if kind == "specific":
                return self._absence(question, hits, lang)
        except Exception as exc:  # noqa: BLE001 - a bypass: the readout stands without it
            log.warning("decision bypass skipped: %s", str(exc)[:200])
        return None

    @staticmethod
    def _shown(m: Message) -> str:
        return f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"

    def _contradiction(self, question: str, hits: list[Message], lang: str) -> str | None:
        # the person's messages only: the model sometimes takes the assistant's words for theirs
        mine = [m for m in hits if m.role == "user"]
        if len(mine) < 2:
            return None
        with ThreadPoolExecutor(self.workers) as pool:
            scores = list(
                pool.map(
                    lambda m: self._ask(
                        {"question": question, "message": self._shown(m)[:6000]}, SAID
                    ),
                    mine,
                )
            )
        a = max(range(len(mine)), key=lambda i: scores[i]["affirm"]["noul"])
        d = max(range(len(mine)), key=lambda i: scores[i]["deny"]["noul"])
        if a == d or scores[a]["affirm"]["noul"] < 0.5 or scores[d]["deny"]["noul"] < 0.5:
            return None

        def quote(i: int) -> str:
            return json.dumps(self._shown(mine[i])[:3000], ensure_ascii=False)

        return CONTRA[lang].format(a=quote(a), d=quote(d))

    def _absence(self, question: str, hits: list[Message], lang: str) -> str | None:
        messages = "\n\n".join(self._shown(m)[:500] for m in hits)[:15000]
        where = self._ask(
            {"question": question, "messages": messages},
            {
                "src": {
                    "type": "choice",
                    "instructions": "问题要的具体信息在这些对话消息里是什么情况？",
                    "criteria": WHERE,
                }
            },
        )["src"]["choice"]
        return None if where not in WHY[lang] else ABSENT[lang].format(why=WHY[lang][where])
