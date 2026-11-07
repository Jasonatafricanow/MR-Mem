"""A completion backend that speaks the one compile path (docs/LCE_FROZEN.md 2) for the AML
runtime tests: step 1 gives one SB per message in the chat scene (one per message of the person
in the work scene), the stretch layer one open stretch over the window's SBs."""

import json
import re

_NEW = re.compile(r"\nNEW[:：]\n")
_SBS = re.compile(r"\nSBS[:：]\n")
_MESSAGE = re.compile(r"^\[M(\d+)[^\]]*?\s(\S+)\] ?(.*)$")
_PERSON = {"Person", "人"}


class LiveBackend:
    def __init__(self, carry_last: int = 0):
        self.windows: list[list[str]] = []  # the NEW messages' texts of each step-1 call
        self.calls = 0  # every model call, the ones answered "not json" included
        self.stretch_calls = 0
        self.fail_next = 0  # this many next calls answer "not json"
        self.carry_last = carry_last  # chat: stop this many messages before the window's end

    def complete(self, prompt: str) -> str:
        self.calls += 1
        if self.fail_next:
            self.fail_next -= 1
            return "not json"
        if _SBS.search(prompt):
            return self._stretches(prompt)
        messages = []  # (number, role, text)
        for line in _NEW.split(prompt, maxsplit=1)[1].split("\n"):
            m = _MESSAGE.match(line)
            if m:
                role = "user" if m.group(2) in _PERSON else "assistant"
                messages.append((int(m.group(1)), role, m.group(3)))
        self.windows.append([t for _, _, t in messages])
        if '"moves"' in prompt:  # the work scene: the person's moves
            return json.dumps({"moves": self._moves(messages)})
        n = len(messages)
        stop = n - self.carry_last if self.carry_last and n > self.carry_last else n
        sbs = [
            {
                "turns": f"M{k}",
                "text": f"{role}: {text}",
                "when": "before",
                "status": "confirmed",
                "standing": False,
            }
            for k, role, text in messages[:stop]
        ]
        return "ANALYSIS: test\nJSON:\n" + json.dumps(sbs) + f"\nSTOP: M{stop}"

    @staticmethod
    def _moves(messages) -> list[dict]:
        out, about = [], []
        for k, role, text in messages:
            if role != "user":
                about.append(f"M{k}")
                continue
            out.append(
                {"msg": f"M{k}", "about": about, "situation": "-", "move": text, "happened": True}
            )
            about = []
        return out

    def _stretches(self, prompt: str) -> str:
        self.stretch_calls += 1
        body = _SBS.split(prompt, maxsplit=1)[1]
        opened = len(re.findall(r"^o\d+\b", body, re.M))
        sbs = len(re.findall(r"^s\d+\b", body, re.M))
        first = "o1" if opened else "s1"
        return json.dumps(
            {
                # left open, so the next window's SBs carry it on (across windows on the Line)
                "stretches": [{"sbs": f"{first}-s{sbs}", "from": None, "closed": False}],
                "settles": {},
            }
        )
