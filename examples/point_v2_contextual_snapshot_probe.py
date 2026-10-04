"""P1-R1 experiment: change semantic task only; never production instructions."""

from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

from examples.point_v2_scratchpad_order_probe import (
    CONTEXT_MARKER,
    SEMANTIC_MARKER,
    fingerprint,
    request_bytes,
)
from mr_mem.point_sidecar_v2 import allocate_point_context

BASE_HEAD = "80b907789e225d69c7c9022f5bcf94d6af3f320b"
TASK_B = (
    "Emit a concise snapshot of the contextual understanding you are currently using to "
    "answer this user turn. This understanding may and should use relevant prior dialogue. "
    "Preserve resolved references, actors, scope, distinctions, corrections, qualifications "
    "and prior constraints when they are actually part of how you understand the current "
    "turn. Do not reinterpret the current utterance in isolation. Do not merely summarize "
    "the current sentence, and do not reconstruct unrelated conversation history. Record "
    "what this user turn means in its actual conversational context. If the context still "
    "permits multiple materially different interpretations, preserve that ambiguity in "
    "unresolved_refs. The Point is a turn-time contextual understanding snapshot, not a "
    "context-free proposition and not a whole-dialogue summary. Do not turn your own advice, "
    "new claims or answer into user commitments. Do not invent context or compile a final "
    "whole-topic semantic closure. Default ONE Point for the whole turn-time understanding. "
    "Multiple Points are allowed ONLY for truly independent semantic lines in the same turn "
    "that may develop separately. Unknown specific identity, missing implementation details "
    "or generic discourse objects are not by themselves materially different interpretations. "
    "Do not output status; the host derives it solely from unresolved_refs. Local understanding "
    "is not context-complete or canonical. Use slots 0,1,... in order; the host assigns all "
    "stable identities. context_refs is a list of activated aliases ONLY for context objects "
    "explicitly pointed to by the current turn in its dialogue. If the pointer is uncertain "
    "or merely topically related, leave it out. No relation text, host identity, authority, "
    "extra fields or markdown in tool arguments. At most 4 Points, meaning <=2048 characters, "
    "context/unresolved refs <=8 each, ref <=256 characters, Point array <=12288 UTF-8 bytes. "
    "Empty points only when no semantic content is present. Schema example values below "
    "are placeholders; use the contextual task defined here for meaning."
)


def freeze_reset_inputs(original, pack):
    """Keep original raw prefixes, validation contexts and baseline alias payloads."""
    clauses = {t["instruction_tail"].split(CONTEXT_MARKER, 1)[0] for t in original}
    orders = {t["order_a_clause"] for t in original}
    if len(clauses) != 1 or len(orders) != 1:
        raise ValueError("baseline clauses differ")
    task_a, order = clauses.pop(), orders.pop()
    if not task_a.startswith(SEMANTIC_MARKER) or "reply as ordinary text FIRST" not in order:
        raise ValueError("unexpected P1.5 response-first baseline")
    frozen = deepcopy(original)
    for item in pack:
        context = allocate_point_context(item["id"], "turn-0")
        if item["activated_targets"]:
            raise ValueError("new raw-context probes have no baseline Point hints")
        frozen.append({"case_id": item["id"], "turn_index": 0,
                       "raw_turn": item["current_turn"], "conversation": item["conversation"],
                       "context": asdict(context), "order_a_clause": order,
                       "instruction_tail": task_a + CONTEXT_MARKER
                       + " [0, 1, 2, 3]\nActivated targets: []\nTool arguments: "
                       + '{"points": [{"slot": 0, "meaning": "minimal semantic commitment '
                       + 'explicitly made by the current user turn", "context_refs": [], '
                       + '"unresolved_refs": []}]}\nPrevious Point proposals: []',
                       "conversation_prefix_sha256": fingerprint(item["conversation"])})
    for turn in frozen:
        _, activated = turn["instruction_tail"].split(CONTEXT_MARKER, 1)
        turn["semantic_clauses"] = {"A": task_a, "B": TASK_B}
        turn["activated_payload"] = CONTEXT_MARKER + activated
        turn["activated_context_sha256"] = sha256(turn["activated_payload"].encode()).hexdigest()
    return frozen


def make_request(turn, arm):
    if arm not in ("A", "B"):
        raise ValueError("unknown arm")
    instruction = (turn["order_a_clause"] + turn["semantic_clauses"][arm]
                   + turn["activated_payload"])
    return {"model": "deepseek-flash", "messages": [
        {"role": "system", "content": "Respond helpfully to the user in Chinese."},
        *[dict(m) for m in turn["conversation"]],
        {"role": "system", "content": instruction},
    ], "temperature": .2, "max_tokens": 16384, "thinking": {"type": "enabled"},
        "logprobs": True}


def verify_pair(turn):
    a, b = make_request(turn, "A"), make_request(turn, "B")
    for payload in (a, b):
        if fingerprint(payload["messages"][1:-1]) != turn["conversation_prefix_sha256"]:
            raise ValueError("raw prefix changed")
        if sha256(turn["activated_payload"].encode()).hexdigest() != turn[
            "activated_context_sha256"
        ]:
            raise ValueError("activated payload changed")
        payload["messages"][-1]["content"] = (
            turn["order_a_clause"] + "<SEMANTIC>" + turn["activated_payload"]
        )
    if request_bytes(a) != request_bytes(b):
        raise ValueError("paired inputs differ outside semantic task")
    return sha256(request_bytes(a)).hexdigest()
