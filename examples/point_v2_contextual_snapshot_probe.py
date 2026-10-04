"""P1-R1 experiment: change semantic task only; never production instructions."""

import json
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256

from examples.point_v2_scratchpad_order_probe import (
    CONTEXT_MARKER,
    SEMANTIC_MARKER,
    decode_ordered_frame,
    fingerprint,
    measure_frame_span,
    request_bytes,
)
from mr_mem.point_sidecar_v2 import (
    BodyResponseError,
    PointSidecarContext,
    PointTarget,
    allocate_point_context,
)

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
                       "raw_turn": item["current_turn"],
                       "conversation": deepcopy(item["conversation"]),
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
        turn["task_sha256"] = {k: sha256(v.encode()).hexdigest()
                               for k, v in turn["semantic_clauses"].items()}
        turn["activated_payload"] = CONTEXT_MARKER + activated
        turn["activated_context_sha256"] = sha256(turn["activated_payload"].encode()).hexdigest()
        turn["order_sha256"] = sha256(turn["order_a_clause"].encode()).hexdigest()
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
    for arm in ("A", "B"):
        if sha256(turn["semantic_clauses"][arm].encode()).hexdigest() != turn["task_sha256"][arm]:
            raise ValueError("frozen semantic task changed")
    if sha256(turn["order_a_clause"].encode()).hexdigest() != turn["order_sha256"]:
        raise ValueError("frozen response-first order changed")
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


def run_plan(frozen, plan, infer, output, gold, *, workers=2):
    """One call per frozen observation; reject source/gold drift before any calls."""
    if output.exists():
        raise ValueError("refuse to overwrite observations")
    if workers not in (1, 2):
        raise ValueError("at most two independent requests")
    index = {entry["id"]: entry for entry in gold["entries"]}
    if len(index) != len(gold["entries"]):
        raise ValueError("duplicate gold entry")
    for item in plan:
        turn = frozen[item["input_index"]]
        key = f"{turn['case_id']}/{turn['turn_index']}"
        entry = index[key]
        if (entry["conversation_prefix_sha256"] != turn["conversation_prefix_sha256"]
            or entry["current_turn_sha256"] != sha256(turn["raw_turn"].encode()).hexdigest()):
            raise ValueError("gold raw-prefix binding differs")
        verify_pair(turn)

    def one(item):
        turn = frozen[item["input_index"]]
        payload = make_request(turn, item["arm"])
        record = {**item, "case_id": turn["case_id"], "turn_index": turn["turn_index"],
                  "raw_turn": turn["raw_turn"], "model_call_count": 1,
                  "conversation_prefix_sha256": turn["conversation_prefix_sha256"],
                  "activated_context_sha256": turn["activated_context_sha256"],
                  "semantic_clause_sha256": turn["task_sha256"][item["arm"]],
                  "order_clause_sha256": turn["order_sha256"],
                  "normalized_request_sha256": verify_pair(turn),
                  "request_body_sha256": sha256(request_bytes(payload)).hexdigest()}
        value = turn["context"]
        context = PointSidecarContext(
            value["interaction_id"], value["turn_id"], tuple(value["point_ids"]),
            tuple(PointTarget(**p) for p in value["activated_targets"]),
        )
        try:
            raw, usage = infer(payload)
            record.update(raw_body_output=raw, provider_usage=usage)
            result, actual = decode_ordered_frame(raw, context, usage.get("finish_reason", "stop"))
            record.update(response=result.response, points=[asdict(p) for p in result.points],
                          sidecar_error=result.sidecar_error,
                          sidecar_status="REJECTED" if result.sidecar_error else "ACCEPTED",
                          actual_order=actual, order_violation=actual != "RESPONSE_FIRST",
                          point_coverage_gap=not bool(result.points))
            record.update(measure_frame_span(raw, usage))
        except (RuntimeError, BodyResponseError) as exc:
            record.update(error=str(exc), points=[], point_coverage_gap=True)
        return record

    output.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(plan), 2):
            futures = [pool.submit(one, item) for item in plan[start:start + 2]]
            records.extend(future.result() for future in futures)
            output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"completed": len(records), "planned": len(plan)}), flush=True)
    return records
