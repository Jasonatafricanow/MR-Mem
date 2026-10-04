"""Experiment only: frozen hints and P1.5 semantics; vary visible output order."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from hashlib import sha256
from math import comb

from mr_mem.point_sidecar_v2 import (
    FRAME_END,
    FRAME_START,
    BodyResponseError,
    PointSidecarContext,
    PointTarget,
    allocate_point_context,
    body_point_sidecar_instruction,
    parse_body_frame_v2,
)

BASE_HEAD = "7b545f5442da0eb1576735bfb213646feb3fee08"
SEMANTIC_MARKER = "meaning is minimal semantic commitment:"
CONTEXT_MARKER = "\nAvailable current slots:"
ORDER_B = (
    "In this SAME normal Body inference, "
    "write the independent sidecar frame FIRST: <point_sidecar> followed by JSON tool "
    "arguments, then </point_sidecar> on a new line, then write the complete normal "
    "user-facing reply as ordinary text on a new line. These markers are reserved; "
    "never use them within the normal reply. No text before the opening marker. "
    "Do not put the normal reply inside JSON. "
)
CATEGORIES = (
    "BOTH_CORRECT", "POINT_WRONG_RESPONSE_CORRECT", "POINT_CORRECT_RESPONSE_WRONG",
    "BOTH_WRONG_SAME", "BOTH_WRONG_DIFFERENT", "UNCERTAIN",
)


def fingerprint(value):
    return sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def freeze_inputs(cases, baseline):
    """Reconstruct the saved baseline input, never any new arm's output."""
    by_turn = {(r["case_id"], r["turn_index"]): r for r in baseline}
    if len(by_turn) != len(baseline):
        raise ValueError("duplicate baseline turn")
    frozen = []
    for case in cases:
        conversation = list(case.get("history", []))
        replies = case.get("native_assistant_after_turn", case.get("assistant_after_turn"))
        if replies is None or len(replies) != len(case["turns"]):
            raise ValueError("original source replies are required")
        prior = []
        for index, text in enumerate(case["turns"]):
            saved = by_turn[(case["id"], index)]
            conversation.append({"role": "user", "content": text})
            if saved["raw_turn"] != text or saved["conversation_prefix_sha256"] != fingerprint(
                conversation
            ):
                raise ValueError("original prefix differs from baseline")
            context = allocate_point_context(
                case["id"], f"turn-{index}",
                activated_targets=tuple(PointTarget("POINT", p["point_id"]) for p in prior),
            )
            aliases = {
                target.target_id: alias for alias, target in context.target_aliases().items()
            }
            proposals = [
                {"alias": aliases[p["point_id"]], "meaning": p["meaning"],
                 "unresolved_refs": p["unresolved_refs"],
                 "context_refs": [aliases[r["target_id"]] for r in p["context_refs"]]}
                for p in prior
            ]
            instruction = body_point_sidecar_instruction(context, transport="frame")
            instruction += "\nPrevious Point proposals: " + json.dumps(
                proposals, ensure_ascii=False
            )
            order_a, semantic_and_context = instruction.split(SEMANTIC_MARKER, 1)
            semantic, activated = semantic_and_context.split(CONTEXT_MARKER, 1)
            frozen.append({
                "case_id": case["id"], "turn_index": index, "raw_turn": text,
                "conversation": [dict(m) for m in conversation], "context": asdict(context),
                "order_a_clause": order_a,
                "instruction_tail": SEMANTIC_MARKER + semantic_and_context,
                "conversation_prefix_sha256": fingerprint(conversation),
                "activated_context_sha256": sha256(
                    (CONTEXT_MARKER + activated).encode()
                ).hexdigest(),
                "semantic_clause_sha256": sha256(semantic.encode()).hexdigest(),
            })
            conversation.extend(replies[index])
            prior.extend(saved.get("points", []))
    return frozen


def make_request(turn, arm):
    if arm not in ("A", "B"):
        raise ValueError("unknown arm")
    order = turn["order_a_clause"] if arm == "A" else ORDER_B
    instruction = order + turn["instruction_tail"]
    messages = [
        {"role": "system", "content": "Respond helpfully to the user in Chinese."},
        *[dict(m) for m in turn["conversation"]],
        {"role": "system", "content": instruction},
    ]
    return {"model": "deepseek-flash", "messages": messages, "temperature": .2,
            "max_tokens": 16384, "thinking": {"type": "enabled"}, "logprobs": True}


def request_bytes(payload):
    """Hash exactly the UTF-8 bytes that the provider adapter sends."""
    return json.dumps(payload, ensure_ascii=False).encode()


def verify_pair(turn):
    a, b = make_request(turn, "A"), make_request(turn, "B")
    for payload, order in ((a, turn["order_a_clause"]), (b, ORDER_B)):
        instruction = payload["messages"][-1]["content"]
        if not instruction.startswith(order):
            raise ValueError("unexpected order clause")
        common = instruction[len(order):]
        if common != turn["instruction_tail"]:
            raise ValueError("semantic/context payload changed")
        semantic, activated = common.split(SEMANTIC_MARKER, 1)[1].split(CONTEXT_MARKER, 1)
        if (sha256(semantic.encode()).hexdigest() != turn["semantic_clause_sha256"]
            or sha256((CONTEXT_MARKER + activated).encode()).hexdigest()
                != turn["activated_context_sha256"]
            or fingerprint(payload["messages"][1:-1]) != turn["conversation_prefix_sha256"]):
            raise ValueError("frozen input hash changed")
        payload["messages"][-1]["content"] = "<ORDER>" + common
    if request_bytes(a) != request_bytes(b):
        raise ValueError("paired inputs differ outside output-order clause")
    return sha256(request_bytes(a)).hexdigest()


def decode_ordered_frame(raw, context, finish_reason="stop"):
    """Experiment adapter; shared P1.5 validator still decides Point admission."""
    if not isinstance(raw, str):
        raise BodyResponseError("invalid normal Body response")
    if raw.lstrip().startswith("<point_sidecar>"):
        frame = raw.lstrip()[len("<point_sidecar>"):]
        arguments, end, response = frame.rpartition(FRAME_END)
        if not end:
            raise BodyResponseError("first frame has no closing boundary for normal response")
        # Reorder only for the existing validator; do not rewrite JSON or meaning.
        result = parse_body_frame_v2(
            response + FRAME_START + arguments + FRAME_END,
            context=context, finish_reason=finish_reason,
        )
        return result, "POINT_FIRST"
    result = parse_body_frame_v2(raw, context=context, finish_reason=finish_reason)
    return result, "RESPONSE_FIRST" if FRAME_START in raw else "MISSING_FRAME"


def measure_frame_span(raw, usage):
    """Byte trace of frame span only, even when normal response follows it."""
    unknown = {"visible_frame_tokens_range": None}
    trace = usage.get("output_token_trace") or []
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
    completion = usage.get("completion_tokens")
    if not isinstance(raw, str) or not trace or type(reasoning) is not int:
        return unknown
    try:
        chunks = [bytes(t["bytes"]) for t in trace]
    except (KeyError, TypeError, ValueError):
        return unknown
    if b"".join(chunks) != raw.encode() or type(completion) is not int:
        return unknown
    visible = completion - reasoning
    start = raw.find("<point_sidecar>")
    end = raw.find(FRAME_END, start)
    if start < 0 or end < 0 or visible < len(chunks):
        return unknown
    if start and raw[start - 1] == "\n":
        start -= 1
    lo, hi = len(raw[:start].encode()), len(raw[:end + len(FRAME_END)].encode())
    offset = lower = upper = 0
    for chunk in chunks:
        stop = offset + len(chunk)
        lower += int(offset >= lo and stop <= hi)
        upper += int(offset < hi and stop > lo)
        offset = stop
    return {"visible_frame_tokens_range": [lower, upper + visible - len(chunks)],
            "visible_output_tokens": visible, "byte_traced_visible_tokens": len(chunks)}


def exact_paired_test(a_only, b_only):
    """Two-sided exact McNemar test; report counts and direction separately."""
    n = a_only + b_only
    return min(1., 2 * sum(comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2**n)


def paired_endpoint_bounds(pairs):
    """Unknown scores are 0/1 possibilities, never silently scored as success."""
    states = {(0, 0)}
    definite = []
    for a, b in pairs:
        if a not in (0, 1, None) or b not in (0, 1, None):
            raise ValueError("endpoint scores must be zero, one or unknown")
        if a is not None and b is not None:
            definite.append((a, b))
        options_a = (0, 1) if a is None else (a,)
        options_b = (0, 1) if b is None else (b,)
        states = {(x + int(aa == 1 and bb == 0), y + int(bb == 1 and aa == 0))
                  for x, y in states for aa in options_a for bb in options_b}
    a_only = sum(a == 1 and b == 0 for a, b in definite)
    b_only = sum(b == 1 and a == 0 for a, b in definite)
    return {"definite_pairs": len(definite), "uncertain_pairs": len(pairs) - len(definite),
            "a_only": a_only, "b_only": b_only,
            "complete_case_p": exact_paired_test(a_only, b_only),
            "min_a_minus_b": min(a - b for a, b in states),
            "max_a_minus_b": max(a - b for a, b in states),
            "min_p": min(exact_paired_test(a, b) for a, b in states),
            "max_p": max(exact_paired_test(a, b) for a, b in states)}


def run_plan(frozen, plan, infer, output, *, workers=2):
    """Independent frozen pairs, one callback per row; no retries or feedback."""
    if output.exists():
        raise ValueError("refuse to overwrite experiment receipts")
    if workers not in (1, 2):
        raise ValueError("at most two independent paired requests")
    output.parent.mkdir(parents=True, exist_ok=True)

    def one(item):
        turn = frozen[item["input_index"]]
        neutral_hash = verify_pair(turn)
        payload = make_request(turn, item["arm"])
        order = turn["order_a_clause"] if item["arm"] == "A" else ORDER_B
        record = {**item, "case_id": turn["case_id"], "turn_index": turn["turn_index"],
                  "raw_turn": turn["raw_turn"], "model_call_count": 1,
                  "source_turn_sha256": sha256(turn["raw_turn"].encode()).hexdigest(),
                  "conversation_prefix_sha256": turn["conversation_prefix_sha256"],
                  "activated_context_sha256": turn["activated_context_sha256"],
                  "semantic_clause_sha256": turn["semantic_clause_sha256"],
                  "order_clause_sha256": sha256(order.encode()).hexdigest(),
                  "normalized_request_sha256": neutral_hash,
                  "request_body_sha256": sha256(request_bytes(payload)).hexdigest()}
        value = turn["context"]
        context = PointSidecarContext(
            value["interaction_id"], value["turn_id"], tuple(value["point_ids"]),
            tuple(PointTarget(**t) for t in value["activated_targets"]),
        )
        try:
            raw, usage = infer(payload)
            record.update(raw_body_output=raw, provider_usage=usage)
            result, actual = decode_ordered_frame(raw, context, usage.get("finish_reason", "stop"))
            expected = "RESPONSE_FIRST" if item["arm"] == "A" else "POINT_FIRST"
            record.update(response=result.response, points=[asdict(p) for p in result.points],
                          sidecar_error=result.sidecar_error,
                          sidecar_status="REJECTED" if result.sidecar_error else "ACCEPTED",
                          actual_order=actual, order_violation=actual != expected,
                          point_coverage_gap=not bool(result.points))
            record.update(measure_frame_span(raw, usage))
        except (BodyResponseError, RuntimeError) as exc:
            record.update(error=str(exc), points=[], point_coverage_gap=True)
        return record

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0, len(plan), 2):
            futures = [pool.submit(one, item) for item in plan[start:start + 2]]
            for future in futures:
                records.append(future.result())
            output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"completed": len(records), "planned": len(plan)}), flush=True)
    return records
