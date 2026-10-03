"""Host-owned, one-call-per-turn probe; caller supplies its existing normal Body route."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from mr_mem.point_sidecar_v2 import (
    PointSidecarError,
    PointTarget,
    allocate_point_context,
    body_point_sidecar_instruction,
    parse_body_turn_v2,
)


def run_probe(cases, body_infer, output: Path):
    """body_infer(messages) returns (raw JSON Body content, provider usage).

    No retries, repair passes, semantic-only calls, admission, or downstream consumers.
    The full local evidence includes raw turns; publish only a reviewed aggregate.
    """
    records = []
    output.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        conversation = list(case.get("history", []))
        if not 1 <= len(case["turns"]) <= 16 or len(conversation) > 32:
            raise ValueError("probe fixture exceeds turn bound")
        if (
            sum(len(t) for t in case["turns"]) + sum(len(m["content"]) for m in conversation)
            > 32_768
        ):
            raise ValueError("probe fixture exceeds text bound")
        targets = []
        prior_points = []
        for index, user_text in enumerate(case["turns"]):
            context = allocate_point_context(
                case["id"], f"turn-{index}", activated_targets=tuple(targets)
            )
            conversation.append({"role": "user", "content": user_text})
            instruction = body_point_sidecar_instruction(context)
            instruction += "\nPrevious Point proposals: " + json.dumps(
                prior_points, ensure_ascii=False
            )
            messages = [
                {"role": "system", "content": "Respond helpfully to the user in Chinese."},
                *conversation,
                {"role": "system", "content": instruction},
            ]
            record = {
                "case_id": case["id"],
                "turn_index": index,
                "raw_turn": user_text,
                "model_call_count": 1,
                "request_sha256": sha256(json.dumps(messages).encode()).hexdigest(),
            }
            try:
                raw, usage = body_infer(messages)
                record.update(raw_body_output=raw, provider_usage=usage)
                if usage.get("finish_reason", "stop") != "stop":
                    raise RuntimeError("Body output is incomplete")
                result = parse_body_turn_v2(raw, context=context)
                record.update(measure_body_output_tokens(raw, usage))
                record.update(response=result.response, points=[asdict(p) for p in result.points])
                record["sidecar_utf8_bytes"] = len(
                    json.dumps(json.loads(raw)["points"], ensure_ascii=False).encode()
                )
            except (PointSidecarError, RuntimeError) as exc:
                record["error"] = str(exc)
                records.append(record)
                output.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                raise
            records.append(record)
            output.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
            native_replies = case.get("native_assistant_after_turn")
            if native_replies is None:
                conversation.append({"role": "assistant", "content": result.response})
            else:
                # Teacher-forced offline replay preserves the original native conversation.
                # The newly generated response is evidence, not a replacement source turn.
                conversation.extend(native_replies[index])
            for point in result.points:
                targets.append(PointTarget("POINT", point.point_id))
                prior_points.append(asdict(point))
    return records


def measure_body_output_tokens(raw, usage):
    """Attribute visible output using provider bytes; no substitute tokenizer or estimate.

    Overhead includes the Point array and wire envelope. Tokens crossing the response
    value's byte boundary give a lower/upper bound, not fabricated exact attribution.
    Reasoning cost is separate and cannot be isolated without a matched control.
    """
    tokens = usage.get("output_token_trace") or []
    unknown = {"visible_output_tokens": None, "visible_overhead_tokens_range": None}
    if not tokens or any(not isinstance(token.get("bytes"), list) for token in tokens):
        return unknown
    try:
        chunks = [bytes(token["bytes"]) for token in tokens]
    except (TypeError, ValueError):
        return unknown
    raw_bytes = raw.encode("utf-8")
    adjustment = 0
    if b"".join(chunks) != raw_bytes:
        visible = usage.get("completion_tokens", 0) - usage.get(
            "completion_tokens_details", {}
        ).get("reasoning_tokens", 0)
        # Some JSON-mode traces repeat one prefill token. Require both independent
        # byte equality and provider token accounting before excluding that duplicate.
        if (
            len(chunks) == visible
            and chunks[0]
            and raw_bytes.startswith(chunks[0])
            and b"".join(chunks[1:]) == raw_bytes
        ):
            chunks = chunks[1:]
            adjustment = 1
        else:
            return unknown
    decoder = json.JSONDecoder()
    cursor = raw.index("{") + 1
    response_span = None
    while cursor < len(raw):
        while raw[cursor] in " \r\n\t,":
            cursor += 1
        if raw[cursor] == "}":
            break
        key, cursor = decoder.raw_decode(raw, cursor)
        cursor = raw.index(":", cursor) + 1
        while raw[cursor].isspace():
            cursor += 1
        _, end = decoder.raw_decode(raw, cursor)
        if key == "response":
            response_span = (len(raw[:cursor].encode()), len(raw[:end].encode()))
        cursor = end
    if response_span is None:
        return unknown
    start, end = response_span
    offset = response_tokens = boundary_tokens = 0
    for chunk in chunks:
        stop = offset + len(chunk)
        if offset < end and stop > start:
            response_tokens += 1
            boundary_tokens += int(offset < start or stop > end)
        offset = stop
    lower = len(chunks) - response_tokens
    return {
        "visible_output_tokens": len(chunks),
        "visible_overhead_tokens_range": [lower, lower + boundary_tokens],
        "token_trace_prefill_adjustment": adjustment,
        "provider_reported_visible_tokens": usage.get("completion_tokens", 0)
        - usage.get("completion_tokens_details", {}).get("reasoning_tokens", 0),
    }
