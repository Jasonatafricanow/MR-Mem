"""Host-owned, one-call-per-turn probe; caller supplies its existing normal Body route."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from mr_mem.point_sidecar_v2 import (
    BodyResponseError,
    PointTarget,
    allocate_point_context,
    body_point_sidecar_instruction,
    body_point_sidecar_tool,
    parse_body_turn_v2,
)


def run_probe(cases, body_infer, output: Path):
    """body_infer(messages, tool) returns (provider message, provider usage).

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
            aliases = {
                target.target_id: alias for alias, target in context.target_aliases().items()
            }
            proposals = [
                {
                    "alias": aliases[p.point_id],
                    "meaning": p.meaning,
                    "status": p.status,
                    "unresolved_refs": p.unresolved_refs,
                    "context_links": [
                        {"target": aliases[link.target_id], "relation": link.relation}
                        for link in p.context_links
                    ],
                }
                for p in prior_points
            ]
            instruction += "\nPrevious Point proposals: " + json.dumps(
                proposals, ensure_ascii=False
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
                "protocol": "body_point_sidecar_v2_1",
                "request_sha256": sha256(
                    json.dumps([messages, body_point_sidecar_tool(context)]).encode()
                ).hexdigest(),
            }
            try:
                raw, usage = body_infer(messages, body_point_sidecar_tool(context))
                record.update(raw_body_output=raw, provider_usage=usage)
                result = parse_body_turn_v2(
                    raw, context=context, finish_reason=usage.get("finish_reason", "tool_calls")
                )
                record.update(measure_body_output_tokens(raw, usage))
                record.update(response=result.response, points=[asdict(p) for p in result.points])
                record["sidecar_error"] = result.sidecar_error
                record["sidecar_status"] = "REJECTED" if result.sidecar_error else "ACCEPTED"
                record["sidecar_utf8_bytes"] = (
                    None
                    if result.sidecar_error
                    else len(
                        json.dumps(
                            json.loads(raw["tool_calls"][0]["function"]["arguments"])["points"],
                            ensure_ascii=False,
                        ).encode()
                    )
                )
            except (BodyResponseError, RuntimeError) as exc:
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
                prior_points.append(point)
    return records


def measure_body_output_tokens(raw, usage):
    """Attribute visible output using provider bytes; no substitute tokenizer or estimate.

    The provider must expose a complete byte-aligned trace of normal content and
    report separate visible/reasoning counts. The difference includes tool framing.
    Otherwise sidecar attribution stays unknown. Reasoning overhead is not isolated.
    """
    tokens = usage.get("output_token_trace") or []
    unknown = {"visible_output_tokens": None, "visible_overhead_tokens_range": None}
    if not tokens or any(not isinstance(token.get("bytes"), list) for token in tokens):
        return unknown
    try:
        chunks = [bytes(token["bytes"]) for token in tokens]
    except (TypeError, ValueError):
        return unknown
    if not isinstance(raw, dict) or not isinstance(raw.get("content"), str):
        return unknown
    if b"".join(chunks) != raw["content"].encode("utf-8"):
        return unknown
    completion = usage.get("completion_tokens")
    reasoning = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
    if type(completion) is not int or type(reasoning) is not int:
        return unknown
    visible = completion - reasoning
    if reasoning < 0 or visible < len(chunks):
        return unknown
    overhead = visible - len(chunks)
    return {
        "visible_output_tokens": visible,
        "visible_response_trace_tokens": len(chunks),
        "visible_overhead_tokens_range": [overhead, overhead],
        "token_attribution_method": "provider_visible_minus_byte_aligned_content_trace",
    }
