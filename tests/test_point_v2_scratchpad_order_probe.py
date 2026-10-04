import json
from copy import deepcopy
from hashlib import sha256

import pytest

from examples.point_v2_scratchpad_order_probe import (
    decode_ordered_frame,
    exact_paired_test,
    freeze_inputs,
    make_request,
    measure_frame_span,
    paired_endpoint_bounds,
    request_bytes,
    run_plan,
    verify_pair,
)
from mr_mem.point_sidecar_v2 import BodyResponseError, allocate_point_context


def fixture_input():
    cases = [{"id": "frozen", "turns": ["first", "later"],
              "assistant_after_turn": [[{"role": "assistant", "content": "original"}], []]}]
    prefixes = [[{"role": "user", "content": "first"}],
                [{"role": "user", "content": "first"},
                 {"role": "assistant", "content": "original"},
                 {"role": "user", "content": "later"}]]
    rows = [{"case_id": "frozen", "turn_index": i, "raw_turn": cases[0]["turns"][i],
             "conversation_prefix_sha256": sha256(
                 json.dumps(p, ensure_ascii=False).encode()
             ).hexdigest(), "points": []} for i, p in enumerate(prefixes)]
    return cases, rows


def test_frozen_requests_never_feed_generated_outputs_back_into_other_turns():
    cases, rows = fixture_input()
    rows[0]["points"] = [{"point_id": "saved-point", "meaning": "baseline hint",
                          "context_refs": [], "unresolved_refs": []}]
    frozen = freeze_inputs(cases, rows)
    snapshot = deepcopy(frozen)
    a, b = make_request(frozen[1], "A"), make_request(frozen[1], "B")
    assert a["messages"][1:-1] == b["messages"][1:-1] == snapshot[1]["conversation"]
    assert '"meaning": "baseline hint"' in a["messages"][-1]["content"]
    assert "saved-point" not in json.dumps(a["messages"])
    assert verify_pair(frozen[1])
    a["messages"][-1]["content"] = "different generated proposal"
    a["messages"][1]["content"] = "mutated request prefix"
    assert frozen == snapshot
    assert make_request(frozen[1], "A")["messages"][-1] == make_request(
        frozen[1], "A"
    )["messages"][-1]
    assert request_bytes(make_request(frozen[1], "A")) != request_bytes(b)


def test_changed_native_prefix_fails_before_any_request():
    cases, rows = fixture_input()
    cases[0]["assistant_after_turn"][0][0]["content"] = "replacement generated reply"
    with pytest.raises(ValueError, match="prefix differs"):
        freeze_inputs(cases, rows)


def framed(first=True, bad=False):
    point = {"slot": 0, "meaning": "local meaning", "context_refs": [], "unresolved_refs": []}
    if bad:
        point.pop("unresolved_refs")
    frame = "<point_sidecar>" + json.dumps({"points": [point]}) + "</point_sidecar>"
    return frame + "\nNormal reply" if first else "Normal reply\n" + frame


@pytest.mark.parametrize("first", [True, False])
@pytest.mark.parametrize("bad", [True, False])
def test_both_orders_hide_frame_and_keep_response_after_rejected_points(first, bad):
    result, actual = decode_ordered_frame(framed(first, bad), allocate_point_context("s", "t"))
    assert result.response.strip() == "Normal reply"
    assert "point_sidecar" not in result.response and "local meaning" not in result.response
    assert bool(result.points) != bad and bool(result.sidecar_error) == bad
    assert actual == ("POINT_FIRST" if first else "RESPONSE_FIRST")


def test_missing_frame_keeps_response_but_unclosed_first_frame_cannot_invent_one():
    context = allocate_point_context("s", "t")
    result, actual = decode_ordered_frame("Normal reply", context)
    assert result.response == "Normal reply" and not result.points
    assert actual == "MISSING_FRAME" and result.sidecar_error == "missing sidecar frame"
    with pytest.raises(BodyResponseError, match="closing boundary"):
        decode_ordered_frame('<point_sidecar>{"points":[]}', context)


def test_closing_marker_inside_rejected_first_frame_never_leaks_point_prose():
    point = {"slot": 0, "meaning": "private </point_sidecar> hidden commitment",
             "context_refs": [], "unresolved_refs": []}
    raw = '<point_sidecar>' + json.dumps({"points": [point]})
    raw += '</point_sidecar>\nNormal reply'
    result, _ = decode_ordered_frame(raw, allocate_point_context("s", "t"))
    assert result.response.strip() == "Normal reply" and not result.points
    assert result.sidecar_error and "hidden commitment" not in result.response


def test_first_frame_overhead_does_not_count_following_normal_reply():
    raw = framed()
    trace = [{"bytes": [b]} for b in raw.encode()]
    usage = {"output_token_trace": trace, "completion_tokens": len(trace) + 9,
             "completion_tokens_details": {"reasoning_tokens": 9}}
    measured = measure_frame_span(raw, usage)
    assert measured["visible_frame_tokens_range"] == [raw.index("\n"), raw.index("\n")]
    usage["output_token_trace"][0]["bytes"] = [0]
    assert measure_frame_span(raw, usage)["visible_frame_tokens_range"] is None


def test_paired_plan_keeps_all_failed_samples_without_retry_or_hint_feedback(tmp_path):
    cases, rows = fixture_input()
    frozen = freeze_inputs(cases, rows)
    snapshot = deepcopy(frozen)
    plan = [{"input_index": i, "arm": arm, "stage": "full", "repeat": 0}
            for i in range(2) for arm in ("A", "B")]
    calls = []

    def infer(payload):
        calls.append(payload)
        is_a = "reply as ordinary text FIRST" in payload["messages"][-1]["content"]
        return framed(first=not is_a, bad=not is_a), {"finish_reason": "stop"}

    path = tmp_path / "receipts.json"
    records = run_plan(frozen, plan, infer, path)
    assert len(calls) == len(records) == 4 and frozen == snapshot
    assert all(r["response"].strip() == "Normal reply" for r in records)
    assert [r["point_coverage_gap"] for r in records] == [False, True, False, True]
    for a, b in (records[:2], records[2:]):
        for key in ("conversation_prefix_sha256", "activated_context_sha256",
                    "semantic_clause_sha256", "normalized_request_sha256"):
            assert a[key] == b[key]
        assert a["order_clause_sha256"] != b["order_clause_sha256"]
    with pytest.raises(ValueError, match="overwrite"):
        run_plan(frozen, plan, infer, path)


def test_wrong_requested_order_is_disclosed_without_hiding_usable_reply(tmp_path):
    frozen = freeze_inputs(*fixture_input())
    records = run_plan(frozen, [{"input_index": 0, "arm": "B"}],
                       lambda _: (framed(False), {}), tmp_path / "wrong.json")
    assert records[0]["order_violation"]
    assert records[0]["actual_order"] == "RESPONSE_FIRST"
    assert records[0]["response"] == "Normal reply" and records[0]["points"]


def test_exact_paired_probability_known_discordant_counts():
    assert exact_paired_test(0, 0) == 1
    assert exact_paired_test(8, 0) == 1 / 128
    assert exact_paired_test(4, 4) == 1


def test_uncertainty_can_remove_apparent_significance_and_must_not_be_imputed():
    known = paired_endpoint_bounds([(1, 0)] * 8)
    assert known["max_p"] == 1 / 128 and known["min_a_minus_b"] == 8
    unknown = paired_endpoint_bounds([(1, 0)] * 8 + [(None, 1)] * 8)
    assert unknown["complete_case_p"] == 1 / 128
    assert unknown["uncertain_pairs"] == 8 and unknown["min_a_minus_b"] == 0
    assert unknown["max_p"] == 1
