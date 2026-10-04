import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from examples import point_v2_scratchpad_order_probe as old
from examples.point_v2_contextual_snapshot_probe import (
    freeze_reset_inputs,
    make_request,
    run_plan,
    verify_pair,
)
from tests.test_point_v2_scratchpad_order_probe import fixture_input


def inputs():
    cases, baseline = fixture_input()
    return old.freeze_inputs(cases, baseline)


def test_only_semantic_task_changes_and_a_is_exact_baseline():
    original = inputs()
    frozen = freeze_reset_inputs(original, [])
    assert make_request(frozen[1], "A") == old.make_request(original[1], "A")
    assert verify_pair(frozen[1])
    a, b = (make_request(frozen[1], arm) for arm in ("A", "B"))
    assert a["messages"][:-1] == b["messages"][:-1]
    assert a["messages"][-1]["content"].split(old.CONTEXT_MARKER)[1] == b[
        "messages"
    ][-1]["content"].split(old.CONTEXT_MARKER)[1]
    assert "ordinary text FIRST" in b["messages"][-1]["content"]
    assert "record ONLY what the current user turn explicitly expresses" not in b[
        "messages"
    ][-1]["content"].split(old.CONTEXT_MARKER)[0]


def test_request_mutations_cannot_change_later_frozen_turns():
    frozen = freeze_reset_inputs(inputs(), [])
    saved = deepcopy(frozen)
    request = make_request(frozen[0], "B")
    request["messages"][1]["content"] = "new reply or Point"
    assert frozen == saved
    frozen[1]["conversation"][1]["content"] = "changed prefix"
    with pytest.raises(ValueError, match="prefix changed"):
        verify_pair(frozen[1])


def test_new_gold_binds_34_raw_prefixes_and_pack_requires_context():
    root = Path(__file__).parents[1] / "examples/fixtures"
    gold = json.loads((root / "contextual_snapshot_gold_v1.json").read_text(encoding="utf-8"))
    pack = json.loads((root / "contextual_snapshot_probe_pack_v1.json").read_text(
        encoding="utf-8"
    ))["cases"]
    assert len(gold["entries"]) == 34 and len(pack) == 6
    assert len({entry["id"] for entry in gold["entries"]}) == 34
    index = {entry["id"]: entry for entry in gold["entries"]}
    frozen = freeze_reset_inputs(inputs(), pack)
    for case, turn in zip(pack, frozen[-6:], strict=True):
        entry = index[case["id"] + "/0"]
        assert old.fingerprint(case["conversation"]) == entry["conversation_prefix_sha256"]
        assert sha256(case["current_turn"].encode()).hexdigest() == entry["current_turn_sha256"]
        assert entry["required_axes"] and entry["unresolved_refs"] == "EMPTY"
        assert verify_pair(turn)
        assert "expected_understanding" not in json.dumps(make_request(turn, "B"))


def test_semantic_or_order_mutation_fails_before_inference():
    frozen = freeze_reset_inputs(inputs(), [])
    frozen[0]["semantic_clauses"]["B"] += " added prohibition"
    with pytest.raises(ValueError, match="semantic task changed"):
        verify_pair(frozen[0])
    frozen = freeze_reset_inputs(inputs(), [])
    frozen[0]["order_a_clause"] = "Point first"
    with pytest.raises(ValueError, match="order changed"):
        verify_pair(frozen[0])


def test_one_inference_isolated_failure_and_no_gold_leak_or_feedback(tmp_path):
    frozen = freeze_reset_inputs(inputs(), [])
    saved = deepcopy(frozen)
    gold = {"entries": [{"id": f"{t['case_id']}/{t['turn_index']}",
                          "conversation_prefix_sha256": t["conversation_prefix_sha256"],
                          "current_turn_sha256": sha256(t["raw_turn"].encode()).hexdigest(),
                          "expected_understanding": "private gold never in request"}
                        for t in frozen]}
    plan = [{"input_index":i,"arm":arm,"pair_id":str(i)}
            for i in range(2) for arm in ("A", "B")]
    calls = []

    def infer(payload):
        calls.append(payload)
        assert "private gold" not in json.dumps(payload)
        if payload["messages"][1]["content"] == "first" and len(payload["messages"]) == 3:
            return 'reply\n<point_sidecar>{}</point_sidecar>', {"finish_reason":"stop"}
        return old_test_frame(), {"finish_reason":"stop"}

    output = tmp_path / "receipts.json"
    result = run_plan(frozen, plan, infer, output, gold)
    assert len(calls) == len(result) == 4
    assert frozen == saved
    assert all(r["model_call_count"] == 1 for r in result)
    assert all(r["response"] == "reply" and r["point_coverage_gap"] for r in result[:2])
    assert all(r["points"] for r in result[2:])
    with pytest.raises(ValueError, match="overwrite"):
        run_plan(frozen, plan, infer, output, gold)
    gold["entries"][0]["conversation_prefix_sha256"] = "wrong"
    with pytest.raises(ValueError, match="binding differs"):
        run_plan(frozen, plan, infer, tmp_path / "new.json", gold)
    assert len(calls) == 4


def old_test_frame():
    return ('reply\n<point_sidecar>{"points": [{"slot": 0, "meaning": "snapshot", '
            '"context_refs": [], "unresolved_refs": []}]}</point_sidecar>')
