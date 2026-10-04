import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest

from examples import point_v2_scratchpad_order_probe as old
from examples.point_v2_contextual_snapshot_probe import (
    freeze_reset_inputs,
    make_request,
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
