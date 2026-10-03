import json

import pytest

from mr_mem.point_sidecar_v2 import (
    MAX_SIDECAR_BYTES,
    PointSidecarContext,
    PointSidecarError,
    PointTarget,
    allocate_point_context,
    body_point_sidecar_instruction,
    parse_body_turn_v2,
)


def context(**overrides):
    return PointSidecarContext(
        **{
            "interaction_id": "session-1",
            "turn_id": "turn-2",
            "point_ids": ("session-1/turn-2/p0", "session-1/turn-2/p1"),
            "activated_targets": (PointTarget("POINT", "session-1/turn-1/p0"),),
            **overrides,
        }
    )


def payload(**overrides):
    return {
        "schema_version": "body_point_sidecar_v2",
        "response": "明白，只调整第一部分。",
        "points": [
            {
                "point_id": "session-1/turn-2/p0",
                "meaning": "第一部分改成周五，其余安排沿用前一轮。",
                "status": "RESOLVED",
                "context_links": [
                    {
                        "target_kind": "POINT",
                        "target_id": "session-1/turn-1/p0",
                        "relation": "仅修改第一部分，保留其余安排",
                    }
                ],
                "unresolved_refs": [],
            }
        ],
        **overrides,
    }


def test_same_body_output_keeps_local_meaning_and_binds_host_ids():
    result = parse_body_turn_v2(json.dumps(payload()), context=context())
    assert result.response == payload()["response"]
    point = result.points[0]
    assert point.meaning == payload()["points"][0]["meaning"]
    assert (point.interaction_id, point.turn_id) == ("session-1", "turn-2")
    assert point.context_links[0].relation == "仅修改第一部分，保留其余安排"
    assert point.polarity is None


@pytest.mark.parametrize(
    "field",
    ["scope", "source_ref", "known_at", "occurred_at", "revision", "turn_id", "canonical_id"],
)
def test_body_cannot_supply_host_authority(field):
    value = payload()
    value["points"][0][field] = "forged"
    with pytest.raises(PointSidecarError):
        parse_body_turn_v2(json.dumps(value), context=context())


@pytest.mark.parametrize(
    "changes",
    [
        {"point_id": "invented"},
        {"status": "CLOSED"},
        {"meaning": " "},
        {"meaning": "x" * 2049},
        {"context_links": [{"target_kind": "POINT", "target_id": "unknown", "relation": "修正"}]},
        {
            "context_links": [
                {"target_kind": "BLOCK", "target_id": "session-1/turn-1/p0", "relation": "修正"}
            ]
        },
        {"unresolved_refs": ["他指的是谁"]},
        {"unresolved_refs": ["未解"] * 9},
        {"epistemic_status": None},
        {"temporal_scope": True},
    ],
)
def test_invalid_semantics_reject_entire_sidecar(changes):
    value = payload()
    value["points"][0].update(changes)
    with pytest.raises(PointSidecarError):
        parse_body_turn_v2(json.dumps(value), context=context())


@pytest.mark.parametrize(
    "value",
    [
        payload(schema_version="semantic_delta_v1"),
        payload(points=[payload()["points"][0]] * 2),
        payload(points=[payload()["points"][0]] * 5),
        payload(points={}),
        payload(response=""),
        payload(scope="forged"),
    ],
)
def test_invalid_envelope_is_not_repaired(value):
    with pytest.raises(PointSidecarError):
        parse_body_turn_v2(json.dumps(value), context=context())


@pytest.mark.parametrize("value", ['{"response":"one","response":"two"}', "NaN", "```json\n{}"])
def test_strict_json(value):
    with pytest.raises(PointSidecarError):
        parse_body_turn_v2(value, context=context())


def test_defer_preserves_unresolved_reference_and_material_qualifiers():
    value = payload()
    value["points"][0].update(
        status="DEFER",
        unresolved_refs=["他"],
        polarity="仅否定基础设施自营",
        epistemic_status="尚待核实",
        temporal_scope="未来",
        temporal_expression="下个月",
    )
    point = parse_body_turn_v2(json.dumps(value), context=context()).points[0]
    assert point.status == "DEFER" and point.unresolved_refs == ("他",)
    assert point.epistemic_status == "尚待核实" and point.temporal_expression == "下个月"


def test_host_allocates_replay_stable_and_unambiguous_identities():
    assert allocate_point_context("a/b", "c") == allocate_point_context("a/b", "c")
    assert (
        allocate_point_context("a/b", "c").point_ids != allocate_point_context("a", "b/c").point_ids
    )
    instruction = body_point_sidecar_instruction(context())
    assert context().point_ids[0] in instruction
    assert "SAME normal Body inference" in instruction


def test_bound_is_on_total_utf8_sidecar_and_not_only_point_count():
    values = []
    for i in range(4):
        point = {**payload()["points"][0], "point_id": f"p{i}", "meaning": "界" * 2048}
        values.append(point)
    assert len(json.dumps(values, ensure_ascii=False).encode()) > MAX_SIDECAR_BYTES
    with pytest.raises(PointSidecarError):
        parse_body_turn_v2(
            json.dumps(payload(points=values)),
            context=context(point_ids=tuple(f"p{i}" for i in range(4))),
        )


@pytest.mark.parametrize("value", ["\ud800", json.dumps(payload(response="\ud800"))])
def test_invalid_unicode_is_a_protocol_rejection(value):
    with pytest.raises(PointSidecarError):
        parse_body_turn_v2(value, context=context())


@pytest.mark.parametrize(
    "overrides",
    [
        {"point_ids": ()},
        {"point_ids": ("p", "p")},
        {"point_ids": ["p"]},
        {"activated_targets": (PointTarget("UNKNOWN", "p"),)},
        {"activated_targets": (PointTarget("POINT", "session-1/turn-2/p0"),)},
    ],
)
def test_host_context_is_bounded_and_cannot_activate_current_slots(overrides):
    with pytest.raises(PointSidecarError):
        context(**overrides)
