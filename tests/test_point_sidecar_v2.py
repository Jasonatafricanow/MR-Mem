import json

import pytest

from mr_mem.point_sidecar_v2 import (
    MAX_SIDECAR_BYTES,
    SIDECAR_TOOL,
    BodyResponseError,
    PointSidecarContext,
    PointSidecarError,
    PointTarget,
    allocate_point_context,
    body_point_sidecar_instruction,
    body_point_sidecar_tool,
    parse_body_frame_v2,
    parse_body_turn_v2,
    parse_point_sidecar_v2,
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
        "points": [
            {
                "slot": 0,
                "meaning": "第一部分改成周五，其余安排沿用前一轮。",
                "context_refs": ["p0"],
                "unresolved_refs": [],
            }
        ],
        **overrides,
    }


def message(arguments=None):
    return {
        "content": "明白，只调整第一部分。",
        "tool_calls": [
            {
                "id": "provider-call",
                "type": "function",
                "function": {
                    "name": SIDECAR_TOOL,
                    "arguments": json.dumps({"response": "明白，只调整第一部分。", **payload()})
                    if arguments is None
                    else arguments,
                },
            }
        ],
    }


def test_same_inference_binds_slot_and_alias_to_host_ids():
    result = parse_body_turn_v2(message(), context=context())
    assert result.response == message()["content"] and result.sidecar_error is None
    point = result.points[0]
    assert (point.point_id, point.interaction_id, point.turn_id) == (
        "session-1/turn-2/p0",
        "session-1",
        "turn-2",
    )
    assert point.meaning == payload()["points"][0]["meaning"]
    assert point.context_refs[0].target_id == "session-1/turn-1/p0"


@pytest.mark.parametrize(
    "field",
    [
        "point_id",
        "scope",
        "source_ref",
        "known_at",
        "occurred_at",
        "revision",
        "turn_id",
        "canonical_id",
        "polarity",
        "context_links",
        "relation",
    ],
)
def test_body_cannot_supply_host_authority_or_extra_fields(field):
    value = payload()
    value["points"][0][field] = "forged"
    with pytest.raises(PointSidecarError):
        parse_point_sidecar_v2(json.dumps(value), context=context())


@pytest.mark.parametrize(
    "changes",
    [
        {"slot": -1},
        {"slot": 2},
        {"slot": True},
        {"slot": 0.0},
        {"slot": "0"},
        {"status": "CLOSED"},
        {"meaning": " "},
        {"meaning": "x" * 2049},
        {"context_refs": ["b0"]},
        {"context_refs": ["session-1/turn-1/p0"]},
        {"context_refs": [{"target": "p0", "target_id": "forged", "relation": "修正"}]},
        {"context_refs": ["p0", "p0"]},
        {"context_refs": ["p0"] * 9},
        {"unresolved_refs": ["未解"] * 9},
    ],
)
def test_invalid_point_rejects_entire_sidecar(changes):
    value = payload()
    value["points"][0].update(changes)
    with pytest.raises(PointSidecarError):
        parse_point_sidecar_v2(json.dumps(value), context=context())


@pytest.mark.parametrize(
    "value",
    [
        payload(schema_version="old"),
        payload(points=[payload()["points"][0]] * 2),
        payload(points=[payload()["points"][0]] * 5),
        payload(points={}),
        payload(response="reply"),
        payload(scope="forged"),
        payload(points=[{**payload()["points"][0], "slot": 1}]),
    ],
)
def test_invalid_envelope_is_not_repaired(value):
    with pytest.raises(PointSidecarError):
        parse_point_sidecar_v2(json.dumps(value), context=context())


@pytest.mark.parametrize(
    "raw",
    [
        '{"points":[],"points":[]}',
        "NaN",
        "```json\n{}",
        "{",
        "\ud800",
        json.dumps(payload(points=[{**payload()["points"][0], "meaning": "\ud800"}])),
    ],
)
def test_bad_envelope_json_never_uses_untrusted_content_as_a_reply(raw):
    with pytest.raises(BodyResponseError):
        parse_body_turn_v2(message(raw), context=context())


@pytest.mark.parametrize("failure", ["missing", "extra", "wrong_name", "length", "no_calls"])
def test_sidecar_channel_failure_keeps_response(failure):
    value = message()
    finish = "tool_calls"
    if failure == "missing":
        value.pop("tool_calls")
    elif failure == "extra":
        value["tool_calls"] *= 2
    elif failure == "wrong_name":
        value["tool_calls"][0]["function"]["name"] = "other_tool"
    elif failure == "length":
        finish = "length"
    else:
        value["tool_calls"] = []
    if failure == "length":
        result = parse_body_turn_v2(value, context=context(), finish_reason=finish)
        assert result.response == value["content"] and result.points == () and result.sidecar_error
    else:
        with pytest.raises(BodyResponseError):
            parse_body_turn_v2(value, context=context(), finish_reason=finish)


@pytest.mark.parametrize("content", [None, "", "\ud800"])
def test_sidecar_never_substitutes_for_missing_response(content):
    value = message(json.dumps({"response": content, **payload()}))
    with pytest.raises(BodyResponseError):
        parse_body_turn_v2(value, context=context())


@pytest.mark.parametrize(
    "suffix",
    [
        "",
        "\n<point_sidecar>{",
        "\n<point_sidecar>{}\n</point_sidecar>trailing",
        "\n<point_sidecar>{}\n<point_sidecar>{}\n</point_sidecar>",
        '\n<point_sidecar>{"points":[],"points":[]}\n</point_sidecar>',
        "\n<point_sidecar>\ud800\n</point_sidecar>",
    ],
)
def test_frame_failure_keeps_response_even_if_sidecar_has_invalid_unicode(suffix):
    result = parse_body_frame_v2("正常回复。" + suffix, context=context())
    assert result.response == "正常回复。" and result.points == () and result.sidecar_error


@pytest.mark.parametrize("closing_space", ["", "\n", " \n"])
def test_valid_frame_maps_host_ids_and_length_rejects_only_sidecar(closing_space):
    raw = "正常回复。\n<point_sidecar>" + json.dumps(payload()) + closing_space + "</point_sidecar>"
    result = parse_body_frame_v2(raw, context=context())
    assert result.points[0].point_id == context().point_ids[0] and not result.sidecar_error
    truncated = parse_body_frame_v2(raw, context=context(), finish_reason="length")
    assert truncated.response == "正常回复。" and truncated.points == ()


def test_defer_preserves_expression_without_guessing_antecedent():
    value = payload()
    value["points"][0].update(
        unresolved_refs=["它"],
        meaning="它需要先加密才能上传；对话尚未确定它指什么。",
    )
    point = parse_point_sidecar_v2(json.dumps(value), context=context())[0]
    assert point.status == "DEFER" and point.unresolved_refs == ("它",)


def test_empty_refs_resolve_and_model_authored_status_is_forbidden():
    assert parse_point_sidecar_v2(json.dumps(payload()), context=context())[0].status == "RESOLVED"
    value = payload()
    value["points"][0]["status"] = "RESOLVED"
    result = parse_body_turn_v2(
        message(json.dumps({"response": "正常回复", **value})), context=context()
    )
    assert result.response == "正常回复" and result.points == () and result.sidecar_error


def test_envelope_owns_response_even_when_content_is_empty_or_conflicting():
    for content in (None, "", "This is not the reply authority"):
        value = {**message(), "content": content}
        result = parse_body_turn_v2(value, context=context())
        assert result.response == "明白，只调整第一部分。" and not result.sidecar_error


def test_aliases_are_typed_request_local_and_never_expose_identities():
    ctx = context(
        activated_targets=(
            PointTarget("POINT", "long-point-id"),
            PointTarget("BLOCK", "long-block-id"),
            PointTarget("POINT", "second-point-id"),
            PointTarget("MEMORY", "long-memory-id"),
        )
    )
    assert list(ctx.target_aliases()) == ["p0", "b0", "p1", "m0"]
    instruction = body_point_sidecar_instruction(ctx)
    wire = instruction + json.dumps(body_point_sidecar_tool(ctx))
    assert all(target.target_id not in wire for target in ctx.activated_targets)
    assert all(point_id not in wire for point_id in ctx.point_ids)
    assert "minimal semantic commitment" in instruction and "Default ONE Point" in instruction
    value = payload()
    value["points"][0]["context_refs"] = ["b0"]
    reference = parse_point_sidecar_v2(json.dumps(value), context=ctx)[0].context_refs[0]
    assert (reference.target_kind, reference.target_id) == ("BLOCK", "long-block-id")


def test_host_identities_remain_replay_stable():
    assert allocate_point_context("a/b", "c") == allocate_point_context("a/b", "c")
    assert (
        allocate_point_context("a/b", "c").point_ids != allocate_point_context("a", "b/c").point_ids
    )


def test_bound_is_on_total_utf8_sidecar():
    values = [{**payload()["points"][0], "slot": i, "meaning": "界" * 2048} for i in range(4)]
    assert len(json.dumps(values, ensure_ascii=False).encode()) > MAX_SIDECAR_BYTES
    with pytest.raises(PointSidecarError):
        parse_point_sidecar_v2(
            json.dumps(payload(points=values)),
            context=context(point_ids=tuple(f"p{i}" for i in range(4))),
        )


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
def test_host_context_is_bounded(overrides):
    with pytest.raises(PointSidecarError):
        context(**overrides)
