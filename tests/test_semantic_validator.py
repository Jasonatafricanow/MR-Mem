"""Frozen V1 validation boundaries; no provider or store dependencies."""

from copy import deepcopy

import pytest

from mr_mem.memory.semantic_contracts import SemanticDeltaError, SemanticDeltaErrorCode
from mr_mem.memory.semantic_validator import validate_semantic_delta


def point(pid="p1", meaning="用户要求日报只展示异常和结论。", **changes):
    return dict(
        point_id=pid,
        meaning=meaning,
        status="resolved",
        speech_act="directive",
        polarity="positive",
        epistemic_status="asserted",
        temporal_scope="future",
        **changes,
    )


def delta():
    return dict(schema_version="semantic_delta_v1", points=[point()], dependencies=[])


def edge(**changes):
    value = dict(
        from_point_id="p1",
        target_kind="memory",
        target_id="memory-old",
        relation="supersedes",
        boundary_policy="context",
        lifecycle_effect="supersede",
    )
    value.update(changes)
    return value


def test_valid_empty_deferred_and_typed_revalidation():
    payload = delta()
    payload["points"][0].update(status="defer", unresolved_refs=["那个"])
    accepted = validate_semantic_delta(payload, activated_memory_ids=())
    assert accepted.points[0].unresolved_refs == ("那个",)
    assert validate_semantic_delta(accepted, activated_memory_ids=()) == accepted
    assert (
        validate_semantic_delta(
            dict(schema_version="semantic_delta_v1", points=[], dependencies=[]),
            activated_memory_ids=(),
        ).points
        == ()
    )


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda d: d.update(interaction_id="forged"), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (lambda d: d.update(schema_version="other"), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (lambda d: d["points"][0].update(meaning=" "), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (lambda d: d["points"][0].update(status="other"), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (lambda d: d["points"][0].update(memory_id="forged"), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (lambda d: d["points"].append(deepcopy(d["points"][0])), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (
            lambda d: d.update(dependencies=[edge(from_point_id="missing")]),
            "SEMANTIC_DELTA_UNKNOWN_POINT",
        ),
        (
            lambda d: d.update(dependencies=[edge(target_id="not-activated")]),
            "SEMANTIC_DELTA_UNKNOWN_MEMORY",
        ),
        (lambda d: d.update(dependencies=[edge(target_id=123)]), "SEMANTIC_DELTA_INVALID_SCHEMA"),
        (
            lambda d: d.update(dependencies=[edge(boundary_policy="cohabit")]),
            "SEMANTIC_DELTA_INVALID_DEPENDENCY",
        ),
        (
            lambda d: d.update(dependencies=[edge(target_kind="point", target_id="p1")]),
            "SEMANTIC_DELTA_INVALID_DEPENDENCY",
        ),
        (
            lambda d: d.update(dependencies=[edge(boundary_policy="separate")]),
            "SEMANTIC_DELTA_INVALID_SCHEMA",
        ),
        (
            lambda d: d.update(
                dependencies=[edge(target_kind="point", target_id="p2", lifecycle_effect="none")]
            ),
            "SEMANTIC_DELTA_UNKNOWN_POINT",
        ),
    ],
)
def test_invalid_sidecar_is_rejected_whole(mutation, code):
    payload = delta()
    mutation(payload)
    with pytest.raises(SemanticDeltaError) as caught:
        validate_semantic_delta(payload, activated_memory_ids=("memory-old",))
    assert caught.value.code == SemanticDeltaErrorCode(code)


@pytest.mark.parametrize(
    "payload",
    [
        '{"schema_version":"semantic_delta_v1","points":[],"points":[],"dependencies":[]}',
        '{"schema_version":"semantic_delta_v1","points":NaN,"dependencies":[]}',
        "{malformed",
        [],
    ],
)
def test_json_and_shape_fail_closed(payload):
    with pytest.raises(SemanticDeltaError):
        validate_semantic_delta(payload, activated_memory_ids=())


def test_context_target_must_be_activated_and_deferred_dependency_cannot_resolve_neighbor():
    payload = delta()
    payload["dependencies"] = [edge()]
    assert (
        len(validate_semantic_delta(payload, activated_memory_ids=("memory-old",)).dependencies)
        == 1
    )
    payload["points"].append(point("p2", "无法确定所指。"))
    payload["points"][1].update(status="defer", unresolved_refs=["所指"])
    payload["dependencies"] = [
        edge(
            target_kind="point", target_id="p2", boundary_policy="cohabit", lifecycle_effect="none"
        )
    ]
    with pytest.raises(SemanticDeltaError) as caught:
        validate_semantic_delta(payload, activated_memory_ids=())
    assert caught.value.code is SemanticDeltaErrorCode.INVALID_DEPENDENCY
