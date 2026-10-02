"""Strict validation of current-turn proposals; no inference, IO, or rewriting."""

import json
from dataclasses import asdict

from mr_mem.memory.semantic_contracts import (
    SCHEMA_VERSION,
    BoundaryPolicy,
    DeltaSemanticDependency,
    DeltaSemanticPoint,
    DependencyTargetKind,
    EpistemicStatus,
    LifecycleEffect,
    Polarity,
    SemanticDeltaError,
    SemanticDeltaErrorCode,
    SemanticDeltaV1,
    SemanticPointStatus,
    SpeechAct,
    TemporalScope,
)


def _fail(code: SemanticDeltaErrorCode, detail: str):
    raise SemanticDeltaError(code, detail)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, f"{name} must be nonempty text")
    return value


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "unknown fields or non-object")
    if set(required) - set(value):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "missing fields")


def _enum(cls, value):
    if not isinstance(value, str):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, f"{cls.__name__} must be text")
    try:
        return cls(value)
    except ValueError:
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, f"invalid {cls.__name__}")


def _point(value):
    _fields(
        value,
        (
            "point_id",
            "meaning",
            "status",
            "speech_act",
            "polarity",
            "epistemic_status",
            "temporal_scope",
        ),
        ("temporal_expression", "unresolved_refs"),
    )
    expression = value.get("temporal_expression")
    if expression is not None:
        _text(expression, "temporal_expression")
    refs = value.get("unresolved_refs", ())
    if not isinstance(refs, (list, tuple)):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "unresolved_refs must be an array")
    return DeltaSemanticPoint(
        _text(value["point_id"], "point_id"),
        _text(value["meaning"], "meaning"),
        _enum(SemanticPointStatus, value["status"]),
        _enum(SpeechAct, value["speech_act"]),
        _enum(Polarity, value["polarity"]),
        _enum(EpistemicStatus, value["epistemic_status"]),
        _enum(TemporalScope, value["temporal_scope"]),
        expression,
        tuple(_text(ref, "unresolved ref") for ref in refs),
    )


def _dependency(value):
    _fields(
        value,
        (
            "from_point_id",
            "target_kind",
            "target_id",
            "relation",
            "boundary_policy",
            "lifecycle_effect",
        ),
    )
    return DeltaSemanticDependency(
        _text(value["from_point_id"], "from_point_id"),
        _enum(DependencyTargetKind, value["target_kind"]),
        _text(value["target_id"], "target_id"),
        _text(value["relation"], "relation"),
        _enum(BoundaryPolicy, value["boundary_policy"]),
        _enum(LifecycleEffect, value["lifecycle_effect"]),
    )


def validate_semantic_delta(payload, *, activated_memory_ids: tuple[str, ...]) -> SemanticDeltaV1:
    """Reject the complete proposal on schema, reference, or boundary violations."""
    if not isinstance(activated_memory_ids, tuple):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "activated IDs must be a system tuple")
    for mid in activated_memory_ids:
        _text(mid, "activated memory ID")
    if len(set(activated_memory_ids)) != len(activated_memory_ids):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "duplicate activated memory IDs")
    if isinstance(payload, SemanticDeltaV1):
        payload = asdict(payload)
    if isinstance(payload, str):
        try:
            payload = json.loads(
                payload,
                object_pairs_hook=_object,
                parse_constant=lambda x: _fail(
                    SemanticDeltaErrorCode.INVALID_SCHEMA, f"invalid JSON {x}"
                ),
            )
        except (ValueError, RecursionError) as exc:
            if isinstance(exc, SemanticDeltaError):
                raise
            _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "malformed JSON")
    _fields(payload, ("schema_version", "points", "dependencies"))
    if payload["schema_version"] != SCHEMA_VERSION:
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "unsupported schema version")
    if not all(isinstance(payload[name], (list, tuple)) for name in ("points", "dependencies")):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "points/dependencies must be arrays")
    points = tuple(_point(value) for value in payload["points"])
    dependencies = tuple(_dependency(value) for value in payload["dependencies"])
    by_id = {point.point_id: point for point in points}
    if len(by_id) != len(points):
        _fail(SemanticDeltaErrorCode.INVALID_SCHEMA, "duplicate point IDs")
    for edge in dependencies:
        if edge.from_point_id not in by_id:
            _fail(SemanticDeltaErrorCode.UNKNOWN_POINT, "unknown dependency source")
        if edge.target_kind is DependencyTargetKind.POINT:
            if edge.lifecycle_effect is LifecycleEffect.SUPERSEDE:
                _fail(SemanticDeltaErrorCode.INVALID_DEPENDENCY, "supersede requires memory")
            if edge.target_id not in by_id:
                _fail(SemanticDeltaErrorCode.UNKNOWN_POINT, "unknown dependency target")
            # A resolved cognition cannot borrow meaning from a DEFER point.
            if by_id[edge.from_point_id].eligible != by_id[edge.target_id].eligible:
                _fail(SemanticDeltaErrorCode.INVALID_DEPENDENCY, "unresolved local dependency")
        else:
            if edge.boundary_policy is not BoundaryPolicy.CONTEXT:
                _fail(SemanticDeltaErrorCode.INVALID_DEPENDENCY, "memory requires context")
            if edge.target_id not in activated_memory_ids:
                _fail(SemanticDeltaErrorCode.UNKNOWN_MEMORY, "memory was not activated")
    return SemanticDeltaV1(SCHEMA_VERSION, points, dependencies)
