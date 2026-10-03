"""Bounded Body output decoding. Points remain proposals; no inference or admission."""

import json
from dataclasses import dataclass
from hashlib import sha256

SCHEMA_VERSION = "body_point_sidecar_v2"
MAX_POINTS = 4
MAX_SIDECAR_BYTES = 12_288


class PointSidecarError(ValueError):
    """The complete sidecar is rejected without repair or model retry."""


@dataclass(frozen=True, slots=True)
class PointTarget:
    target_kind: str
    target_id: str


@dataclass(frozen=True, slots=True)
class PointContextLink:
    target_kind: str
    target_id: str
    relation: str


@dataclass(frozen=True, slots=True)
class PointSidecarContext:
    interaction_id: str
    turn_id: str
    point_ids: tuple[str, ...]
    activated_targets: tuple[PointTarget, ...] = ()

    def __post_init__(self):
        _text(self.interaction_id)
        _text(self.turn_id)
        if not isinstance(self.point_ids, tuple) or not 1 <= len(self.point_ids) <= MAX_POINTS:
            raise PointSidecarError("host must allocate a bounded tuple of Point IDs")
        for point_id in self.point_ids:
            _text(point_id)
        if len(set(self.point_ids)) != len(self.point_ids):
            raise PointSidecarError("duplicate host Point IDs")
        if not isinstance(self.activated_targets, tuple) or len(self.activated_targets) > 128:
            raise PointSidecarError("activated targets must be a bounded host tuple")
        for target in self.activated_targets:
            if not isinstance(target, PointTarget) or target.target_kind not in (
                "POINT",
                "BLOCK",
                "MEMORY",
            ):
                raise PointSidecarError("invalid activated target")
            _text(target.target_id)
            if target.target_id in self.point_ids:
                raise PointSidecarError("current Point IDs cannot be previous context")
        if len(set(self.activated_targets)) != len(self.activated_targets):
            raise PointSidecarError("duplicate activated targets")


@dataclass(frozen=True, slots=True)
class SemanticPointV2:
    point_id: str
    interaction_id: str
    turn_id: str
    meaning: str
    status: str
    context_links: tuple[PointContextLink, ...]
    unresolved_refs: tuple[str, ...]
    polarity: str | None = None
    epistemic_status: str | None = None
    temporal_scope: str | None = None
    temporal_expression: str | None = None


@dataclass(frozen=True, slots=True)
class BodyTurnOutputV2:
    response: str
    points: tuple[SemanticPointV2, ...]


def _text(value, limit=256):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise PointSidecarError("invalid or oversized text")
    _utf8_size(value)
    return value


def _utf8_size(value):
    try:
        return len(value.encode("utf-8"))
    except UnicodeEncodeError as exc:
        raise PointSidecarError("invalid Unicode text") from exc


def _fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional):
        raise PointSidecarError("unknown fields or non-object")
    if set(required) - set(value):
        raise PointSidecarError("missing fields")


def _array(value, limit):
    if not isinstance(value, list) or len(value) > limit:
        raise PointSidecarError("invalid or oversized array")
    return value


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PointSidecarError("duplicate JSON key")
        result[key] = value
    return result


def _point(value, context):
    optional = ("polarity", "epistemic_status", "temporal_scope", "temporal_expression")
    _fields(value, ("point_id", "meaning", "status", "context_links", "unresolved_refs"), optional)
    if _text(value["point_id"]) not in context.point_ids:
        raise PointSidecarError("Point identity was not allocated by the host")
    if value["status"] not in ("RESOLVED", "DEFER"):
        raise PointSidecarError("invalid Point status")
    refs = tuple(_text(ref) for ref in _array(value["unresolved_refs"], 8))
    if refs and value["status"] != "DEFER":
        raise PointSidecarError("unresolved references require DEFER")
    targets = set(context.activated_targets)
    links = []
    for link in _array(value["context_links"], 8):
        _fields(link, ("target_kind", "target_id", "relation"))
        target = PointTarget(_text(link["target_kind"]), _text(link["target_id"]))
        if target not in targets:
            raise PointSidecarError("context target was not activated by the host")
        links.append(
            PointContextLink(target.target_kind, target.target_id, _text(link["relation"]))
        )
    metadata = {name: _text(value[name]) for name in optional if name in value}
    return SemanticPointV2(
        value["point_id"],
        context.interaction_id,
        context.turn_id,
        _text(value["meaning"], 2048),
        value["status"],
        tuple(links),
        refs,
        **metadata,
    )


def parse_body_turn_v2(payload: str, *, context: PointSidecarContext) -> BodyTurnOutputV2:
    """Decode the response and Point siblings from one already completed Body call."""
    if not isinstance(payload, str) or _utf8_size(payload) > 65_536:
        raise PointSidecarError("invalid or oversized Body output")
    try:
        value = json.loads(payload, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except (ValueError, RecursionError) as exc:
        raise PointSidecarError("invalid Body JSON") from exc
    _fields(value, ("schema_version", "response", "points"))
    if value["schema_version"] != SCHEMA_VERSION:
        raise PointSidecarError("unsupported Body sidecar schema")
    response = _text(value["response"], 16_384)
    values = _array(value["points"], MAX_POINTS)
    if _utf8_size(json.dumps(values, ensure_ascii=False)) > MAX_SIDECAR_BYTES:
        raise PointSidecarError("sidecar byte budget exceeded")
    points = tuple(_point(point, context) for point in values)
    if len({point.point_id for point in points}) != len(points):
        raise PointSidecarError("duplicate Point identities")
    return BodyTurnOutputV2(response, points)


def _invalid_constant(value):
    raise PointSidecarError(f"invalid JSON constant: {value}")


def allocate_point_context(
    interaction_id: str,
    turn_id: str,
    *,
    activated_targets: tuple[PointTarget, ...] = (),
) -> PointSidecarContext:
    """Allocate replay-stable local slots. They are not canonical memory identities."""
    point_ids = tuple(
        "point-v2:" + sha256(json.dumps([interaction_id, turn_id, i]).encode()).hexdigest()
        for i in range(MAX_POINTS)
    )
    return PointSidecarContext(interaction_id, turn_id, point_ids, activated_targets)


def body_point_sidecar_instruction(context: PointSidecarContext) -> str:
    """Attach to the normal Body request; the host owns inference and source authority."""
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "response": "your normal user-facing response",
        "points": [
            {
                "point_id": context.point_ids[0],
                "meaning": "faithful turn-local understanding in this conversation",
                "status": "RESOLVED",
                "context_links": [],
                "unresolved_refs": [],
            }
        ],
    }
    activated = [
        {"target_kind": target.target_kind, "target_id": target.target_id}
        for target in context.activated_targets
    ]
    return (
        "In this SAME normal Body inference, return one JSON object with sibling response "
        "and Point sidecar. Do not call another model or tool to produce Points. "
        "Points describe the current turn's local meaning; they may depend on prior turns. "
        "Do not make them artificially context-free or split them into NLP atoms. "
        "Prefer one Point per coherent local understanding; use multiple only for materially "
        "different meanings, never near-duplicate reformulations of the same correction. "
        "Preserve correction, qualification, uncertainty, negation, time, and partial plan "
        "updates. Only include optional polarity, epistemic_status, temporal_scope, or "
        "temporal_expression (nonempty text) when materially relevant. "
        "Use open-text relations for relevant prior context_links, with exactly "
        "target_kind, target_id, relation. Only link to the host's activated targets. "
        "Unresolved references require explicit unresolved_refs and status DEFER; do not guess. "
        "RESOLVED means locally understood, not canonical or complete outside this context. "
        "Never output Scope, SourceRef, revision, time authority, interaction_id, turn_id, "
        "or canonical identity. Points remain proposals awaiting later Block compilation. "
        "Output at most 4 Points, meaning <=2048 characters, links/refs <=8 each, "
        "optional text/relation/ref <=256 characters, sidecar <=12288 UTF-8 bytes. "
        "Use an empty points array only when no semantic content is present. "
        "No extra fields, markdown fences, or prose outside JSON.\n"
        f"Allocated current-turn Point IDs: {json.dumps(context.point_ids)}\n"
        f"Activated prior targets: {json.dumps(activated, ensure_ascii=False)}\n"
        f"Output shape: {json.dumps(envelope, ensure_ascii=False)}"
    )
