"""Bounded Body output decoding. Points remain proposals; no inference or admission."""

import json
from dataclasses import dataclass
from hashlib import sha256

SCHEMA_VERSION = "body_point_sidecar_v2_1"
SIDECAR_TOOL = "emit_point_sidecar"
FRAME_START = "\n<point_sidecar>"
FRAME_END = "</point_sidecar>"
MAX_POINTS = 4
MAX_SIDECAR_BYTES = 12_288


class PointSidecarError(ValueError):
    """The complete sidecar is rejected without repair or model retry."""


class BodyResponseError(ValueError):
    """The provider supplied no usable normal response; a sidecar cannot replace it."""


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

    def target_aliases(self) -> dict[str, PointTarget]:
        """Request-local aliases; only the host sees the stable target identities."""
        prefixes = {"POINT": "p", "BLOCK": "b", "MEMORY": "m"}
        counts = dict.fromkeys(prefixes, 0)
        aliases = {}
        for target in self.activated_targets:
            aliases[f"{prefixes[target.target_kind]}{counts[target.target_kind]}"] = target
            counts[target.target_kind] += 1
        return aliases


@dataclass(frozen=True, slots=True)
class SemanticPointV2:
    point_id: str
    interaction_id: str
    turn_id: str
    meaning: str
    status: str
    context_links: tuple[PointContextLink, ...]
    unresolved_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BodyTurnOutputV2:
    response: str
    points: tuple[SemanticPointV2, ...]
    sidecar_error: str | None = None


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
    _fields(value, ("slot", "meaning", "status", "context_links", "unresolved_refs"))
    slot = value["slot"]
    if type(slot) is not int or not 0 <= slot < len(context.point_ids):
        raise PointSidecarError("invalid host Point slot")
    if value["status"] not in ("RESOLVED", "DEFER"):
        raise PointSidecarError("invalid Point status")
    refs = tuple(_text(ref) for ref in _array(value["unresolved_refs"], 8))
    if refs and value["status"] != "DEFER":
        raise PointSidecarError("unresolved references require DEFER")
    targets = context.target_aliases()
    links = []
    for link in _array(value["context_links"], 8):
        _fields(link, ("target", "relation"))
        alias = _text(link["target"])
        if alias not in targets:
            raise PointSidecarError("context target was not activated by the host")
        target = targets[alias]
        links.append(
            PointContextLink(target.target_kind, target.target_id, _text(link["relation"]))
        )
    return SemanticPointV2(
        context.point_ids[slot],
        context.interaction_id,
        context.turn_id,
        _text(value["meaning"], 2048),
        value["status"],
        tuple(links),
        refs,
    )


def parse_point_sidecar_v2(payload: str, *, context: PointSidecarContext):
    """Strictly decode tool arguments. No repair, partial acceptance, or inference."""
    if not isinstance(payload, str) or _utf8_size(payload) > 65_536:
        raise PointSidecarError("invalid or oversized sidecar output")
    try:
        value = json.loads(payload, object_pairs_hook=_object, parse_constant=_invalid_constant)
    except (ValueError, RecursionError) as exc:
        raise PointSidecarError("invalid sidecar JSON") from exc
    _fields(value, ("points",))
    values = _array(value["points"], MAX_POINTS)
    if _utf8_size(json.dumps(values, ensure_ascii=False)) > MAX_SIDECAR_BYTES:
        raise PointSidecarError("sidecar byte budget exceeded")
    points = tuple(_point(point, context) for point in values)
    if [value["slot"] for value in values] != list(range(len(points))):
        raise PointSidecarError("Point slots must be unique and consecutive from zero")
    return points


def parse_body_turn_v2(
    message: dict, *, context: PointSidecarContext, finish_reason: str = "tool_calls"
) -> BodyTurnOutputV2:
    """Keep normal content even when the independent tool sidecar is rejected."""
    try:
        if not isinstance(message, dict):
            raise PointSidecarError("invalid provider message")
        response = _text(message.get("content"), 16_384)
    except PointSidecarError as exc:
        raise BodyResponseError("missing or invalid normal Body response") from exc
    try:
        if finish_reason not in ("stop", "tool_calls"):
            raise PointSidecarError("incomplete Body sidecar")
        calls = _array(message.get("tool_calls"), 1)
        if len(calls) != 1 or not isinstance(calls[0], dict):
            raise PointSidecarError("expected one sidecar tool call")
        call = calls[0]
        function = call.get("function")
        if (
            call.get("type") != "function"
            or not isinstance(function, dict)
            or function.get("name") != SIDECAR_TOOL
        ):
            raise PointSidecarError("unexpected sidecar tool")
        points = parse_point_sidecar_v2(function.get("arguments"), context=context)
        return BodyTurnOutputV2(response, points)
    except PointSidecarError as exc:
        return BodyTurnOutputV2(response, (), str(exc))


def parse_body_frame_v2(
    payload: str, *, context: PointSidecarContext, finish_reason: str = "stop"
) -> BodyTurnOutputV2:
    """Decode a normal text reply followed by an independent, reserved sidecar frame."""
    if not isinstance(payload, str):
        raise BodyResponseError("invalid normal Body response")
    response, separator, frame = payload.partition(FRAME_START)
    try:
        response = _text(response, 16_384)
    except PointSidecarError as exc:
        raise BodyResponseError("missing or invalid normal Body response") from exc
    try:
        if finish_reason != "stop":
            raise PointSidecarError("incomplete Body sidecar")
        if not separator:
            raise PointSidecarError("missing sidecar frame")
        arguments, end, trailing = frame.partition(FRAME_END)
        if not end or trailing.strip() or FRAME_START in frame:
            raise PointSidecarError("invalid sidecar frame boundary")
        points = parse_point_sidecar_v2(arguments, context=context)
        return BodyTurnOutputV2(response, points)
    except PointSidecarError as exc:
        return BodyTurnOutputV2(response, (), str(exc))


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


def body_point_sidecar_instruction(context: PointSidecarContext, *, transport="frame") -> str:
    """Attach to the normal Body request; the host owns inference and source authority."""
    envelope = {
        "points": [
            {
                "slot": 0,
                "meaning": "minimal semantic commitment explicitly made by the current user turn",
                "status": "RESOLVED",
                "context_links": [],
                "unresolved_refs": [],
            }
        ],
    }
    if transport == "tool":
        channel = (
            f"give the normal user-facing reply in message content, then call {SIDECAR_TOOL} "
            "once for the independent Point sidecar. Do not wait for a tool result or another "
            "inference to give the reply. "
        )
    elif transport == "frame":
        channel = (
            "write the normal user-facing reply as ordinary text FIRST, then append an "
            "independent sidecar frame on a new line: <point_sidecar> followed by JSON tool "
            "arguments, then </point_sidecar> on a new line. These markers are reserved; "
            "never use them within the normal reply. No text after the closing marker. "
            "Do not put the normal reply inside JSON. "
        )
    else:
        raise ValueError("unknown Body sidecar transport")
    return (
        "In this SAME normal Body inference, "
        + channel
        + "meaning is minimal semantic commitment: record ONLY what the current user turn "
        "explicitly commits in the existing dialogue. Prior context may identify an explicit "
        "correction or qualification, but must not add a new commitment. Never copy your own "
        "response, explanations, advice or world knowledge into the user's Point. "
        "Do not add unexpressed independence, exclusivity, causation, permanence, preference "
        "or negation. Preserve expressed uncertainty, scope, time and partial updates. "
        "Default ONE Point for the whole turn-local understanding state. Multiple Points "
        "are allowed ONLY for truly independent semantic lines in the same turn that may "
        "develop separately; never split one correction/qualification into NLP atoms. "
        "RESOLVED requires the dialogue itself to determine a unique antecedent. If multiple "
        "antecedents remain semantically possible, or selection relies on common sense/world "
        "knowledge, use DEFER and record the ambiguous expression in unresolved_refs. "
        "Do not decide ambiguity from which antecedent seems more plausible. "
        "RESOLVED is local, not context-complete or canonical. "
        "Use slots 0,1,... in order; the host assigns all stable identities. "
        "context_links use only activated aliases with open-text relation; link relevant "
        "corrections/qualifications without promoting previous proposals into facts. "
        "No host identity, authority, extra fields or markdown in tool arguments. "
        "At most 4 Points, meaning <=2048 characters, links/refs <=8 each, "
        "relation/ref <=256 characters, Point array <=12288 UTF-8 bytes. "
        "Empty points only when no semantic content is present.\n"
        f"Available current slots: {list(range(len(context.point_ids)))}\n"
        f"Activated targets: {list(context.target_aliases())}\n"
        f"Tool arguments: {json.dumps(envelope, ensure_ascii=False)}"
    )


def body_point_sidecar_tool(context: PointSidecarContext) -> dict:
    """Strict function schema; provider support is a host capability, not assumed."""
    link = {
        "type": "object",
        "properties": {"target": {"type": "string"}, "relation": {"type": "string"}},
        "required": ["target", "relation"],
        "additionalProperties": False,
    }
    fields = {
        "slot": {"type": "integer", "minimum": 0, "maximum": len(context.point_ids) - 1},
        "meaning": {"type": "string"},
        "status": {"type": "string", "enum": ["RESOLVED", "DEFER"]},
        "context_links": {"type": "array", "items": link},
        "unresolved_refs": {"type": "array", "items": {"type": "string"}},
    }
    point = {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }
    return {
        "type": "function",
        "function": {
            "name": SIDECAR_TOOL,
            "strict": True,
            "description": "Propose current-turn minimal semantic commitment; no tool execution.",
            "parameters": {
                "type": "object",
                "properties": {"points": {"type": "array", "items": point}},
                "required": ["points"],
                "additionalProperties": False,
            },
        },
    }
