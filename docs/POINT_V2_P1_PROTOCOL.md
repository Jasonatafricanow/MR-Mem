# P1.1 — minimal Point commitment and independent Body sidecar

Architecture authority: [ADR-0003](ADR-0003-TURN-POINT-LIVE-WINDOW-BLOCK-COMPILATION.md).
P1.1 repairs the opt-in P1 producer protocol. The original
[Gate A NO-GO](POINT_V2_P1_GATE_A.md) remains historical evidence.

## Meaning and reference resolution

`meaning` is **minimal semantic commitment**: only what the current user turn
explicitly commits in the existing dialogue. Earlier context may identify a
correction or qualification; it cannot supply a new commitment. The Body's own
response, explanations, advice and world knowledge are not the user's Point.
Do not add unexpressed independence, exclusivity, causation, permanence,
preference or negation. Keep expressed uncertainty, time, scope and partial updates
in `meaning`, without requiring separate metadata fields or a semantic ontology.

Reference resolution is conservative:

- A unique antecedent determined by the dialogue itself permits `RESOLVED`.
- Multiple semantically viable antecedents require `DEFER` and `unresolved_refs`.
- Choosing via common sense or world knowledge requires `DEFER`.

`RESOLVED` is local understanding, not context completeness or canonical admission.
The host validates the protocol; it does not decide semantic fidelity by string
matching. Default **one Point per turn-local understanding state**. Multiple Points
are permitted only for truly independent semantic lines in the same turn that
may develop separately. One correction must not become several NLP atoms.

## Host integration contract

```python
from mr_mem.point_sidecar_v2 import (
    allocate_point_context, body_point_sidecar_instruction,
    parse_body_frame_v2,
)

context = allocate_point_context(native_session_id, native_turn_id)
# Host attaches the frame instruction to its existing normal Body request.
message, usage = existing_body_inference(
    existing_live_messages, body_point_sidecar_instruction(context),
)
result = parse_body_frame_v2(
    message, context=context, finish_reason=usage["finish_reason"],
)
show_response(result.response)
if result.sidecar_error:
    record_rejected_sidecar(result.sidecar_error)  # no repair inference
else:
    record_point_proposals(result.points)  # intermediate, noncanonical
```

The host owns one inference, provider capabilities, request assembly and source
authority. MR-Mem supplies no model client or retry. This remains opt-in with no
production routing, P2 buffer, Block compiler, canonical admission or LCE wiring.

The selected fallback returns ordinary text followed by an independent reserved
frame. Normal text is never JSON-escaped or placed inside the sidecar object:

```text
Normal user-facing response.
<point_sidecar>
{"points":[{
  "slot":0,
  "meaning":"minimal semantic commitment explicitly made by this turn",
  "status":"RESOLVED",
  "context_links":[{"target":"p0","relation":"qualifies the earlier proposal"}],
  "unresolved_refs":[]
}]}
</point_sidecar>
```

Both markers are reserved and must not appear inside normal response content.
The opening marker starts on a new line. Whitespace around argument JSON and
before the closing marker is insignificant; trailing nonwhitespace or repeated
frames reject the sidecar. A broken/missing close cannot swallow the preceding
normal response. If no opening marker exists, content is preserved and the sidecar
is marked missing. This is bounded framing, not recovery of arbitrarily corrupted
response bytes.

The existing Body route was tested first with ordinary tools and beta strict tools.
Both capability samples returned content and `emit_point_sidecar` in one inference,
but the full strict-tool fixture run returned empty content on request eight.
Consequently tool support does not establish reliable simultaneous normal replies.
The full rerun uses frames on the original chat-completions route, without JSON
mode, tools, mode/model changes or a second inference. Tool arguments remain
available as an opt-in transport via `body_point_sidecar_tool`,
`body_point_sidecar_instruction(transport="tool")` and `parse_body_turn_v2` for hosts
that can guarantee simultaneous content. No returned sidecar tool is executed.
The old P1 envelope is intentionally not decoded by this unmerged revision.

The host allocates replay-stable Point IDs from interaction, native turn and slot;
the model sees only integer slots `0,1,...`. Slot numbers must be consecutive and
unique. Historical targets use request-local `p0/p1` (Point), `b0` (Block), and `m0`
(Memory) aliases. The immutable request context maps aliases back to typed stable
IDs after validation. Only the host knows these IDs. Prior proposal projections
also replace their own IDs and linked target IDs with aliases.

The host activates prior targets after its scope/source/lifecycle checks. P1.1
checks membership in that set without certifying canonical freshness. Unknown
aliases, host authority fields, duplicate JSON keys, invalid Unicode/statuses,
slot violations and budgets reject the entire sidecar. Explicit unresolved refs
require `DEFER`. Rejected sidecars produce no Points and cannot contaminate the
next request's Point proposals. Their normal response remains available and the
probe continues with that response. No partial acceptance, repair or retry occurs.

`finish_reason=stop` is not proof of valid arguments. Frames require `stop`; the
tool transport accepts `stop` or `tool_calls`. Other finish reasons reject the
sidecar while preserving any usable content. If normal content is missing or invalid, the host gets
`BodyResponseError`; sidecar arguments never become a substitute user reply.

## Bounds and evidence

- At most four Points; default one. Empty only when no semantic content is present.
- Point array at most 12,288 UTF-8 bytes; raw arguments at most 65,536 bytes.
- Meaning at most 2,048 characters; links/refs at most eight each.
- Relation/ref and host ID text at most 256 characters; activated targets at most 128.
- Normal response at most 16,384 characters.

The optional strict provider schema describes structure. Host checks remain necessary for
unsupported schema bounds, identities and cross-field consistency. Strict JSON
does not certify meaning, antecedents or independent semantic lines.

The seven frozen synthetic fixtures and the private frozen native replay remain
unchanged. `examples/point_v2_probe.py` calls `body_infer(messages, tool_or_none)` once per
turn (`None` for frames). Native teacher-forced replay keeps original assistant turns as subsequent
source history; this is offline replay, not a deployed Hermes gateway integration.
An empty generated response remains a failed receipt; native replay can continue
with the original source replies to cover later turns without inventing content.
For ordinary synthetic conversations, a missing reply stops the probe.
Evidence records raw output, response, host-bound Points,
separate sidecar rejection, usage, wire bytes and request hashes. Full native
evidence stays private; only reviewed aggregates may be published.

Frame overhead is attributed only when provider token bytes exactly reproduce
the raw output and provider completion/reasoning counts are available. Boundary
tokens give a range. Provider-visible tokens absent from that exact byte trace are
recorded separately and widen the upper bound; they are never called measured
sidecar text. Otherwise attribution remains unknown; wire
bytes are not substituted for tokens. Reasoning overhead is not isolated without
a matched control. Gate A still requires reviewed semantics and native evidence;
unit tests and strict syntax alone do not open P2.
