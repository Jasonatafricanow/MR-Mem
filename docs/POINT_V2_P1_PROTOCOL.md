# P1.2 — host-derived status and single-envelope Body output

Architecture authority: [ADR-0003](ADR-0003-TURN-POINT-LIVE-WINDOW-BLOCK-COMPILATION.md).
P1.2 changes the opt-in transport/contract only. The original
[P1 NO-GO](POINT_V2_P1_GATE_A.md) and [P1.1 NO-GO](POINT_V2_P11_GATE_A.md)
remain unchanged; [P1.2 A/B evidence](POINT_V2_P12_GATE_A.md) records the new result.
No P2, Block, canonical admission or Thread/LCE work is included.

## Meaning and host-derived status

The semantic policy remains P1.1: `meaning` is minimal semantic commitment of
the current user turn in the existing dialogue. Do not import the Body's own
explanations/advice, world knowledge or unexpressed independence, exclusivity,
causation, permanence, preference or negation. Preserve expressed uncertainty,
scope, time, corrections and partial updates. Default one Point per turn-local
understanding state; multiple only for independently developing semantic lines.

The Body emits `slot`, `meaning`, `context_links` and `unresolved_refs`, **never
`status`**. After structural validation, the host computes:

```python
status = "DEFER" if unresolved_refs else "RESOLVED"
```

This computes a redundant protocol bit; it does not infer, delete, invent or
repair the Body's unresolved content. Model-authored status is an unknown field
and rejects the sidecar. References are ambiguous when multiple antecedents
remain viable or choosing one requires common sense/world knowledge. Only a
unique antecedent determined by the dialogue itself is resolved. Status is local,
not context completeness or canonical authority.

`open_context` is not introduced in this increment: no separate nonreferential
deferral contract has been established. Unspecified execution parameters are not
automatically a referent ambiguity, and expressed uncertainty belongs in meaning.
The Gate explicitly measures excessive DEFER as well as false RESOLVED. A host-
derived status can be perfectly consistent while its Body-authored refs are wrong.

## Strict single-envelope transport

`body_point_sidecar_tool(context)` defines one strict function, `emit_body_turn`,
whose arguments contain both normal reply and Point proposals:

```json
{"response":"normal user-facing reply","points":[{
  "slot":0,
  "meaning":"minimal current-turn commitment",
  "context_links":[{"target":"p0","relation":"qualifies the prior proposal"}],
  "unresolved_refs":[]
}]}
```

The host forces exactly that function using the provider's named `tool_choice`;
strict schema alone does not force output. The host never executes a tool, sends
a tool result or makes a second inference. `message.content` may be empty or null:
the reply authority is `tool.arguments.response`.

```python
context = allocate_point_context(native_session_id, native_turn_id)
instruction = body_point_sidecar_instruction(context, transport="tool")
tool = body_point_sidecar_tool(context)
message, usage = existing_body_inference(
    live_messages, instruction, tools=[tool],
    tool_choice={"type":"function","function":{"name":"emit_body_turn"}},
)
result = parse_body_turn_v2(
    message, context=context, finish_reason=usage["finish_reason"],
)
show_response(result.response)
record_rejection(result.sidecar_error) if result.sidecar_error else record_proposals(result.points)
```

MR-Mem supplies no provider client, retry or model settings. Provider capability
must be verified by the host. DeepSeek's existing thinking-enabled Chat and
Responses routes rejected forced tool choice with HTTP 400. The controlled P1.2
A/B uses the same model with thinking disabled **in both arms**, without modifying
the live profile. A nonthinking experiment does not certify the unchanged thinking
route. The exported schema remains provider-neutral; this capability limit is not
silently worked around by auto tools or a mode switch in MR-Mem.

Once a valid envelope contains a usable response, Point validation failure returns
that response, zero Points and `sidecar_error`. Missing/wrong/multiple functions,
unparseable envelope JSON or invalid/missing response produce `BodyResponseError`.
The decoder never rescues response text from untrusted `message.content` or repairs
malformed JSON. Completion must finish with `tool_calls`; a parseable incomplete
envelope preserves any usable response but rejects all Points.

## Frame control retained

The frame route remains available via the default
`body_point_sidecar_instruction(context, transport="frame")` and
`parse_body_frame_v2`. It uses the same status-free Point contract:

```text
Normal reply.
<point_sidecar>
{"points":[{"slot":0,"meaning":"...","context_links":[],"unresolved_refs":[]}]}
</point_sidecar>
```

Reserved markers must not occur within normal response content. The opening marker
begins on a new line; whitespace around argument JSON and before the closing
marker is insignificant. Invalid/duplicate/truncated/missing frames reject the
sidecar and preserve usable preceding content. Missing opening marker means the
whole normal content remains available with a missing-sidecar receipt. This is
bounded framing, not recovery of arbitrary malformed response bytes.

## Identity, bounds and evidence

Current model slots are consecutive unique integers `0,1,...`; the host maps them
to replay-stable Point IDs. Prior targets use typed request-local `p0/p1/b0/m0`
aliases. Stable IDs and host scope/source/time authority never enter model output.
The immutable host context checks eligible aliases without certifying freshness.
Prior proposals sent to the model also use aliases and omit the redundant status.

- At most four Points, default one; empty only when no semantic content exists.
- Point array at most 12,288 UTF-8 bytes; raw arguments at most 65,536 bytes.
- Meaning at most 2,048 characters; links/refs at most eight each.
- Relation/ref/host identity text at most 256 characters; activated targets at most 128.
- Normal reply at most 16,384 characters.

Unknown fields/aliases, duplicate keys, invalid Unicode, slot errors and budgets
reject all Points. Strict JSON structure does not certify meaning or correct refs.
The old unmerged P1/P1.1 wires are intentionally not decoded by this revision;
existing V1 production contracts are unchanged.

`run_probe` supports the selected transport with one callback per turn. For paired
A/B, both arms reuse original P1.1 assistant replies for synthetic history and
original native assistant replies for native history. Equal conversation-prefix
hashes are checked for all 28 turns. The source fixture files and native database
are unchanged; neither arm's generated reply replaces the shared prefix. This
teacher-forced offline replay is not a live gateway integration or deployed hook.

Provider traces must reproduce raw frame bytes exactly to measure Point/frame
tokens. Missing tool traces keep Point-only token attribution unknown. Explicitly
disabled thinking is recorded; completion totals still include normal replies and
cannot be called Point cost. Full native responses and token traces stay private.
