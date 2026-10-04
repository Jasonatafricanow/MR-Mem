# P1.5 — meaning-changing ambiguity and minimal historical abstraction

Architecture authority: [ADR-0003](ADR-0003-TURN-POINT-LIVE-WINDOW-BLOCK-COMPILATION.md).
P1.4 removed cross-turn relation compilation from Point. P1.5 changes only the
ambiguity and historical-demonstrative clauses; transport, wire, host-derived
status, weak refs and coverage-gap policy remain unchanged. The original
[P1 NO-GO](POINT_V2_P1_GATE_A.md), [P1.1 NO-GO](POINT_V2_P11_GATE_A.md) and
[P1.2 A/B evidence](POINT_V2_P12_GATE_A.md) remain unchanged.
[P1.3 strict replay](POINT_V2_P13_GATE_A.md) remains historical evidence.
[P1.4 replay](POINT_V2_P14_GATE_A.md) remains historical evidence.
[P1.5 replay](POINT_V2_P15_GATE_A.md) records the current result.
No P2, Block, canonical admission or Thread/LCE work is included.

## Meaning and host-derived status

The semantic policy remains P1.1: `meaning` is minimal semantic commitment of
the current user turn in the existing dialogue. Do not import the Body's own
explanations/advice, world knowledge or unexpressed independence, exclusivity,
causation, permanence, preference or negation. Preserve expressed uncertainty,
scope, time, corrections and partial updates. Default one Point per turn-local
understanding state; multiple only for independently developing semantic lines.

Historical demonstratives such as "the earlier calculation/treatment/plan/conclusion"
stay at the user's expressed level of abstraction. Do not expand such an object
into a specific historical rule or proposition merely because the prior dialogue
suggests one. The current Point records the expressed commitment; reconstructing
the historical object belongs to Block.

The Body emits `slot`, `meaning`, `context_refs` and `unresolved_refs`, **never
`status`**. After structural validation, the host computes:

```python
status = "DEFER" if unresolved_refs else "RESOLVED"
```

This computes a redundant protocol bit; it does not infer, delete, invent or
repair the Body's unresolved content. Model-authored status is an unknown field
and rejects the sidecar. Status is local, not context completeness or canonical
authority.

`unresolved_refs` contains only a current-turn expression with **two or more
semantically viable referents whose different resolutions change the current
Point's proposition**. Unknown specific identity or a generic discourse object
does not qualify when the local commitment can be faithfully expressed at that
level. Missing execution details, causes, further actor information, future
parameters or an unfinished account do not qualify either. For example, a request
to move the referred meeting to the afternoon is locally clear even if its precise
identity is unknown; two viable backup plans for "it" requiring encryption before
uploading produce different propositions and require DEFER. Common sense, world
knowledge or greater plausibility cannot eliminate a viable alternative. Prior
unresolved refs are not inherited unless a current expression still has this
meaning-changing ambiguity. The field name is retained to freeze the wire.

`open_context` is not introduced in this increment; expressed uncertainty belongs
in meaning.
The Gate explicitly measures excessive DEFER as well as false RESOLVED. A host-
derived status can be perfectly consistent while its Body-authored refs are wrong.

`context_refs` is only a list of short aliases for activated objects explicitly
referenced by the current turn. Omit uncertain or merely topical pointers. It
carries no relation, operation, rejected/affirmed-state description or historical
belief reconstruction. Host maps aliases to typed targets; it does not judge the
logical relationship between turns. `meaning` remains the current turn's minimal
commitment, without importing the assistant's explanations.

Correction, qualification, supersession, assistant misunderstanding, retained
scope and semantic closure belong to contextual Block compilation over raw live
turns plus available Points. This refines the Point contract in ADR-0003 and the
implementation plan without changing their turn-local Point/live-window Block
architecture. The old draft wires are not decoded by this revision.

## Strict single-envelope transport

`body_point_sidecar_tool(context)` defines one strict function, `emit_body_turn`,
whose arguments contain both normal reply and Point proposals:

```json
{"response":"normal user-facing reply","points":[{
  "slot":0,
  "meaning":"minimal current-turn commitment",
  "context_refs":["p0"],
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
record_turn_receipt(
    native_session_id, native_turn_id, points=result.points,
    point_coverage_gap=not result.points, rejection=result.sidecar_error,
)
```

MR-Mem supplies no provider client, retry or model settings. Provider capability
must be verified by the host. DeepSeek's existing thinking-enabled Chat and
Responses routes rejected forced tool choice with HTTP 400. The controlled P1.2
A/B uses the same model with thinking disabled **in both arms**, without modifying
the live profile. A nonthinking experiment does not certify the unchanged thinking
route. The exported schema remains provider-neutral; this capability limit is not
silently worked around by auto tools or a mode switch in MR-Mem.

P1.5 reruns only strict + thinking disabled on the same 28 source turns and original
reply prefixes. It does not repeat the frame A/B or choose a production mode.
The host still validates every argument. P1.3 showed a required-field omission
despite strict output; invalid Points are rejected without repair.

Once a valid envelope contains a usable response, Point validation failure returns
that response, zero Points and `sidecar_error`. Record the source-linked Point
coverage gap; a missing Point does not exclude its raw turn from later compilation.
Missing/wrong/multiple functions, unparseable envelope JSON or invalid/missing
response produce `BodyResponseError`.
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
{"points":[{"slot":0,"meaning":"...","context_refs":[],"unresolved_refs":[]}]}
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
Prior proposals sent to the model also use aliases, with no relation or redundant
status. Schema `body_point_sidecar_v2_4` emits `context_refs: ["p0", ...]`; each
decoded ref is a host-bound `PointTarget` with no semantic relation attribute.

- At most four Points, default one; empty only when no semantic content exists.
- Point array at most 12,288 UTF-8 bytes; raw arguments at most 65,536 bytes.
- Meaning at most 2,048 characters; context/unresolved refs at most eight each.
- Ref/host identity text at most 256 characters; activated targets at most 128.
- Normal reply at most 16,384 characters.

Unknown fields/aliases, duplicate keys, invalid Unicode, slot errors and budgets
reject all Points. Strict JSON structure does not certify meaning or correct refs.
The old unmerged P1 through P1.3 wires are intentionally not decoded by this revision;
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

## Coverage gaps

Each probe receipt records the source turn fingerprint and whether no Point was
accepted. Sidecar rejection preserves any usable normal response, admits zero
Points, records the reason and retains the original raw-source prefix. Empty
Point output also records absent Point coverage without inventing semantics.

A schema miss alone is no longer an automatic Gate A architecture blocker.
Incorrect Point admission, lost usable response or untracked raw-source coverage
is a failure. No retry, partial salvage or empty-ref repair is added. The later
Block compiler's authority input is raw live turns; available Points are hints
and cannot be assumed complete.
