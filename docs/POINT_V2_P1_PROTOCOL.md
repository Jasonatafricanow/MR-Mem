# P1 — normal Body response and SemanticPoint V2 sidecar

Architecture authority: [ADR-0003](ADR-0003-TURN-POINT-LIVE-WINDOW-BLOCK-COMPILATION.md).
This increment implements the Point protocol and a host-owned dialogue probe.
It does not establish Block fidelity or production readiness.

## Host integration contract

```python
from mr_mem.point_sidecar_v2 import (
    allocate_point_context,
    body_point_sidecar_instruction,
    parse_body_turn_v2,
)

context = allocate_point_context(native_session_id, native_turn_id)
instruction = body_point_sidecar_instruction(context)
# Append instruction to the normal Body request's existing live conversation.
# The HOST calls its existing Body exactly once, producing response + points.
raw_body_output = existing_body_inference(existing_live_messages, instruction)
result = parse_body_turn_v2(raw_body_output, context=context)
show_response(result.response)
# result.points remains intermediate, noncanonical semantic state.
```

MR-Mem provides no model client, retry, semantic fallback, or model configuration.
The producer owns the inference and request assembly. Parsing never invokes a model.
This module is opt-in and does not modify existing V1 admission or production routing.

The wire envelope is exactly:

```json
{
  "schema_version": "body_point_sidecar_v2",
  "response": "normal user-facing response",
  "points": [{
    "point_id": "one host-allocated current-turn slot",
    "meaning": "faithful local interpretation; may depend on earlier turns",
    "status": "RESOLVED",
    "context_links": [],
    "unresolved_refs": []
  }]
}
```

Optional Point fields are nonempty text: `polarity`, `epistemic_status`,
`temporal_scope`, and `temporal_expression`. They preserve material interpretation;
they are not a finite semantic ontology. A context link has exactly `target_kind`
(`POINT`, `BLOCK`, or `MEMORY`), `target_id`, and an open-text `relation`.

The host supplies the eligible prior targets, after its own scope/source/lifecycle
checks. P1 checks membership in that supplied set; it does not certify canonical
freshness. An unknown target is rejected, not guessed or silently discarded.

The host binds interaction/turn IDs after decoding. It allocates four stable local
Point slots using a versioned hash of the session, native turn ID, and slot number.
These are not canonical memory identities. Replaying the same host turn allocates
the same slots; new semantics cannot silently change an already persisted receipt.
Receipt persistence and lifecycle are later stages, not implemented by P1.

The Body cannot output system-owned Scope, SourceRef, revisions, timestamps,
interaction/turn IDs, or canonical identity. Duplicate JSON keys, invented IDs,
unknown fields, malformed Unicode, invalid statuses, and budget violations reject
the entire sidecar. P1 does not repair, partially accept, or rerun inference.
The host must define how to preserve the normal response when a sidecar is rejected;
this opt-in parser alone is not a gateway failure-handling policy.

`RESOLVED` means locally understood. It does not mean context-complete, canonical,
or safe to retrieve alone. Explicit unresolved references require `DEFER`.
No Point emitted here is wired to Thread, Path B, canonical admission, or retrieval.

## Operational bounds

- At most four Points per turn; at most 12,288 UTF-8 bytes for the emitted Point array.
- Meaning at most 2,048 characters; at most eight context links and eight unresolved refs.
- ID, optional text, relation, and unresolved-ref text at most 256 characters each.
- At most 128 host-activated prior targets; current slots cannot be prior targets.
- Full Body JSON at most 65,536 UTF-8 bytes; response at most 16,384 characters.

These are initial protocol safety bounds, not semantic definitions or tokenizer
estimates. Token overhead and semantic quality require provider evidence.

## Fixtures and real inference evidence

`tests/fixtures/point_v2_dialogues.json` freezes seven **synthetic** dialogue groups:
repeated correction, delayed qualification, scope distinction, unresolved reference,
partial plan update, topic diversion/return, and certainty/withdrawal. They are
dialogues with review criteria, not premade Point graphs or a claim of real-data quality.

`examples/point_v2_probe.py::run_probe` accepts the host's existing `body_infer`
callback. Each normal user turn gets one response-and-Point request. Previous raw
turns/responses and Point proposals remain available in the bounded probe context.
The probe records raw turns, response, emitted Points, model calls, provider usage,
wire bytes, and failures. Provider token bytes may attribute visible overhead as a
range; missing traces remain unknown. Reasoning overhead is not isolated by this
measurement. A rejected/truncated output stops that probe without a
repair call. Full probe evidence is private; publish only reviewed aggregate results.

Some provider traces duplicate a JSON prefill token. Excluding that duplicate for
byte alignment requires exact remaining-byte equality and matching provider visible
token accounting; the adjustment and provider count are retained separately. This
does not repair or alter the raw Body JSON, and does not estimate unseen tokens.

Native replay fixtures may provide `history` and `native_assistant_after_turn`.
The next request then keeps the original native assistant turns rather than replacing
source history with the probe response. This is teacher-forced offline replay,
not evidence of a deployed Hermes gateway hook or an unchanged real conversation.

Unit tests check protocol, references, bounds, and one-call probe execution.
They do not establish local semantic fidelity. Gate A additionally requires actual
normal Body output, reviewed multi-turn meaning and links, acceptable measured
overhead, and validation on representative native dialogue. P2 remains gated on A.
