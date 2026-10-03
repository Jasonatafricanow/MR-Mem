# Point V2 P1 — Gate A result: NO-GO

Date: 2026-10-03. Architecture baseline: MR-Mem PR #15,
`8c7d7f5085dd0c22374ecd1a57cc3ba4754f1096`.
Protocol: [P1 contract](POINT_V2_P1_PROTOCOL.md).

## Engineering result

The opt-in module decodes normal Body response and turn-local Point siblings from
one completed inference. It binds host turn/interaction/Point identities, validates
activated context targets, preserves open-text relations and material qualifiers,
and rejects malformed, unbounded, invented-authority output or explicit unresolved
references declared RESOLVED. It cannot detect a model's undeclared semantic guessing.

Local verification: 124 tests, Ruff over source/tests/examples, diff whitespace
checks, and wheel build. The local interpreter is Python 3.14; repository CI targets
Python 3.12. These results prove the protocol and failure handling, not model fidelity.
No canonical admission, buffer, Block compiler, Thread/LCE consumer, or gateway
production hook was enabled by this increment.

## Actual Body experiments

The private adapter used the existing Hermes profile's configured `deepseek-flash`
endpoint and credential. Thinking mode was retained; no new semantic model/provider
was installed. Each processed user turn had one response-and-Point request, with
no repair inference or semantic-only model call.

| Run | Body requests | Valid wire outputs | Emitted Points | Outcome |
| --- | ---: | ---: | ---: | --- |
| Initial synthetic run, 4K output cap | 3 | 2 | 3 | Third response truncated; stopped |
| Seven synthetic dialogue groups, 16K cap | 22 | 22 | 33 | Wire valid; fidelity failures found |
| Native dialogue replay, 16K cap | 3 | 2 | 7 | Third output malformed; stopped |

Total: 28 actual Body requests. The native export contained three groups/six user
turns; only three turns ran before the first rejection. The final rhetorical
correction group did not run. Native raw turns/replies were read with SQLite
`mode=ro` and exact content fingerprints. Replay retained original native assistant
replies between turns. This is offline teacher-forced replay, not deployed gateway
evidence. No raw native transcript or credential is committed here.

For the 22-turn synthetic run, visible wire/sidecar overhead was 109–493 tokens per
turn, measured using provider token bytes and response-boundary attribution; median
Body latency was 11.495 seconds and maximum emitted sidecar size was 1,378 UTF-8
bytes. The two accepted native outputs had 539–543 visible overhead tokens, with
maximum sidecar size 1,664 bytes. Token-prefill trace adjustments are recorded in
the private evidence, separately from provider visible token counts.
Reasoning overhead has not been isolated against a matched normal-Body control;
acceptable total overhead therefore remains unproven.

## Gate blockers

1. **Invented qualification.** In `scope_distinction`, turn 0, the user only says
   “换电本身有价值”. The Point adds that value does not depend on supporting modes or
   external conditions. That condition independence was not expressed by the user.
2. **Premature reference resolution.** In `cross_turn_reference`, turn 1, the
   fixture's unresolved “它” is assigned to the cloud/data interpretation with
   `RESOLVED` before the later clarification. The candidate failed the frozen
   fixture's explicit DEFER requirement. Its final correct referent does not prove
   that the earlier certainty was justified.
3. **Malformed actual Body output.** Native replay's first execution-correction
   turn returned `finish_reason=stop` with invalid JSON. The strict parser rejected
   it; no repair call, partial Point acceptance, or canonical write followed.
4. **Acceptance evidence incomplete.** Representative native replay stopped early;
   production hook behavior and total reasoning overhead remain unverified.

Additional observations: the model sometimes adds operational DEFER points or
conversational-adjacency links. Their granularity and usefulness still need review;
they are not accepted Block membership or canonical facts.

## Decision and next permitted work

**Gate A = NO-GO. P2 has not started.**

The candidate protocol/probe can be reviewed in draft. Resume within P1 by improving
same-inference Body emission, uncertainty handling, and host failure behavior;
then rerun the frozen dialogue suite and representative native replay with source,
prompt, model, request-count, and overhead receipts. Do not weaken the fixtures,
guess missing references, add a second online interpreter, or bypass A to build
the buffer/Block/Thread/LCE stages.

These findings concern the current Point producer implementation. They neither
redefine Point/Block nor alter Issue #14's limited negative conclusion.
