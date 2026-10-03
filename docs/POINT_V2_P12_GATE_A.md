# P1.2 Gate A — strict transport passes; overall NO-GO

Producer/decoder evaluated at `562a27296f3be0af8a992b76bb7ff2fba453c3a8`.
P1.2 removes Body-authored status and tests the requested forced strict single-
envelope against the retained frame route. **Strict output is complete and legal
on all 28 requests, but overall Gate A remains NO-GO**: excessive DEFER and one
native correction-target/causal-commitment error remain. P2 is not started.
No Point/Block architecture or semantic policy is redefined.

## Contract and capability

The Body emits slots, meaning, links and unresolved refs. The host alone derives
`DEFER` from nonempty refs, otherwise `RESOLVED`. No model-authored status or
field-consistency prompt is required. No `open_context` field is added without a
separately established nonreferential deferral case.

The strict function `emit_body_turn` carries `response` and `points` in its
arguments. Named tool choice forces the function; no tool is executed, no result
is sent back and no second inference is made. Empty `message.content` is valid.

The unchanged default-thinking DeepSeek route rejects forced choice with HTTP 400
on both Chat and Responses API capability requests. This matches the provider's
[documented restriction](https://api-docs.deepseek.com/api/create-chat-completion/).
A nonthinking capability request succeeds with a complete envelope and empty
content. Accordingly **both A/B arms explicitly disable thinking**, retaining
`deepseek-flash`, temperature 0.2 and max completion tokens 16,384. The live profile
is unchanged; these results cannot certify forced output on its thinking route.

## Paired replay and gate fixed before output

Original fixtures are unchanged:

- Synthetic SHA256: `c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
- Private native SHA256: `1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.

Each arm receives the same seven synthetic groups (22 turns) plus three native
cases (6 turns). Both reuse the original P1.1 synthetic assistant replies and the
original native assistant replies. Conversation-prefix hashes match on all 28
pairs; generated replies do not replace source prefixes. Both arms have identical
semantic-policy text, with only transport/shape differing. Prior proposal states
remain arm-specific; native text remains private. Requests ran in concurrent
cohorts, so cache/order effects prevent causal latency conclusions.

The gate was recorded before A/B output: zero false RESOLVE, zero missing/empty
sidecar on these semantic turns, zero illegal contract and zero missing reply.
Obvious over-DEFER may affect at most 5% of 24 clearly interpretable turns (21
synthetic turns excluding the ambiguous upload, plus three clear native turns).
Thus at most one over-DEFER turn is allowed. All emitted meaning, links and refs
were manually reviewed; no model judge or repair call was used.

## Results

| Criterion | Strict single-envelope | Frame control |
|---|---:|---:|
| Normal responses | 28/28 | 28/28 |
| Accepted sidecars | 28/28 | 25/28 |
| Missing sidecars | 0 | 3 native |
| Illegal contract | 0 | 0 |
| Reviewed false RESOLVE | 1 native correction target | 0 in emitted Points |
| Obvious over-DEFER, predeclared denominator | 6/24 = 25.0% | 3/24 = 12.5% |
| Accepted Points | 32 | 32 |
| Single-Point turns | 25 | 21 of 25 emitted turns |
| Overall Gate A | NO-GO | NO-GO |

All 28 strict requests return one legal function envelope with nonempty response
and Points, finish `tool_calls`, and omit Body-authored status. The redundant-state
illegal combination is eliminated by construction. All framed argument JSON is
legal; frame failures are missing output, not malformed JSON. Frame misses occur
at `native_model_correction/1`, `native_execution_correction/0` and
`native_execution_correction/2` (zero-based turns).

The original scope-distinction Point remains the bare value assertion in both
arms. The ambiguous upload emits unresolved `it` and derives DEFER in both; the
next explicit B clarification resolves without guessing. These successes are
specific reviewed regressions, not universal semantic certification.

## Remaining fidelity and over-DEFER

Strict over-DEFER turns: all three `repeated_correction` turns,
`diversion_and_return/1`, `diversion_and_return/2`, and
`native_execution_correction/1`. Refs include unspecified exit procedure/actor
role, a meeting identity, the storage proposal despite a unique linked SQLite
line, and the proposed pricing-method phrase. Missing execution parameters are
being promoted into local semantic ambiguity; refs also propagate across the
correction chain despite the current turn's explicit narrow commitment.

Frame over-DEFER turns: `diversion_and_return/1`, `diversion_and_return/2`, and
`certainty_and_withdrawal/2`. The retained city-preference commitment can stay
minimal without inventing a city or deferring solely for its detailed content.

The meeting identity and native pricing-method case have debatable ambiguity
classification. The preregistered score is retained rather than changing gold
after seeing output. A sensitivity analysis excludes those two turns in both
arms: strict still has **4/22 = 18.2%**, frame **2/22 = 9.1%**. Both still exceed
5%; frame also misses a clear native turn, so over-DEFER is not its only failure.

Strict `native_execution_correction/2` has a hard fidelity error: a resolved Point
attributes the future-function complaint to the trailing drawdown exit rule the
user affirms, instead of preserving the criticism of the previous calculation.
It adds an unuttered retrospective-peak causal explanation. This is one incident
failing both correction-target resolution and minimal commitment; it is not two
independent error counts. Strict syntax cannot prevent it. The turn also splits
one correction into two Points, which is a granularity concern.

Frame `repeated_correction/1` adds a personal-use restriction that the user did
not express. Frame's four-Point native rhetorical/software/payment turn also
splits the related fee/recharge line more finely than the earlier accepted state.
These observations are retained; the semantic prompt is not rewritten in P1.2.

## Measurement and reproducibility

Strict tool `logprobs` is null on all 28 requests, so Point-only visible overhead
is **unknown**, not estimated with another tokenizer. Strict completion total is
7,564 tokens, including ordinary replies and the envelope. Frame completion total
is 5,510 tokens; measured visible frame/Point overhead is 46–160 tokens on 25
framed turns. These are not an incremental reasoning baseline for the unchanged
thinking Body. Strict's median request latency is 1.209 seconds versus frame's
1.009; concurrency/cache/response-length differences limit interpretation.

The A/B contains 56 completed Body inferences. One nonthinking capability sample
adds one inference; two thinking capability requests were rejected before
inference. Total: 59 HTTP attempts, 57 completed inferences, zero tool executions,
zero repair/semantic-only calls. All receipts are retained.

Raw A/B SHA256:

- Strict: `9690bb5834f0f0bcd13c352ba9831cab2e165e871ec2fd1833eadeb2762aa6e2`.
- Frame: `2082cca4ee674122174a9a44f14ff3b4c6376972825bad503e8a4ef7d50cc788`.

The local sanitized ledger records predeclared gate/review hashes, capability
hashes, shared-prefix hashes, request IDs, counts and decisions without native
prose. Original P1/P1.1 NO-GO documents and raw evidence are unchanged.

Mechanical validation: 154 tests pass, Ruff passes, Python 3.14 wheel builds.
Draft PR #16 remains stacked on #15; no default merge or production enable.
**Disposition: keep the P1.2 contract cleanup and strict-envelope experiment,
retain the frame control, keep Gate A NO-GO and P2 closed.**
