# P1.5 diagnostic — reply/Point inconsistency and unchanged thinking-ON replay

Date: 2026-10-04. Source head:
`8b4d14b9337c677d1692f88fc06788acbd3f4eb5`.
Semantic producer remains `69475220a3e25652181e0cd33f6202a083c14c04`.

**Two original strict-OFF failures show correct targeted understanding in the
normal reply and an inconsistent Point in the same inference.** They cannot all
be attributed to the Body's inability to understand the current turn. The unchanged
P1.5 thinking-ON frame sample retains questions better, but does not certify all
three semantic axes. This is a diagnostic result, not a replacement Gate A
certificate. The [original P1.5 NO-GO](POINT_V2_P15_GATE_A.md) is preserved;
P1 remains open and P2 has not started.

## Original failure audit

Read the three existing strict + thinking OFF receipts without further inference.
Compare only the implicated local semantic axis; normal replies are not certified
for their complete factual, arithmetic or algorithmic quality.

| Original sample | Normal response | Point | Finding |
|---|---|---|---|
| `native_model_correction/1`, actor | Assistant acknowledges its own earlier research | Research assigned to the user | Clear same-inference reply/Point mismatch |
| `native_execution_correction/2`, target | Keeps drawdown rule distinct from the criticized earlier treatment | Names the exit rule as the rejected future-function object | Clear same-inference reply/Point mismatch |
| `native_execution_correction/0`, speech act | Answers affirmatively and supplies group definitions unsupported by the prefix | Records affirmative user confirmation | Point is wrong; an assertion used to answer a question does not by itself establish that the reply misread the user's speech act |

The G1 sample therefore remains indeterminate for locating the speech-act error's
internal cause. A reply evidence-support gap is observable, but it is not proof
that the model treated the user as asserting the answer. Do not force this sample
into the binary tree or use an assistant continuation unavailable to the request
as if it were model-visible evidence.

Function arguments and host-decoded meanings are identical. These disagreements
already exist in model output; no host semantic rewriting is involved. The first
two samples support a generated-sidecar representation/consistency investigation,
not a diagnosis that all failures come from thinking OFF.

## Unchanged protocol, original 28 turns

Run `deepseek-flash` with explicit thinking enabled and the existing frame route:
ordinary normal response, followed by the reserved Point frame. No forced tool,
tool execution, JSON response format, retry, repair, semantic-only inference or
new prompt prohibition. The live profile is not modified. This reproduces the
production reasoning setting offline; it is not a deployed gateway integration.

The P1.5 semantic clause is identical across tool and frame instructions and is
checked on every request. SHA256:
`d0c59c68362c535531bb77cf4846e2c966d27c2899136225fc930119e269e722`.
Only the existing transport framing differs. Source code, protocol, Point fields,
host-derived status, weak refs, aliases and gap handling remain unchanged.

All 28 raw turns and original assistant-prefix hashes match the strict replay.
Prior generated Points remain proposals, not injected gold. Fixture hashes remain:

- Synthetic: `c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
- Private native: `1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.

The original 24 clear-turn labels, <=5% threshold and inherited two-case sensitivity
exclusions are retained. Missing Points would be separately disclosed as coverage
gaps, not scored as resolved or over-DEFER and not removed from later raw spans.
No gap occurs in this sample. A missing diagnostic Point could not be called fixed.

## Observations

| Observation | Thinking ON + existing frame |
|---|---:|
| Requests / usable normal replies / accepted sidecars | 28 / 28 / 28 |
| Contract errors / observed coverage gaps | 0 / 0 |
| Clear-turn Point coverage | 24/24 |
| Points / single-Point turns | 33 / 24 of 28 |
| Obvious over-DEFER | 1/24 = 4.17%, passes <=5% |
| Inherited sensitivity | 1/22 = 4.55% |
| Required ambiguous-upload DEFER | Preserved |
| Reviewed guessed ambiguous referents / wrong context refs | 0 / 0 |
| Explicit context refs | 10 |
| Responses with reasoning content and token counts | 28/28 |

The clear-turn DEFER is `native_model_correction/1`: the model considers the current
demonstrative to mean either the model or the whole replacement solution. Retain
the original clear gold and count this as over-DEFER. The identity-only DEFER at
`native_model_correction/0` does not recur; keep that turn outside the original
denominator. The genuine backup pronoun ambiguity remains deferred.

| Original failure axis | Thinking-ON normal response | Thinking-ON Point | Diagnostic outcome |
|---|---|---|---|
| G1 question | Acknowledges missing group-definition evidence; offers conditional interpretations | Explicitly records a question, with no affirmative user confirmation | Question preserved |
| Research actor | Assistant still acknowledges its own research | Says the user states research already happened, without preserving the explicit researcher | Actor underspecified; do not claim an identical definitive reversal, or certify fixed |
| Historical criticism target | Separates rule from earlier treatment and asks which operation is criticized | Starts with the abstract assistant algorithm, then describes an unclear use of the drawdown phrase and defers it | Weaker than the old explicit exit-rule rejection; target remains uncertified |

The last two are **uncertified meanings**, not asserted to be identical recurrences
of the original hard errors. Actor omission and an unclear predicate target must
not be reported as clean fixes. Conversely, the historical Point no longer
explicitly condemns the exit rule, so do not inflate the remaining-error count.
This diagnostic does not invent a new Gate threshold or turn uncertainty into
an automatic NO-GO metric. It cannot establish that the original three axes have
all passed, so it does not replace the existing strict Gate A verdict.

Four turns emit multiple Points: the initial plan's independently updated dimensions,
cancelled move versus retained preference, renaming consent versus analysis request,
and symptom correction versus software questions. Disclose these splits without
adding a numerical granularity gate. No historical relation or closed taxonomy
is introduced in Point.

## Limits and evidence

Completion total is 48,633 tokens: 43,500 reasoning and 5,133 visible. Prompt total
is 31,389. Byte-aligned frame overhead is 41–122 visible tokens per turn, with a
total interval of 1,734–1,790 tokens. Median request latency is 7.139 seconds;
maximum accepted Point array is 388 UTF-8 bytes. Extra reasoning cannot be
attributed to the sidecar without a matched normal-Body baseline. These numbers
are not an incremental-cost or production-mode acceptance.

Thinking and transport both change from strict OFF, and generated prior Point
hints can differ. Equal raw prefixes and semantic clauses do not isolate a causal
thinking effect. A clean 28-frame sample also does not prove perfect future frame
delivery. Coverage-gap policy remains required, and raw live turns remain final
semantic evidence for later contextual Block compilation.

Raw thinking-frame replay SHA256:
`c8f0fc9c667ecc8cb9500fa10dcffd040cf22e434199cfda6c36aa6f72bd0e66`.
Private source/reply prose and the full three-turn audit are retained locally.
The sanitized ledger records review classes, request IDs, source/prefix/instruction
hashes and counts. Only this sanitized diagnostic document is added to the repo.

No source code changed, so prior P1.5 mechanical checks remain applicable. The
diagnostic validates frozen hashes and raw-wire/decoded equality. No P1.6 prompt,
new Point representation, provider switch, P2, Block/Thread/LCE work, default merge
or production enable is included.
