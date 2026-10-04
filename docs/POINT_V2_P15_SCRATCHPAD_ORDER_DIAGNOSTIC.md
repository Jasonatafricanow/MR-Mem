# P1.5-SO — Semantic Scratchpad Output-Order Diagnostic

Pre-registration date: 2026-10-04. Baseline Draft PR #16 head:
`7b545f5442da0eb1576735bfb213646feb3fee08`.
Frozen semantic producer: `69475220a3e25652181e0cd33f6202a083c14c04`.
This experiment tests visible Point/response order only. It does not establish
CoT faithfulness, access to hidden state, a new Point definition or P1 closure.
P2 remains closed regardless of outcome.

## Frozen request inputs and policy

The existing probe feeds each arm's generated Points into later turns, so it is
not used for this experiment. An experiment-only harness reconstructs all 28
original prefixes and previous Point/alias payloads from the saved thinking-ON
baseline. Arm A's complete instruction hash matches that baseline for every turn.
No new response or Point can be fed into any later request in either arm.

- Synthetic fixture SHA256:
  `c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
- Private native fixture SHA256:
  `1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.
- Frozen input manifest SHA256:
  `52782e9cb1009e9415b87eb3ed490b273c486848f6906603cb2bd5407842c2ad`.
- Request schedule SHA256:
  `cfeca4c9d841aa19f8d1ea868966b928430f2b049c021705223f15bab0029d3b`.
- Predeclared experiment manifest SHA256:
  `71b48cd10dd139878df83682401050210d3efad3ae6c2bf4a09b5cbb474d74bc`.
- Shared P1.5 semantic clause SHA256:
  `d0c59c68362c535531bb77cf4846e2c966d27c2899136225fc930119e269e722`.

Production `src/mr_mem/point_sidecar_v2.py`, semantic policy, fields, weak refs,
host status, gap policy, fixtures, gold and architecture documents are unchanged.
Every request uses `deepseek-flash`, thinking enabled, temperature 0.2 and 16,384
completion-token ceiling. One inference per observation, no retry, repair,
tool execution, JSON response format or second-model judge. Logprobs request and
provider route are the same in both arms. The profile is not changed.

## Exact order directives

The common semantic and activated-context suffix is byte-identical. Only these
order clauses differ; arm labels are not model-visible.

Arm A:

```text
In this SAME normal Body inference, write the normal user-facing reply as ordinary text FIRST, then append an independent sidecar frame on a new line: <point_sidecar> followed by JSON tool arguments, then </point_sidecar> on a new line. These markers are reserved; never use them within the normal reply. No text after the closing marker. Do not put the normal reply inside JSON.
```

The manifest retains one trailing ASCII separator space after each clause.

Arm B:

```text
In this SAME normal Body inference, write the independent sidecar frame FIRST: <point_sidecar> followed by JSON tool arguments, then </point_sidecar> on a new line, then write the complete normal user-facing reply as ordinary text on a new line. These markers are reserved; never use them within the normal reply. No text before the opening marker. Do not put the normal reply inside JSON.
```

Record conversation-prefix, activated-context, semantic-clause, order-clause and
actual UTF-8 request-body SHA256 for every request. Normalizing the order clause
must leave identical A/B request bytes. Both alias hints and host validation
contexts are frozen. Baseline hints may themselves be imperfect; neither arm gets
cleaned gold hints or an arm-specific history.

The experiment decoder handles both actual orders through the unchanged P1.5
Point validator. It strips the frame from user-visible output, preserves usable
normal replies on Point rejection, and records actual-order violations separately.
An unclosed first frame has no trusted response boundary and is a failed receipt.
No JSON/meaning repair is performed. Token attribution traces the frame span only.

## Schedule and manual review

Stage 1: three native turns, eight pairs each, 48 total requests:
`native_model_correction/1`, `native_execution_correction/0`, and
`native_execution_correction/2`. Rotate case order between repeats; alternate
arm submission order. Up to two independent requests are in flight per frozen
pair; no arm output changes the other request.

Stage 2 runs regardless of stage 1: all 28 frozen turns paired once, 56 requests.
All 104 receipts, including failures, are retained. Review is order-blinded before
aggregate unblinding. Point fidelity and response fidelity are each reviewed as
correct, wrong or uncertain against the current turn and supplied dialogue.
Response fidelity here means local user-intent/dialogue fidelity; it is not a
full external technical, financial or arithmetic quality certification.

Use exactly six joint categories:
`BOTH_CORRECT`, `POINT_WRONG_RESPONSE_CORRECT`,
`POINT_CORRECT_RESPONSE_WRONG`, `BOTH_WRONG_SAME`,
`BOTH_WRONG_DIFFERENT`, `UNCERTAIN`.
An omitted/ambiguous research performer is uncertain; explicit wrong performer is
wrong. A Point must retain a question; a reply may answer it, but must not invent
group definitions as if supplied by the prefix. Historical demonstratives can
stay abstract; an explicit rejection of the affirmed exit rule is wrong, and a
bare unclear drawdown target remains uncertain. Normal reply is not Point authority.

## Predeclared endpoints and verdict

Primary endpoint is `POINT_WRONG_RESPONSE_CORRECT` in the 24 repeated pairs.
The 28 full pairs are secondary and supply unchanged Gate A/coverage metrics.
Use two-sided exact McNemar tests with alpha 0.05, pooled and by native case.
Report all six category counts and paired transitions, original denominators,
unknowns, observed component fidelity and adverse-assignment bounds. A missing
or uncertain output is not silently called correct or mismatch-free.

- `SUPPORT_SCRATCHPAD_ORDER`: repeated primary reduction remains significant and
  directional under every admissible uncertain assignment; B's lower-bound Point
  and response correct counts are at least A's upper bounds. Full B Point Gate,
  coverage and actual-order compliance metrics do not worsen; required ambiguity
  remains DEFER. Point fidelity comes first, response fidelity second, agreement third.
- `REJECT_SCRATCHPAD_ORDER`: significant repeated response regression remains
  robust to uncertainty, B increases `BOTH_WRONG_SAME`, and at least two pairs
  change from A primary mismatch to B wrong agreement; or Point-correct regression
  and primary increase are both significant. Aligned wrong outputs are deterioration.
- `NO_CONCLUSION`: all other results, including high variance, insufficient power,
  uncertain significance or mismatch disappearance without fidelity improvement.
  Persistent B mismatches may motivate representation research; they do not prove
  a universal free-text fidelity ceiling. A nonsignificant result is not equivalence.

Original 24 clear gold labels, <=5% over-DEFER limit, genuine-pronoun DEFER and
native context-review cases remain unchanged. Report wrong referents, wrong weak
refs, definite minimal-meaning errors, uncertainties, coverage gaps, Point counts,
normal coverage and frame/schema errors separately. Coverage gaps retain raw-source
availability and cannot be treated as semantic success.

Results and exact evidence lineage will be appended after all planned observations
and locked manual classifications. This document is committed before inference.

## Observed results — 2026-10-04

**Verdict: `NO_CONCLUSION`.** Point-first does not demonstrate the predeclared
fidelity/consistency improvement. It also does not satisfy the predeclared robust
rejection conditions. P1 remains open; P2 remains closed. No production protocol
adoption follows from this diagnostic.

All 104 planned inferences ran once; all provider receipts reported `stop`. One
A repeated receipt had empty visible content despite reasoning tokens, so it has
no usable response or Point. This is retained as `UNCERTAIN`, with no retry.

Experiment producer, locked before the first request:
`9bd6fe7489807d6327b3024fae629ca0661afeba`. The pre-registration/harness commit
is `780549a26a51bf40de5644a68b465afd1460f575`; its decoder-only follow-up keeps
hidden rejected frames out of user output. The semantic producer remains
`69475220a3e25652181e0cd33f6202a083c14c04` at the frozen baseline head.

### Receipt/input lineage

All **52 A/B pairs** passed prefix, activated payload, semantic-clause and
order-normalized complete request-byte equality. Actual UTF-8 request bytes are
hashed and checked against the HTTP adapter before each call. Arm A matches the
saved thinking-ON baseline complete instruction in all 28 turns. Both orders use
identical frozen host validation/alias contexts.

- Frozen thinking-ON hint receipt SHA256: `c8f0fc9c667ecc8cb9500fa10dcffd040cf22e434199cfda6c36aa6f72bd0e66`.
- Locked 104-observation review SHA256: `ed87dbfa16fb0758fe31457c1f5c694874a0426a9e8a6b46f8f582baed0eaad8`.
- Repeated receipt SHA256: `257e7c52051d429b15570fa3532254ecafd96d749747c7120c502d7a098b6f17`.
- Full receipt SHA256: `b9efa712f1a7d55dbe30a03bc4fcffde3389572fdae52a55396ae4bbbbeaa399`.
- 104-call attempt log SHA256: `218eec78e46c3573f43745b590028579d1276518748c0beafc0b14dada477a09`.
- Order A SHA256: `88042a98e7644bb5e7ec2cb103ec73969dab959a25842907b78eee9f3f6e3689`.
- Order B SHA256: `0ae7083eaa6b07e7e160c37f0c7c80c96d94425b9efdf22838f62b2f846cd6f3`.

The separately delivered `p15-scratchpad-order-evidence.json` contains all 104
observations and all five required request hashes, normalized pair hashes, raw
visible output hashes, redacted visible receipts, decoded replies/Points, failed
receipts, usage, reviews and every paired transition. Native original prefixes,
credentials, provider IDs and hidden reasoning prose are not published. Sanitizing
device/local endpoint examples and stable host IDs does not repair failure text.

### Three native turns × eight repeats

Columns use the six fixed categories in order: BC = BOTH_CORRECT; PW/RC =
POINT_WRONG_RESPONSE_CORRECT; PC/RW = POINT_CORRECT_RESPONSE_WRONG; BWS =
BOTH_WRONG_SAME; BWD = BOTH_WRONG_DIFFERENT; U = UNCERTAIN. No extra joint
category was introduced.

| Frozen turn | Arm | BC | PW/RC | PC/RW | BWS | BWD | U |
|---|---|---:|---:|---:|---:|---:|---:|
| native_model_correction/1 | A | 1 | 0 | 0 | 0 | 0 | 7 |
| native_model_correction/1 | B | 1 | 1 | 0 | 0 | 0 | 6 |
| native_execution_correction/0 | A | 3 | 0 | 0 | 0 | 2 | 3 |
| native_execution_correction/0 | B | 2 | 0 | 1 | 0 | 2 | 3 |
| native_execution_correction/2 | A | 0 | 0 | 0 | 0 | 1 | 7 |
| native_execution_correction/2 | B | 0 | 2 | 0 | 0 | 1 | 5 |
| Total (24) | A | 4 | 0 | 0 | 0 | 3 | 17 |
| Total (24) | B | 3 | 3 | 1 | 0 | 3 | 14 |

Point fidelity C/W/U: **A 4/3/17; B 4/6/14**. Response fidelity C/W/U:
**A 17/6/1; B 16/8/0**. Correct agreement is 4 versus 3; wrong agreement is
0 versus 0. Those zero wrong-agreement counts do not certify absence of the
causal bottleneck risk because so many Point readings remain uncertain.

Observed primary mismatches: **A 0/24; B 3/24**. A has 14 endpoint-unknown
observations and B has 10: feasible marginal primary counts are A 0–14 and
B 3–13. `UNCERTAIN` is not assigned zero. A known correct Point or a known
wrong reply can still establish a zero primary endpoint even when the other
component is uncertain.

Only 6/24 paired primary endpoints are definite on both sides. Their discordants
are 0 A-only / 0 B-only, exact McNemar p=1; **this is not an all-24 result or
evidence of equivalence**. Across all admissible unknown assignments, A−B ranges
from **−13 to +11**, p ranges 0.000244–1. Therefore neither effect direction nor
significance is robust. By case, definite pairs/unknown pairs and difference bounds:

| Native turn | Definite pairs | Unknown pairs | A−B bounds | Exact p bounds |
|---|---:|---:|---:|---:|
| native_model_correction/1 | 0 | 8 | -5 to 5 | 0.0625 to 1.0000 |
| native_execution_correction/0 | 4 | 4 | -3 to 2 | 0.2500 to 1.0000 |
| native_execution_correction/2 | 2 | 6 | -5 to 4 | 0.0625 to 1.0000 |

The two B historical-target mismatches are definite meaning errors, not merely
DEFER errors: `R033`/`R047` expand the criticism to drawdown calculation generally,
while their normal replies preserve the causal trailing exit and criticize prior
retrospective pricing. The third, `R045`, has the correct actor in meaning but
over-DEFERs the generic decision-model object. No admitted repeated Point explicitly
assigns the research to the user; many omit the actor and remain uncertified.

A−B response-correct bounds are +1 to +2, with p in 0.753906–1; Point-correct
bounds are −14 to +17. There are **zero** confirmed A primary→B BOTH_WRONG_SAME
transitions, and neither fidelity regression is robustly significant. Support
and rejection conditions both fail. The primary sample is deliberately enriched
for earlier failures and is not a general dialogue failure-rate estimate.

### Complete 28-turn paired replay

| Joint category | A | B |
|---|---:|---:|
| BOTH_CORRECT | 22 | 23 |
| POINT_WRONG_RESPONSE_CORRECT | 2 | 2 |
| POINT_CORRECT_RESPONSE_WRONG | 1 | 1 |
| BOTH_WRONG_SAME | 0 | 0 |
| BOTH_WRONG_DIFFERENT | 0 | 0 |
| UNCERTAIN | 3 | 2 |

Point C/W/U: **A 23/2/3; B 24/2/2**. Response C/W/U: **27/1/0 in both**.
Correct agreement 22→23 comes partly from coverage, and does not establish
an order-driven fidelity improvement. Primary is **2/28 in both**, with 3 A
and 2 B endpoint-unknown observations. Feasible marginal counts: A 2–5, B 2–4.

Of 28 pairs, 24 have definite primary endpoints; discordants 1 A-only / 1 B-only,
exact p=1. Including unknown assignments: A−B −2 to +3, p 0.375–1. Point-correct
A−B −3 to +2; response-correct difference 0 (all 28 pairs definite, p=1).

Per-turn classifications below retain original fixture order. Evidence retains
component judgments, complete outputs and reviewer reasons.

| Frozen turn | A category | B category |
|---|---|---|
| repeated_correction/0 | BC | BC |
| repeated_correction/1 | BC | BC |
| repeated_correction/2 | BC | BC |
| delayed_qualification/0 | BC | BC |
| delayed_qualification/1 | BC | BC |
| delayed_qualification/2 | BC | BC |
| scope_distinction/0 | BC | BC |
| scope_distinction/1 | BC | BC |
| scope_distinction/2 | BC | BC |
| cross_turn_reference/0 | BC | BC |
| cross_turn_reference/1 | PC/RW | PC/RW |
| cross_turn_reference/2 | BC | BC |
| partial_plan_update/0 | BC | BC |
| partial_plan_update/1 | BC | BC |
| partial_plan_update/2 | BC | BC |
| diversion_and_return/0 | BC | BC |
| diversion_and_return/1 | BC | BC |
| diversion_and_return/2 | BC | BC |
| diversion_and_return/3 | PW/RC | BC |
| certainty_and_withdrawal/0 | BC | BC |
| certainty_and_withdrawal/1 | BC | BC |
| certainty_and_withdrawal/2 | BC | BC |
| native_model_correction/0 | U | U |
| native_model_correction/1 | U | PW/RC |
| native_execution_correction/0 | BC | PW/RC |
| native_execution_correction/1 | U | BC |
| native_execution_correction/2 | PW/RC | U |
| native_rhetorical_correction/0 | BC | BC |

### Original Gate A metrics and coverage

| Full replay metric | A | B |
|---|---:|---:|
| Normal responses | 28 | 28 |
| Admitted sidecar turns | 26 | 27 |
| Contract rejections/misses | 2 | 1 |
| Coverage gaps | 2 | 1 |
| Gap raw sources retained | 2 | 1 |
| Total Points | 26 | 31 |
| Single-Point turns | 26 | 24 |
| Multiple-Point turns | 0 | 3 |
| Clear-gold over-DEFER / 24 | 1 | 1 |
| Clear-gold missing Point / 24 | 1 | 0 |
| Required ambiguous DEFER / 1 | 1 | 1 |
| Wrong admitted referent | 0 | 0 |
| Wrong admitted context ref | 0 | 0 |
| Definite minimal-meaning violation | 1 | 0 |
| Uncertified Point fidelity | 3 | 2 |
| Outside-clear-gold over-DEFER | 0 | 1 |
| Confirmed PW/RC mismatch | 2 | 2 |
| Actual order violation | 1 | 0 |

Observed over-DEFER is **1/24 = 4.17% in both**, using unchanged gold. A fails
first-version limitation; B fails decision-model generic identity. The unchanged
22-turn sensitivity gives 1/22 in both. A has one missing clear-gold Point; its
possible over-DEFER total is 1–2/24, so the observed subgate pass is not a complete
semantic certificate. The real ambiguous “它” remains DEFER in both; both replies
nevertheless prematurely commit its referent (data or cloud), so their response
fidelity fails. Point is not overwritten using those replies.

A has a definite historical-target meaning error (`F020`); B keeps a quoted
unclear target (`F051`) and an omitted actor (`F048`), which are not relabeled
as repairs. B also over-DEFERs G1/D1 outside the original clear denominator
(`F044`). **Original Gate A is not certified in either arm; retain NO-GO/P1 open.**
Clear-gold rate alone cannot close the original semantic audit.

Across both phases: A gaps 5+2, B gaps 5+1. All 13 gap turns retain original
raw source and turn hashes. Of the 12 Point-only rejections/misses, all 12 normal
replies survive and no Point is admitted. The thirteenth has empty provider
visible content, so there is no reply to preserve. Invalid JSON is not repaired.

Repeated A: 3 missing frames, 1 unactivated alias rejection, 1 empty response.
Repeated B: 3 unactivated alias rejections, 1 invalid JSON, 1 extra/non-object
field contract rejection. Full A: 1 missing frame, 1 unactivated alias; full B:
1 unactivated alias. Missing frames are disclosed as missing, not structural
JSON errors. Accepted refs have zero definite semantic misbindings. Rejected alias
strings are retained in evidence and not hidden under that admitted-ref zero.

### Token, reasoning and latency observations

Each entry is median (min–max). Visible frame bounds count only the frame, never
the following B response. They use provider visible token byte traces and an EOS
boundary interval; frame misses/empty output are excluded from frame-range samples.

| Phase / arm | Visible completion | Reasoning | Latency seconds | Frame observed bounds |
|---|---:|---:|---:|---:|
| repeated / A | 419.5 (0–707) | 1416.5 (268–7882) | 10.2955 (3.914–41.209) | 51–127 (n=20) |
| repeated / B | 460.5 (285–802) | 1261.5 (359–6108) | 10.124 (4.685–32.702) | 58–192 (n=24) |
| full / A | 85.5 (54–531) | 1299.5 (72–5262) | 8.2395 (1.583–27.507) | 42–92 (n=27) |
| full / B | 109 (64–725) | 1170.5 (348–5652) | 7.873 (2.626–30.815) | 42–141 (n=28) |

The B order directive adds one prompt token with this tokenizer (semantic bytes
are identical). Full-replay prompt cache-hit means differ: A 937.1 versus B 402.3
tokens; B replies are also longer (visible mean 220.4 versus A 160.3). Lower median
reasoning/latency is descriptive, not an incremental Point-cost or performance
certificate. There is no matched no-sidecar baseline. No hidden reasoning text is
used as semantic authority or released.

### Reproduction and checks

Use the experiment functions only: `freeze_inputs(cases, saved_thinking_on_replay)`,
`verify_pair(turn)`, `make_request(turn, arm)`, then `run_plan(frozen, schedule,
infer_once, new_receipt_path, workers=2)`. `infer_once` performs one HTTP request
using exactly the returned payload and records actual request-byte hash, usage and
visible output; it neither executes a tool nor retries. Private prefixes and
baseline hint receipts are required for the exact run and are not reconstructed
from this public report. The locked schedule/inputs/reviews are identified above.

Validation at the experiment producer: **172 pytest tests passed**, Ruff over
`src tests examples` passed, wheel build passed. No changes under `src`, original
fixtures, gold or the architecture ADRs occur between baseline and experiment.
The result commit adds only this report to the preregistered harness/test changes.
Exact final-head CI is checked after publication and linked in the PR and local
deliverable, avoiding a report/CI commit loop.

Manual review is one order-blinded evaluator, not an independent multi-rater
fidelity certificate. All failures and uncertainties remain in the delivered
evidence. This diagnostic neither supports adopting Point-first nor establishes
that order is irrelevant or that free-text Point has a universal ceiling.
