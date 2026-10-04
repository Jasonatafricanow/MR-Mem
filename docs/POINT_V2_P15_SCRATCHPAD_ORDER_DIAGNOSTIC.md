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
