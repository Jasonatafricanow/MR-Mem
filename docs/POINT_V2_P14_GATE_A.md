# P1.4 Gate A — weak refs implemented; local producer remains NO-GO

Date: 2026-10-04. Producer/decoder evaluated at
`d42c4c25283da4f176283f980244a718bbc137dd`.
Point no longer emits or compiles cross-turn relation text. **Gate A remains
NO-GO** on the current narrow criteria: three clear turns are over-DEFER and one
native meaning still selects a stronger criticism target than the user expresses.
P1 is not closed; the conditional entry to P2 has not occurred.

## Implemented boundary

Wire `body_point_sidecar_v2_4` contains only:

```json
{"response":"normal reply","points":[{
  "slot":0,"meaning":"current-turn minimal commitment",
  "context_refs":["p0"],"unresolved_refs":[]
}]}
```

`context_refs` names only explicitly referenced activated objects. Host maps
aliases to typed identities; no relation, operation or historical state assertion
is stored. `context_links`, relation objects and extra fields are rejected.
Instructions to select an actual corrected prior state, describe correction or
retained scope, and distinguish historical user/assistant beliefs are removed.
Those judgments belong to Block compilation over raw live turns plus available
Points. Meaning and unresolved referents remain turn-local; status is host-derived.

ADR-0003 and the implementation plan record this user-directed Point contract
refinement and incomplete-Point coverage rule. Their turn-local Point/live-window
Block definitions and phase ordering are retained. The prior NO-GO reports remain
unchanged. No Block compiler or P2 code is added.

## Frozen replay and current gate

Original seven synthetic groups (22 turns) plus three private native cases
(6 turns), one request each: `deepseek-flash`, forced strict `emit_body_turn`,
thinking disabled, temperature 0.2, 16,384-token ceiling. The live profile is
unchanged. No frame A/B, capability probe, retry, repair or tool execution occurs.

Fixture SHA256 values remain:

- Synthetic: `c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
- Private native: `1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.

All 28 original assistant-prefix hashes match the P1.2 strict source replay.
Original P1.1 synthetic replies and native replies remain teacher-forced source
history. New replies do not replace it; new prior Points are generated hints,
not injected gold.

The gate is recorded before output. Historical correction/qualification/closure
accuracy is not demanded of Point. Current checks are minimal meaning, no guessed
referent or wrong explicit context ref, controlled clear-turn DEFER, and response/
Point failure isolation. The original 24-turn denominator and <=5% limit remain.

Following the user's instruction, a schema miss alone is no longer an automatic
NO-GO: reject all invalid Points, preserve a usable response, record the source-
linked coverage gap, and retain raw turn availability. Gaps are still reported;
failure to isolate or track them fails the gate. No post-output criterion is moved.

## Observed result

| Criterion | Strict + thinking OFF |
|---|---:|
| Requests / normal responses | 28 / 28 |
| Accepted sidecars | 28/28 |
| Illegal contracts / observed Point coverage gaps | 0 / 0 |
| Accepted Points | 34 |
| Single-Point turns | 25/28 |
| Turns with explicit refs / total refs | 8 / 11 |
| Reviewed wrong referent or context-ref targets | 0 |
| Obvious over-DEFER | 3/24 = 12.5%, fails <=5% |
| Inherited sensitivity excluding two debatable turns | 1/22 = 4.55%, passes |
| Minimal-meaning violations | 1 native |
| Gate A | NO-GO |

The ambiguous upload remains DEFER with empty context_refs; no field assigns it
to B before clarification. The later explicit B/A clarification resolves locally.
The initial battery-swap value assertion remains bare. Delayed weekly-report scope
and partial plan updates retain their local meaning without relation text. Sparse
refs are allowed and do not claim complete historical relationships.

## Narrow remaining failures

Clear-turn DEFERs occur at `diversion_and_return/1` (meeting identity),
`diversion_and_return/2` (the storage plan), and `native_execution_correction/1`
(the proposed pricing phrase). The storage plan has a unique earlier SQLite
context and should not defer merely because details are absent. The meeting and
pricing cases retain their inherited debatable classification. Excluding those
two yields 1/22, but does not excuse the separate hard meaning failure.

At `native_execution_correction/2`, meaning identifies a drawdown-exit rule as the
future-function criticism target. The current turn criticizes the previous
calculation/treatment while affirming rising after purchase then exiting on
drawdown. Point need not determine the exact cross-turn relation or reconstruct
the prior calculation; it must avoid naming a stronger rejected rule that the
user has not committed. This is current meaning overcommitment, not a requirement
to restore correction logic or relation text inside Point. No causal rationale is
added in this Point, but the selected target is still wrong.

Three native turns emit three Points. In the model-correction turn, a previous-
research reminder is isolated alongside model type and frontend cleaning. This
remains a granularity concern; no new numerical gate is invented after output.

## Gap isolation and raw authority

Every probe receipt now records a source-turn SHA256 and `point_coverage_gap`.
No gap occurs in this actual replay. A regression test drops mandatory refs from
a function envelope: normal response is preserved, zero Points accepted, error
and source fingerprint/gap recorded, and the next request retains the original
raw turn and source assistant prefix. No repair inference is added.

Later compilation must treat raw live turns as semantic evidence and available
Points as intermediate hints. A missing/rejected Point cannot remove a source
turn from the candidate span. This is documented here; Block implementation and
missing-Point Block fidelity are not claimed by P1 tests.

## Evidence and mechanical checks

Completion total: 7,135 tokens; prompt total: 40,614. Point-only visible overhead
is unknown because tool logprobs are unavailable. Median request latency is 1.195
seconds; maximum accepted Point array is 457 UTF-8 bytes. These are observations,
not a causal comparison, incremental-cost baseline or normal Body quality verdict.

Raw replay SHA256:
`54707c6d19809fcde71ea22f7c5ba56d338f4bf16a870b95021ffb672c7646fc`.
Private raw receipts and all request/prefix/instruction fingerprints are retained.
The local sanitized ledger publishes counts, review decisions and lineage hashes,
without native turn or response prose.

159 tests pass; Ruff passes; Python 3.14 wheel builds. Draft PR #16 remains stacked
on #15. Production mode/provider choice is deferred; no default merge or enable.
**Disposition: retain the weaker Point contract and coverage-gap rule, preserve
this NO-GO, and keep P2 closed because the conditional semantic gate did not pass.**
