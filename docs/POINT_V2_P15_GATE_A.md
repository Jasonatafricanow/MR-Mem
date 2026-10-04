# P1.5 Gate A — clear-turn DEFER passes; minimal commitment remains NO-GO

Date: 2026-10-04. Producer evaluated at
`69475220a3e25652181e0cd33f6202a083c14c04`.
**Gate A remains NO-GO.** The original clear-turn over-DEFER score is now
0/24, but the historical-object blocker remains and two other native Points
change an explicitly expressed actor or turn a question into a confirmed fact.
P1 is not closed; P2 has not started.

## Exact scope

Only two instruction clauses change:

1. `unresolved_refs` requires two or more semantically viable referents whose
   different resolutions change the current Point's proposition. Unknown concrete
   identity, generic discourse objects and missing details alone do not qualify.
2. Historical demonstratives retain the user's expressed abstraction level.
   An earlier calculation/treatment/plan/conclusion must not be expanded into a
   particular historical rule merely because the prior dialogue suggests one.

The field name and wire `body_point_sidecar_v2_4` are retained. All code outside
`body_point_sidecar_instruction` has the same AST as P1.4: strict envelope, host-
derived status, weak context refs, parser, alias allocation, frame route and gap
receipts are unchanged. The architecture ADR and implementation plan are unchanged.
No relation, correction operation, historical-state selection or additional Point
capability is introduced.

## Frozen experiment

Seven original synthetic groups (22 turns) plus three private native cases
(6 turns), one inference each: `deepseek-flash`, forced strict `emit_body_turn`,
thinking disabled, temperature 0.2, 16,384-token ceiling. The live profile stays
unchanged. No frame A/B, capability request, retry, repair or tool execution occurs.

Fixture SHA256 values remain:

- Synthetic: `c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
- Private native: `1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.

All 28 original conversation-prefix hashes match the P1.2 strict replay. Original
synthetic and native assistant continuations are teacher-forced source history;
new replies do not replace them. Prior Points remain newly generated proposals,
not injected gold. All 28 instruction hashes differ from P1.4.

The gate is recorded before replay. The original 24 clear-turn gold labels,
<=5% limit and inherited two-case sensitivity exclusions are unchanged. Meaning,
referents and weak context refs are reviewed manually against the raw turns and
original prefixes. No historical correction/qualification/closure work is demanded
of Point. The existing schema-miss coverage-gap criterion is also unchanged.

## Observed result

| Criterion | Strict + thinking OFF |
|---|---:|
| Requests / normal responses | 28 / 28 |
| Accepted sidecars | 28/28 |
| Missing functions / illegal contracts / coverage gaps | 0 / 0 / 0 |
| Accepted Points / single-Point turns | 34 / 25 of 28 |
| Turns with weak refs / total refs | 9 / 10 |
| Reviewed guessed ambiguous referents / wrong context refs | 0 / 0 |
| Original clear-turn over-DEFER | 0/24 = 0%, passes <=5% |
| Inherited sensitivity analysis | 0/22 = 0% |
| Required ambiguous-upload DEFER | Preserved |
| Minimal-meaning violations | 3 native turns |
| Gate A | NO-GO |

Meeting identity, the storage-plan expression and the native pricing phrase no
longer trigger clear-turn DEFER. The ambiguous upload still retains its pronoun,
DEFER and empty context refs; the later explicit clarification resolves locally.
Battery-swap value remains bare. Weekly-report scope, partial plan updates,
first-version limitation and retained city preference remain local commitments.

## Semantic blockers

At `native_execution_correction/2`, meaning still identifies the drawdown-exit
rule as the future-function criticism target. The current turn criticizes the
previous calculation/treatment while affirming rising after purchase and exiting
on drawdown. The new abstraction clause has not prevented this overcommitment.
Point can retain the demonstrative without deciding which historical rule was
rejected; this failure does not justify restoring relation or correction logic.

At `native_model_correction/1`, the Point assigns earlier research to the user,
although the current turn explicitly says the assistant had researched it.
At `native_execution_correction/0`, an open question about the group and entry day
becomes an affirmative confirmation of the queried proposition. Both violate the
existing minimal-commitment criterion within the current turn. Neither requires
Block to reconstruct history to identify the mistake.

There is also one ambiguity-definition miss **outside the original clear gold**:
`native_model_correction/0` uses unknown specific model/project identity as its
DEFER reason without establishing two viable referents that change the proposition.
This is disclosed separately; it does not alter the 24-turn numerator or denominator.
The zero clear-turn score therefore does not prove the new ambiguity boundary is
fully reliable on native turns.

The execution follow-up emits four Points, separating renaming consent, an early-
entry observation, a loss question and a related pricing proposal. The latter three
remain a granularity concern. No numerical gate is invented after output. The
model-correction turn is one Point; cancellation and city preference retain two
independent developing lines.

## Isolation and evidence limits

Every real receipt retains a normal response, source-turn fingerprint and accepted
Points; no coverage gap is observed. Existing tests still verify that a usable
response survives rejected Points, with zero Point admission, a recorded gap and
the original source prefix retained. A clean sample does not guarantee future
schema success. Later Block input remains raw live turns plus available Points;
missing Points cannot remove their source turns from a candidate span.

Completion total is 7,122 tokens; prompt total is 41,709. Median latency is 1.248
seconds; maximum accepted Point array is 479 UTF-8 bytes. Point-only visible cost
is unknown because tool logprobs are unavailable. Completion totals include normal
responses. This is one nonthinking replay, with no matched normal-Body baseline,
causal quality comparison or production provider/mode certification.

Raw replay SHA256:
`1a7f9c5a48c9940d6164363cb065dfd26bd7e085521e2dd9870f871535e12c30`.
Private raw receipts and request/prefix/instruction fingerprints are retained.
The local sanitized ledger contains counts, semantic review and lineage hashes;
native source turns and normal-response prose are not published.

159 tests pass; Ruff passes; Python 3.14 wheel builds. Draft PR #16 remains stacked
on #15. The prior reports remain unchanged. **Retain this NO-GO, leave P1 open and
P2 unstarted, and keep the frozen transport and Point/Block boundary.** No further
prompt iteration, provider work, downstream code, default merge or production
enable is included in this slice.
