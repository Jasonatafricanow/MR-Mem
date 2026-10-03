# P1.1 Gate A — NO-GO

P1.1 implements the five requested producer/protocol changes. The final unchanged
fixture rerun preserves 28/28 normal responses and fixes the two original hard
synthetic regressions, but native sidecar admission remains 3/6. **Gate A is NO-GO;
P2 remains closed.** The original [P1 NO-GO](POINT_V2_P1_GATE_A.md) is retained.
This result does not revise the frozen Point/Block architecture.

## Evaluated implementation

- Producer/decoder commit: `e07305d56bcf8a28a21916cc9cec4577c8214174`.
- `meaning` now means minimal semantic commitment; no unexpressed independence,
  exclusivity, causation, permanence, preference or negation.
- Dialogue-determined unique antecedents permit `RESOLVED`; ambiguity or reliance
  on common sense requires `DEFER`.
- Current Points emit integer slots; prior targets use host-bound `p0/p1/b0/m0`.
  Neither current stable IDs nor prior target IDs appear in model-visible proposals.
- Default one Point per turn; multiple only for independently developing lines.
- Response and sidecar validation are isolated; rejected sidecars produce zero
  Points without replacing or swallowing normal replies. No inference repair.

The normal Body route remains `deepseek-flash`, existing default thinking mode,
temperature setting and a 16,384 completion-token ceiling. No model switch or
thinking-mode change was used to make the test pass. This is opt-in offline probe
work, not deployment into Hermes.

## Transport capability and selection

First tested ordinary tools and beta strict tools, one inference each. Both
returned user-facing content and sidecar tool arguments without a second call.
The full strict-tool fixture attempt then returned empty content on its eighth
request. It is therefore unsuitable as the default simultaneous-output contract
for this provider; structural tool support alone is insufficient.

The selected fallback is normal text followed by an independent reserved
`<point_sidecar> ... </point_sidecar>` frame on the original chat-completions
route, with JSON mode and tools absent. Argument JSON is validated independently.
Malformed, missing, duplicate, truncated or invalid-reference sidecars are rejected.
The normal response is preserved whenever usable content exists. A provider
response containing only reasoning is recorded as failure, never substituted with
reasoning text or repaired by another model call.

## Frozen fixtures and final results

Neither fixture changed. Synthetic file SHA256:
`c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
Private native replay SHA256:
`1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.
Native replay retains the original source history and assistant continuations.

| Final rerun | Body requests | Normal responses | Accepted sidecars | Accepted Points | Visible overhead |
|---|---:|---:|---:|---:|---:|
| Seven synthetic groups | 22 | 22 | 22 | 24 | 47–121 tokens/turn |
| Three native replay cases | 6 | 6 | 3 | 4 | 77–176 tokens/turn, five framed turns |

The synthetic rerun has 20 single-Point turns and two two-Point turns: independent
A/B backup proposals, and withdrawal of a moving plan versus retention of a city
preference. Correction, qualification and partial schedule updates each stay in
one Point. Native rhetorical correction separates the literal-symptom correction
from an independent software-use/payment question; the other accepted turns emit
one Point.

Reviewed against the existing seven fixture criteria:

- Repeated correction retains the correction chain and the continuing vehicle/
  technology line without turning one correction into several near-duplicates.
- Delayed qualification retains test-group-only automation and formal-group approval.
- Scope distinction starts with exactly `换电本身有价值`; no independent-of-external-
  conditions commitment is introduced into the Point. Economic uncertainty remains.
- The ambiguous upload turn emits `DEFER` with `unresolved_refs=["它"]`; only the
  later explicit clarification resolves B and retains A's unchanged flow.
- Partial plan updates change publication alone and retain demo/budget via context.
- Diversion/return retains local SQLite, temporary no-cloud scope and later review.
- Certainty/withdrawal retains tentative -> confirmed -> cancelled progression
  without rewriting the city-preference line into a cancelled commitment.

This is a limited reviewed sample, not a blanket semantic-quality certification.
Conservative DEFERs remain in the weekly-report start, first-version limitation
and retained city preference. Execution-detail unknowns versus semantic reference
ambiguity still need calibration; no gold or fixture was changed to excuse them.
Normal response prose is not Point authority and was not postprocessed.

## Native blockers, separated from JSON syntax

All six final native requests returned normal content, including the three rejected
sidecars. All five present frame argument objects were legal JSON. The previous
native illegal-JSON failure did not recur in this bounded rerun.

| Native receipt (zero-based turn) | Failure class | Observed result |
|---|---|---|
| `native_model_correction / 0` | Producer reference/status consistency | Nonempty unresolved refs with `RESOLVED`; whole sidecar rejected |
| `native_model_correction / 1` | Output-channel completeness | Normal response but no sidecar frame; zero Points |
| `native_execution_correction / 0` | Producer reference/status consistency | Nonempty unresolved refs with `RESOLVED`; whole sidecar rejected |

The host did not rewrite these statuses to DEFER, salvage partial Points or
retry them. Missing sidecars leave an intermediate semantic coverage gap even
when response delivery succeeds. Consequently syntax isolation and the synthetic
improvements do not satisfy native Gate A.

## Cost, provenance and retained failed attempts

The final synthetic Point array peaks at 348 normalized UTF-8 bytes; accepted
native arrays peak at 601. Median request latency is 10.306 seconds synthetic and
8.983 seconds native. These figures describe these samples, not provider guarantees.
Compared with the earlier observed 109–493 synthetic and 539–543 native visible
overhead, the emitted sidecar is smaller. This is not a matched causal or total-
cost comparison: reasoning overhead is not isolated, and normal reply length varies.

Token ranges require exact provider-byte alignment to raw output. Tokens crossing
the response/frame boundary widen the range. The provider reports one extra visible
token outside the byte trace on the measured samples; it is recorded separately
and included only in the upper bound. The native missing-frame turn has unknown
sidecar attribution. Wire bytes never substitute for measured tokens.

P1.1 used **61 actual Body requests in total**, including retained failed attempts:

| Experiment | Requests | Disposition |
|---|---:|---|
| Ordinary and strict tool capability samples | 2 | Same-inference content + tools demonstrated on samples |
| Strict-tool synthetic attempt | 8 | Stopped on missing normal content |
| First synthetic frame attempt | 22 | Overly strict closing-newline host parser; raw evidence retained |
| First native frame attempt | 1 | Provider returned only reasoning; raw failure retained |
| Final native replay | 6 | Complete coverage; three sidecar rejections |
| Final synthetic rerun | 22 | Complete coverage; 22 accepted sidecars |

The closing-newline parser was fixed by allowing insignificant whitespace around
JSON and the closing marker. The full synthetic test was rerun from the unchanged
fixture with the corrected decoder; earlier failed receipts were not relabelled.
Native replay can continue after missing generated content using original source
assistant turns, while keeping the failed receipt. No repair inference or tool
execution occurred in any experiment.

Final raw-evidence SHA256 values:

- Synthetic: `f0135c4160a9670e19573baf29bff93aeade68dadf150ca138baafc5a59922af`.
- Native: `06c280d583b61a5fd8a03fdfe46bb2d7376417cf42b9d2c52ff169813a822607`.

Full native turns, generated replies and token traces remain private. The local
sanitized evidence ledger includes all attempt hashes and per-turn status/usage
attribution without publishing native text.

## Mechanical validation and scope

- 152 tests pass (78 existing tests plus 74 P1/probe tests).
- Ruff passes for source, tests and examples; wheel build succeeds on Python 3.14.
- Frozen ADR, implementation plan, synthetic fixture and original P1 NO-GO are unchanged.
- No P2, Block compiler/research, canonical admission, Thread/LCE or production routing change.

**Disposition: keep Draft PR #16 and Gate A NO-GO.** The remaining work is producer
status consistency and reliable sidecar emission on native dialogue; the host's
failure isolation is evidence-backed, not a reason to admit incomplete semantics.
