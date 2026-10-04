Point is turn-bounded, not utterance-bounded.

A Point is a concise projection of the Body's context-conditioned understanding at this turn, not a fresh semantic extraction from the current utterance alone.

# P1-R1 — Contextual Understanding Snapshot Reset

Pre-registration date: 2026-10-04. Baseline Draft PR #16 head:
`80b907789e225d69c7c9022f5bcf94d6af3f320b`.
This is an experiment-only task-definition reset. Old P1.5 and P1.5-SO evidence,
gold and NO-GO/NO_CONCLUSION outcomes remain unchanged. The old gold is historical
comparison, not the semantic authority for this task. Production is unchanged.

## Gold and Point/Block boundary

`examples/fixtures/contextual_snapshot_gold_v1.json` has 28 original-prefix
entries and six separately scored raw-context probes. It is curated before any
inference from complete original raw prefixes, not from model outputs or old
semantic gold. Every entry binds prefix and current-turn hashes. Gold is never
sent to Body and cannot be edited after observations. Source continuation after
the current turn is never available to the judge/model as current context.

Required, optional and forbidden context are distinguished. Missing a required
resolved object, research actor, distinction or constraint is CONTEXTUAL_POINT_WRONG,
not automatically UNCERTAIN. Missing/rejected Point remains UNCERTAIN. Review
allows faithful paraphrases; it is not an exact-string or keyword matching judge.
Normal responses are independently judged against raw context, never Point
authority. External technical/financial correctness is outside this semantic audit.

Point may carry relevant resolved references, actors, scope, current corrections
and distinctions already used to understand this turn. It does not compile final
historical closure, exhaustive relations or whole-topic summaries. Block still
rereads relevant live raw turns plus available hints for context-complete compilation.
This visible behavioral audit cannot certify access to hidden Body state or CoT
faithfulness merely because Point and reply agree.

## One-variable request isolation

Both arms use response FIRST, then hidden frame; unchanged schema, aliases,
host-derived status and failure/gap policy. A uses exact P1.5 semantic instruction;
B uses contextual task below. Every original A request is byte-equivalent to
P1.5-SO A for that input. All raw prefixes and activated payloads are copied from
the same previously frozen thinking-ON inputs. Neither arm's generated reply or
Points can enter a later request. Baseline hint imperfections remain identical.

The six new probes use fixed raw prefixes and empty activated Point hints in both
arms: context-only unique pronoun, assistant yesterday investigator/model object,
v1-only storage limitation, elliptic calculation correction, former/latter
distinction, and recipient change with an established approval constraint.
Their gold and raw case pack are committed before any calls. They are not folded
into the original 28-turn metrics.

The frozen shared schema example still has its old meaning placeholder, identical
in both arms. B explicitly identifies example values as placeholders and uses
the new task for actual meaning. No activated payload or schema template is
rewritten to make an arm look better.

Settings: deepseek-flash, thinking ON, temperature0.2, completion ceiling16384,
one inference per observation, no retry, repair, tool execution, JSON format or
judge model. Logprobs and provider route are identical. Normalize only semantic
clause; require otherwise identical complete request bytes for every pair.
Record actual request, normalized pair, prefix, context, task and order hashes.

## Schedule, review and predeclared verdict

Three native turns (`native_model_correction/1`, `native_execution_correction/0`,
`native_execution_correction/2`) ×8 repetitions per arm =48 calls. Then all original
28 turns paired once =56 calls, regardless of first-stage result. Finally six
context dependency probes paired once =12 separate calls. Total116, original104
subset stays separately comparable with P1.5-SO. Rotate case order and alternate
arm submission order, at most two independent requests per frozen pair.

Manual task-arm-blinded review, with labels locked before aggregate unblinding.
Main Point classes: CONTEXTUAL_POINT_CORRECT / CONTEXTUAL_POINT_WRONG / UNCERTAIN.
Separately score actor, speech act, resolved refs, predicate target, scope,
required context retention, over-expansion and unrelated-history leakage.
Known component judgments survive a missing other component. Preserve failures,
all uncertainties and original denominators. The prior six joint reply/Point
categories are used for agreement, with contextual Point judgments; aligned wrong
outputs never count as success. Normal reply is not used to relabel Point.

- test: Two-sided exact paired McNemar, alpha=.05. Contextual Point-correct is primary in repeated24 pairs and independently in full28 pairs. Report paired unknown bounds and marginal original denominators; do not score UNCERTAIN as success or silently as zero. Require robust significance/direction under all admissible unknown assignments for SUPPORT. Probes6 pairs are separate descriptive challenge pack, not added to28/24.
- support: B contextual Point-correct significantly exceeds A in BOTH repeated and full paired stages under all unknown assignments; B lower-bound normal-response correct >= A upper bound in both stages; no increased definite unrelated-history/mini-Block or invented historical relation/over-expansion counts in full/repeated; no worse response/Point coverage or transport compliance. Probe resolved-context axes reported as mechanism evidence only, never substitute for primary conditions.
- reject: Robust significant normal-response regression in either original stage; OR B leakage/mini-Block/invented historical relation turns >=3 in either original stage, at least3 more than A, and at least2 distinct source turns overall. These are predeclared operationalizations of substantial leakage. A significant Point regression alone with no history-leak or response regression is described but does not automatically imply definition/architecture rejection.
- otherwise: NO_CONCLUSION. No P1 closure, protocol adoption, schema extension or automatic P2 entry even for SUPPORT.

No verdict is old Gate A. Even SUPPORT requires a subsequent definition/boundary audit; P1 stays open and P2 closed.

## Locked hashes and semantic instruction

- gold_sha256: `ac0855f7f6b9b71a45c5f45882fa5857df4a467a8b92738f97a489774a2891f3`.
- pack_sha256: `661e4a266f31723fddd3129188f3263536c28e0a9587d80ff667adb3f9cdc89f`.
- frozen_inputs_sha256: `f70d210d3950a3985878d4109527c5306c1d13e34a5e787ad087bb8f145bd0cf`.
- schedule_sha256: `35dc33678005e9e22bfdc722a6af888809ad8e02f2dd5f63f6655d4e3fe7d65b`.
- Semantic A SHA256: `b8cc20f54e28c2c946518310e702e576538bbfe487de2440b1e79bdfa6e51585`.
- Semantic B SHA256: `440f4d9773fae06bf7a4c07d6993d0563919ea15c572fdc8e0fcd975a222035f`.
- Response-first order SHA256: `88042a98e7644bb5e7ec2cb103ec73969dab959a25842907b78eee9f3f6e3689`.

Exact B task clause:

```text
Emit a concise snapshot of the contextual understanding you are currently using to answer this user turn. This understanding may and should use relevant prior dialogue. Preserve resolved references, actors, scope, distinctions, corrections, qualifications and prior constraints when they are actually part of how you understand the current turn. Do not reinterpret the current utterance in isolation. Do not merely summarize the current sentence, and do not reconstruct unrelated conversation history. Record what this user turn means in its actual conversational context. If the context still permits multiple materially different interpretations, preserve that ambiguity in unresolved_refs. The Point is a turn-time contextual understanding snapshot, not a context-free proposition and not a whole-dialogue summary. Do not turn your own advice, new claims or answer into user commitments. Do not invent context or compile a final whole-topic semantic closure. Default ONE Point for the whole turn-time understanding. Multiple Points are allowed ONLY for truly independent semantic lines in the same turn that may develop separately. Unknown specific identity, missing implementation details or generic discourse objects are not by themselves materially different interpretations. Do not output status; the host derives it solely from unresolved_refs. Local understanding is not context-complete or canonical. Use slots 0,1,... in order; the host assigns all stable identities. context_refs is a list of activated aliases ONLY for context objects explicitly pointed to by the current turn in its dialogue. If the pointer is uncertain or merely topically related, leave it out. No relation text, host identity, authority, extra fields or markdown in tool arguments. At most 4 Points, meaning <=2048 characters, context/unresolved refs <=8 each, ref <=256 characters, Point array <=12288 UTF-8 bytes. Empty points only when no semantic content is present. Schema example values below are placeholders; use the contextual task defined here for meaning.
```

Results will be appended after locked observations and manual reviews.

Gold/pack were committed before inference in `7fd1ff593fe241f0ab8b6cdfbfb6419a0d4827b1`.
Predeclared manifest SHA256: `4a79707969d965aea48e079ef64b6d02495e2326cf37502bda450b80c8dcb51d`.

Committed gold Git blob SHA256: `b949de3f372ef9e74930f669a4930b3a839ce2e810bc876fbb67ee5a609ffa41`.
Windows working-file CRLF and Git LF bytes have separate hashes; parsed gold content is identical. No gold labels changed.

## Observed results — 2026-10-04

**Verdict: `NO_CONCLUSION`.** B does not meet the preregistered contextual
fidelity improvement and normal-response/coverage safeguards. The operational
rejection thresholds are not met either. This is neither SUPPORT nor an
architecture-level rejection. P1 remains open; P2 remains closed.

All **116** planned inferences ran once: original repeats48 + replay56 +
separate context probes12. All reported `stop`; all116 normal responses survive.
There were **26 coverage gaps**, every raw source retained, zero repair/retry
calls, zero admitted Points on rejected frames. Frame transport, response order,
schema, provider settings and original raw fixtures remain unchanged.

### Lineage and one-variable verification

Baseline/current-at-start PR head: `80b907789e225d69c7c9022f5bcf94d6af3f320b`.
Experiment producer locked before calls:
`04d3352c1e83d1995fb19f9ddac7b22598848650`. Gold/pack commit
`7fd1ff593fe241f0ab8b6cdfbfb6419a0d4827b1`; run-plan preregistration
`b91f3e6`. Production semantic producer is unchanged from baseline; no `src`
diff in this task. Final result publication adds report and sanitized evidence.

All **58 A/B pairs** pass complete actual request-byte equality after replacing
only the semantic task clause. Prefix, activated payload, host aliases, schema
example, order directive and all provider parameters match within each pair.
A original28 requests match previous SO A bytes exactly. Request bodies were
hashed at request construction and checked at the HTTP adapter before sending.
No generated reply or Point enters another request. Gold was not sent to Body.

- Synthetic raw fixture SHA256: `c6b0cbd1b6bba69af9c7ec97ab92fce499a11cd3a5659428374234de77a592dd`.
- Private native raw fixture SHA256: `1ec7a666fabd07ffc56e22d5c9e3b50473f2f7de2f4dcaac902cfdf94f37d560`.
- Frozen inputs SHA256: `f70d210d3950a3985878d4109527c5306c1d13e34a5e787ad087bb8f145bd0cf`.
- Schedule SHA256: `35dc33678005e9e22bfdc722a6af888809ad8e02f2dd5f63f6655d4e3fe7d65b`.
- Locked116-review SHA256: `7b3d30a0190c8f67651fd63751e96239c194fc593e4d0e52cc5f95a0229d64be`.
- repeated receipt SHA256: `b146275508641fe859783917631a66b83eee3cd0546a3f8bc20c595c2f951928`.
- full receipt SHA256: `bbceb57520b2e5b78ba98709dbe8af042dce7f0385a9b12350ceaaee1e9923a8`.
- probes receipt SHA256: `13f1559460fbaee077122da8e78dbbc4c5e7570aa5c72a78fb2ea5c265460088`.
- Attempt log SHA256: `675ef661b92204a33b6e76705b78ccf70567c5c8c94928351442b46762f52e97`.

Gold Windows CRLF bytes and committed LF blob have distinct recorded hashes;
their parsed contents match. Before inference, preflight detected that byte-line
difference; lineage metadata was corrected and committed, with zero HTTP attempts
and no label changes. That preflight is not an inference retry. No gold/schema/
harness changes occurred while or after the116 inference run.

Exact **A task and common response-first order clauses** are recorded in
`outputs/p1-contextual-snapshot-evidence.json` → `baseline.semantic_clauses.A`
and `baseline.order_clause`. B is printed above. All three clause hashes and
the five per-request hashes are included in every observation.

The evidence includes all116 redacted visible receipts, all failures/rejected
frames, decoded replies, admitted Points, usage, attempts and locked reviews.
Native full prefixes, credentials, provider IDs and hidden reasoning prose remain
private. Raw receipt hashes refer to original private bytes, not redacted bytes.
The manual-review hash can be reconstructed from evidence using its documented
serialization. Nothing is salvaged from rejected Points.

### Three native turns × eight repetitions

C/W/U below = contextual Point correct/wrong/uncertain. These are the new
predeclared classes; old current-utterance gold does not grade either arm.

| Frozen turn | Arm | Point C | Point W | Point U | Reply C | Reply W | Reply U |
|---|---|---:|---:|---:|---:|---:|---:|
| native_model_correction/1 | A | 1 | 6 | 1 | 7 | 1 | 0 |
| native_model_correction/1 | B | 0 | 5 | 3 | 7 | 1 | 0 |
| native_execution_correction/0 | A | 0 | 7 | 1 | 6 | 2 | 0 |
| native_execution_correction/0 | B | 0 | 0 | 8 | 7 | 1 | 0 |
| native_execution_correction/2 | A | 0 | 3 | 5 | 8 | 0 | 0 |
| native_execution_correction/2 | B | 0 | 5 | 3 | 6 | 2 | 0 |
| Total24 | A | 1 | 16 | 7 | 21 | 3 | 0 |
| Total24 | B | 0 | 10 | 14 | 20 | 4 | 0 |

| Joint category | Repeats A | Repeats B | Full A | Full B | Probes A | Probes B |
|---|---:|---:|---:|---:|---:|---:|
| BOTH_CORRECT | 1 | 0 | 11 | 11 | 2 | 2 |
| POINT_WRONG_RESPONSE_CORRECT | 13 | 8 | 11 | 12 | 4 | 2 |
| POINT_CORRECT_RESPONSE_WRONG | 0 | 0 | 0 | 0 | 0 | 0 |
| BOTH_WRONG_SAME | 0 | 0 | 0 | 1 | 0 | 0 |
| BOTH_WRONG_DIFFERENT | 3 | 2 | 0 | 0 | 0 | 0 |
| UNCERTAIN | 7 | 14 | 6 | 4 | 0 | 2 |

The G1 question has **zero admitted B Points in8 repeats**: six frames rejected
for unactivated literal refs (`G1`, `D1`), two missing frames. That cannot be
called a semantic repair. In actor repeats, admitted A has6 omitted researcher
actors and1 correct; admitted B has5 omissions and0 correct, with3 gaps.
No accepted repeated Point explicitly flips researcher to user, but omission is
wrong under new gold. Normal replies independently have one actor error per arm.

For historical-target repeats, B more often names prior peak-day pricing, but
its5 admitted Points all fail another obligation: execution choices become DEFER,
unexpressed conclusions appear, or old requests become current commitments.
A has5 genuinely unclear target readings, retained as U rather than forced wrong
or repaired from the reply. `R005` (B) produces three Points, including old rename
and peak-price state: unrelated-history/mini-Block leakage and fragmentation.

### Paired primary and response safeguards

Primary = contextual Point-correct, not mismatch reduction or old Gate A.
Unknowns remain possible0/1 assignments for bounds, never counted as successes
or silently failures. A−B is a count difference, with B improvement negative.

| Stage | Pairs | Definite Point pairs | A-only correct | B-only correct | Complete-case p | A−B Point bounds | Exact p bounds |
|---|---:|---:|---:|---:|---:|---:|
| repeated | 24 | 5 | 1 | 0 | 1 | -13 to 8 | 0.000976562–1 |
| full | 28 | 22 | 8 | 9 | 1 | -4 to 5 | 0.404873–1 |
| probes | 6 | 4 | 0 | 1 | 1 | -2 to 0 | 0.5–1 |

Repeats Point C/W/U is **A1/16/7 vs B0/10/14**; full is **A11/12/5 vs
B11/13/4**. Repeats B−A correct bounds are −8 to+13; full −5 to+4.
Neither stage demonstrates robust significant benefit. Full has9 definite A-wrong
to B-correct transitions, but8 A-correct to B-wrong transitions. Many B failures
retain useful context but still violate ambiguity, alias or expansion boundaries.

Replies: repeated **A21/3/0 vs B20/4/0**, full **A26/1/1 vs B26/2/0**.
Repeated response A−B=+1 (2 A-only /1 B-only, p=1); full A−B0 to+1
(27 definite pairs,1/1 discordants, p=1 under every unknown assignment).
The preregistered non-degradation safeguard is not met, but significant response
regression is not demonstrated. Reply coverage alone is not fidelity.

Confirmed PW/RC: repeats **A13/24 vs B8/24**, full **A11/28 vs B11/28**.
Repeat mismatch endpoint unknowns are7/13 respectively; whole-pair mismatch
bounds A−B −7 to+12. Full mismatch bounds −4 to+4. Lower observed repeat
mismatch counts are strongly censored by B missing/rejected sidecars. There are
zero confirmed A-mismatch→B-aligned-wrong transitions; that zero does not prove
the problem is absent. Full `F013` (B) has both outputs wrongly resolving it to
B/cloud, and is explicitly BOTH_WRONG_SAME, never an agreement success.

### Original28 paired replay: every turn

P columns are new contextual C/W/U, R columns independent reply C/W/U.
No normal reply is substituted as Point authority. Failure reasons and axes
for each output are in evidence.

| Frozen turn | A Point | B Point | A reply | B reply |
|---|---|---|---|---|
| repeated_correction/0 | C | W | C | C |
| repeated_correction/1 | W | W | C | C |
| repeated_correction/2 | W | C | C | C |
| delayed_qualification/0 | C | W | C | C |
| delayed_qualification/1 | C | W | C | C |
| delayed_qualification/2 | C | W | C | C |
| scope_distinction/0 | C | W | C | C |
| scope_distinction/1 | W | C | C | C |
| scope_distinction/2 | W | C | C | C |
| cross_turn_reference/0 | C | W | C | C |
| cross_turn_reference/1 | W | W | U | W |
| cross_turn_reference/2 | W | C | C | C |
| partial_plan_update/0 | C | C | C | C |
| partial_plan_update/1 | W | C | C | C |
| partial_plan_update/2 | W | C | C | C |
| diversion_and_return/0 | C | C | C | C |
| diversion_and_return/1 | C | W | C | C |
| diversion_and_return/2 | W | C | C | C |
| diversion_and_return/3 | W | C | C | C |
| certainty_and_withdrawal/0 | C | W | C | C |
| certainty_and_withdrawal/1 | W | C | C | C |
| certainty_and_withdrawal/2 | W | W | C | C |
| native_model_correction/0 | U | U | C | C |
| native_model_correction/1 | U | W | C | C |
| native_execution_correction/0 | U | U | C | W |
| native_execution_correction/1 | C | U | C | C |
| native_execution_correction/2 | U | W | C | C |
| native_rhetorical_correction/0 | U | U | W | C |

### Contextual fidelity axes

Cells are C/W/U on all original observations; missing Point makes its axes U.
A genuinely unclear target reading leaves other known axes available. Actor W
includes omission of a required actor; it is not all explicit actor reversal.
Resolved-reference W includes unnecessary DEFER and failure to retain a required
resolved object, so this column must not be interpreted as wrong referent alone.

| Point axis | Repeats A | Repeats B | Full A | Full B |
|---|---:|---:|---:|---:|
| actor | 16/6/2 | 5/5/14 | 24/0/4 | 23/1/4 |
| speech_act | 22/0/2 | 10/0/14 | 24/0/4 | 24/0/4 |
| resolved_references | 12/10/2 | 2/8/14 | 16/8/4 | 11/13/4 |
| predicate_target | 14/2/8 | 9/1/14 | 22/1/5 | 23/1/4 |
| scope | 22/0/2 | 7/3/14 | 21/3/4 | 22/2/4 |
| contextual_constraints | 15/7/2 | 9/1/14 | 13/11/4 | 24/0/4 |
| over_expansion | 22/0/2 | 4/6/14 | 24/0/4 | 21/3/4 |
| unrelated_history | 22/0/2 | 9/1/14 | 24/0/4 | 23/1/4 |

### Contract, gaps and separate semantic failure counts

Each semantic count is definite admitted-output turns, not individual Points;
flags can overlap. Rejected raw proposals are preserved but remain U in primary.
Missing-frame counts include no sidecar, not illegal JSON. Both arms had zero
illegal JSON, zero network/response failures; all receipt finish reasons stop.

| Metric | Repeats A | Repeats B | Full A | Full B | Probes A | Probes B |
|---|---:|---:|---:|---:|---:|---:|
| Normal responses | 24 | 24 | 28 | 28 | 6 | 6 |
| Admitted sidecar turns | 22 | 10 | 24 | 24 | 6 | 4 |
| Contract failures including misses | 2 | 14 | 4 | 4 | 0 | 2 |
| Coverage gaps, raw retained | 2 | 14 | 4 | 4 | 0 | 2 |
| Replies preserved on gap | 2 | 14 | 4 | 4 | 0 | 2 |
| Total Points | 22 | 12 | 26 | 24 | 6 | 4 |
| Single-Point turns | 22 | 9 | 22 | 24 | 6 | 4 |
| Multi-Point turns | 0 | 1 | 2 | 0 | 0 | 0 |
| Explicit wrong actor | 0 | 0 | 0 | 0 | 0 | 0 |
| Required actor omitted | 6 | 5 | 0 | 1 | 1 | 0 |
| Question→fact | 0 | 0 | 0 | 0 | 0 | 0 |
| Wrong resolved referent | 0 | 0 | 0 | 1 | 0 | 0 |
| Unnecessary DEFER | 7 | 8 | 0 | 12 | 0 | 2 |
| Historical target over-expansion | 1 | 1 | 0 | 0 | 0 | 0 |
| Missing required context | 13 | 5 | 11 | 1 | 4 | 0 |
| Unrelated-history leakage | 0 | 1 | 0 | 1 | 0 | 0 |
| Mini-Block leakage | 0 | 1 | 0 | 0 | 0 | 0 |
| Invented historical relation | 0 | 0 | 0 | 1 | 0 | 0 |
| Over-expansion (other) | 0 | 5 | 0 | 3 | 0 | 0 |
| Wrong accepted context_ref | 0 | 1 | 1 | 2 | 0 | 0 |

Full ambiguity control (`cross_turn_reference/1`) fails **both** arms: A accepts
bare it with empty refs; B explicitly guesses B/cloud with empty refs. Both falsely
RESOLVE. B also adds a comparison distinction. A normal reply reading is uncertain
because it mixes assertion and a confirmation question; B reply definitively
commits cloud. These are retained, not hidden by a wrong-ref count of zero for A.

New contextual gold has **26 unambiguous turns**,1 required ambiguous turn and
1 optional material pricing ambiguity in full replay. Full unnecessary DEFER is
A0/26, B12/26, with4/3 missing unambiguous Points respectively. These are observed
counts, not a complete boundary certification. Do not recycle the old clear24
denominator or old Gate A label. Fewer missing-context flags (11 A vs1 B) do show
descriptive contextual-content gain, but do not override status/coverage failures.

Accepted refs: A1/B2 wrong full, B1 repeated. B `F029` links SQLite into the
separate meeting request; B `F035` treats prior user p0 as the assistant personal-use
misunderstanding. B `R005` revives old group renaming with a pointer not explicitly
used in current correction. Two A full multi-Point turns contain independent
A/B plan lines and rename/analysis lines; they are not judged wrong just by count.

Leakage/invented-relation union B is1 repeated and2 full (A0 in both), below
the predeclared rejection threshold of≥3 in an original stage and≥3 more than A.
It nevertheless breaks SUPPORT safeguards. Over-expansion union A/B is1/6 repeats
and0/3 full. The result therefore stays NO_CONCLUSION under the locked rules.

Contract details: repeated A2 missing frames; B8 missing+6 unactivated target
rejections. Full A2 missing+2 unactivated; B1 missing+3 unactivated. Probe B2
unactivated, A0. Literal names such as `G1`, `Jev`, `青舟`, `方案B` are not
activated aliases and are correctly rejected; schema/host mapping is unchanged.

### Six context-dependent probes (separate, not pooled into28)

| Probe | A Point | B Point | A reply | B reply |
|---|---|---|---|---|
| probe_unique_reference | C | U | C | C |
| probe_actor_object | W | U | C | C |
| probe_delayed_scope | W | W | C | C |
| probe_elliptic_correction | W | W | C | C |
| probe_established_distinction | W | C | C | C |
| probe_contextual_only | C | C | C | C |

Probe totals: Point A2/4/0 vs B2/2/2; replies6/0/0 both. B restores elliptic
calculation and SQLite/v1 context, but marks unknown future handling as DEFER.
The former/latter distinction probe improves A-wrong→B-correct. B unique-reference
and actor/object frames fail alias validation; neither is salvaged into a semantic
success. A does resolve scheme B and current formal demonstration recipient in
some probes despite its sentence-restricted instruction; task A is not uniformly
context-blind. A actor/object Point omits assistant investigator while its reply
owns yesterday investigation. No universal representational ceiling follows.

### Token/reasoning/latency observations

Median (min–max); all provider usage retained, hidden reasoning prose omitted.
Frame interval uses visible-token byte boundaries/EOS interval, includes rejected
emitted frames, excludes missing frames. It is not total Point incremental cost.

| Stage / arm | Visible completion | Reasoning | Latency seconds | Frame observed range / n |
|---|---:|---:|---:|---:|
| repeated / A | 421.5 (148–735) | 1472.5 (463–10449) | 9.046 (5.498–51.465) | 52–125 / 22 |
| repeated / B | 505.5 (354–789) | 1195 (499–10125) | 9.2435 (5.105–51.962) | 102–295 / 16 |
| full / A | 93.5 (53–621) | 899 (80–6126) | 6.8675 (1.198–27.563) | 43–120 / 26 |
| full / B | 189.5 (85–811) | 1423.5 (219–4102) | 8.908 (2.221–23.143) | 51–282 / 27 |
| probes / A | 70 (55–102) | 1114 (604–3038) | 6.487 (3.986–14.41) | 45–59 / 6 |
| probes / B | 120.5 (103–180) | 1228 (790–2716) | 7.2005 (5.234–14.319) | 72–120 / 6 |

B clause is91 provider prompt tokens shorter in all pairs. Despite that, full
B median visible output is189.5 vs93.5 A, reasoning1423.5 vs899, latency8.908s
vs6.8675s. Cache-hit means differ (A937.143 vsB416 full; probes both0); replies
and hints also affect measured generation length. Repeats reasoning median is
lower for B, so no stable total-cost benefit can be concluded. No matched normal
Body-only baseline exists; do not certify incremental reasoning overhead.

### Reproduction, checks and interpretation

Call experiment-only `freeze_reset_inputs(saved_SO_frozen_inputs, probe_pack)`,
then `verify_pair`, `make_request`, `run_plan(frozen,schedule,infer_once,path,gold)`.
Use fresh receipt paths; overwrite/source/gold drift stops before provider calls.
The callback must send exactly request bytes and record their hash and usage.
Schedule is repeated48/full56/probes12, pairs alternating arm submission order,
maximum two concurrent requests. Private original prefixes and saved baseline
hints are required for exact original replay, never regenerated from new outputs.

At locked producer: **177 pytest passed**, Ruff over `src tests examples` passed,
wheel build passed and packaged `mr_mem/point_sidecar_v2.py` verified. Result
change is documentation/evidence only; exact final-head GitHub CI is verified
after publication, linked in PR and local deliverable. All original fixtures,
gold, old diagnostics/evidence, Point schema, host status/gaps and production
`src/mr_mem/point_sidecar_v2.py` are unchanged against80b9077.

One manual evaluator hid arm labels and task clauses until116 review labels were
hashed. Output verbosity/style can reveal the intervention, so this is not full
reviewer blindness or an independent multi-rater certificate. The enriched
repeated samples and soft-frame gaps limit semantic causal conclusions. Do not
grade B using old gold, select a lucky subset, or turn reply agreement into
fidelity. No raw rejected Point is used to rewrite a reply or obtain a new label.

This experiment shows context-bound snapshot text can retain information omitted
under utterance restriction, but the tested B instruction does not demonstrate
simpler reliable Point production overall. It also broadens ambiguity and alias
behavior and occasionally leaks history. No next implementation is authorized
by this result: return for definition/boundary audit, preserve P1 open/P2 closed
and PR16 Draft. No production adoption, Block work or new semantic fields.
