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
