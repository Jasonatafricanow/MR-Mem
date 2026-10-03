# Semantic Compilation V2 — implementation plan

- Status: IMPLEMENTATION PLAN
- Date: 2026-10-03
- Architecture authority: ADR-0003
- Scope: first make Point production correct and cheap, then compile context-complete Blocks inside the live conversation window
- Explicit non-goals: redesign LCE algorithms, Thread scoring, RAG, AML, Hot Start, UI, CJK recall, new small semantic LLM

## 0. Goal

Deliver one production semantic chain:

~~~text
normal Body turn
  -> response + Point sidecar
  -> active semantic buffer
  -> low-frequency live-window Block compile
  -> canonical SemanticBlock
  -> Thread / block-bound Point Cloud
  -> LCE
~~~

The work is complete only when multi-turn clarification/correction examples produce a Block that can be consumed without rereading the original conversation and without changing the user's intended meaning.

The success metric is semantic fidelity, not token compression by itself.

## 1. Phase A — SemanticPoint V2 protocol

### Objective

Make every normal Body inference emit a bounded turn-local semantic sidecar without an extra inference.

### Proposed contract

~~~text
SemanticPointV2
- point_id
- interaction_id
- turn_id
- meaning
- status: RESOLVED | DEFER
- context_links[]
    - target_kind: POINT | BLOCK | MEMORY
    - target_id
    - relation: open text
- polarity: optional
- epistemic_status: optional
- temporal_scope: optional
- temporal_expression: optional
- unresolved_refs[]
~~~

System-bound, never Body-authored:
- Scope
- SourceRef
- native revision/fingerprint
- occurred_at
- known_at
- canonical identity

### Protocol rules

1. meaning describes what this turn currently means in context; it does not have to restate the entire conversation.
2. Do not require a Point to become context-free.
3. Do not split one turn into artificial atomic propositions unless the Body naturally needs multiple Points to represent materially different local meanings.
4. context_links preserve meaningful dependence on earlier Points/Blocks.
5. relation text is open-ended; no finite world-semantic taxonomy.
6. unresolved reference or missing context is explicit. Do not hallucinate resolution.
7. Point generation is a sibling structured output of the response.

### Experiment

Use real multi-turn dialogue fixtures, especially:
- repeated correction;
- "not X, I mean Y";
- qualification added two or three turns later;
- apparent contradiction resolved by scope distinction;
- pronoun/reference to an earlier concept;
- user changes one part of a plan while keeping the rest;
- same topic returning after a short diversion.

For each turn save:
- raw turn;
- assistant response;
- emitted Point;
- context links;
- model-call count;
- additional output tokens.

### Gate A

GO only if:
- no extra semantic model call;
- Point meaning remains locally faithful;
- no requirement that every Point stand alone;
- correction/reference links are emitted reliably enough to help later Block compilation;
- bounded sidecar overhead is acceptable.

Do not touch Block production code before the protocol fixture suite is frozen.

## 2. Phase B — ActiveSemanticBuffer

### Objective

Keep the uncompiled semantic frontier aligned with the live conversation window.

### State

~~~text
ActiveSemanticBuffer
- ordered Point receipts
- source turn refs
- current uncompiled span start
- last accepted Block boundary
- unresolved refs
- candidate boundary signals
- live-window residency / eviction risk
~~~

### Requirements

- bounded by turns and/or tokens;
- not a second transcript database;
- exact turn bodies remain host-owned;
- buffer keeps references, not a duplicate authoritative raw history;
- restart can reconstruct from native source + Point receipts;
- accepted Blocks remove/retire covered frontier state;
- unresolved Points stay pending rather than being silently dropped.

### Gate B

Demonstrate:
- restart reconstruction;
- no duplicate Block coverage;
- no gap when turns cross a block boundary;
- pending material is forced to compile or DEFER before context eviction.

## 3. Phase C — Block compiler protocol

### Objective

Compile a bounded cross-turn span into one or more context-complete SemanticBlocks while the original turns are still in the active window.

### Compiler call

This is a separate low-frequency semantic operation.

Input:

~~~text
- current live conversation prefix
- exact candidate span
- Point sidecars for the span
- last adjacent accepted Block when required
- explicitly activated canonical context for unresolved refs
- stable compiler instruction/version
~~~

Output:

~~~text
SemanticBlockProposal
- block_local_id
- member_point_ids[]
- source_turn_refs[]
- compiled_meaning
- internal_relations[]
- external_context_refs[]
- unresolved_context[]
- closure_status: CLOSED | DEFER
~~~

### What compiled_meaning must do

It must represent what the whole span means after incorporating:
- correction;
- qualification;
- clarification;
- local contradiction resolution;
- scope distinction;
- references;
- changed decisions;
- retained unchanged assumptions.

It must not:
- concatenate Point texts;
- simply summarize the latest turn;
- discard earlier statements that are needed to understand why the latest statement means what it means;
- turn uncertain/conditional claims into unconditional facts;
- flatten different scopes into a false contradiction.

### Cache strategy

The compiler should be invoked immediately after a turn while the same conversation prefix is still cacheable.

Preferred request shape:

~~~text
same stable system prefix
+ same conversation prefix through current turn
+ hidden stable semantic-compiler suffix
~~~

Operational goals:
- maximize KV/prompt-prefix cache hit;
- compile only the candidate span, not the entire session;
- never wait for broad RAG reconstruction if the span is still hot;
- preserve accepted compiler receipts so downstream retry never reruns semantic compilation.

Measure:
- cache-hit/prompt-cached tokens where provider exposes them;
- uncached input tokens;
- compiler output tokens;
- wall-clock latency;
- compile frequency per 100 turns.

## 4. Phase D — Block boundary scheduling

### Objective

Call the expensive contextual compiler only when useful, without losing a span before eviction.

### Signals

Scheduling signals may include:
- topic/semantic boundary candidate;
- explicit conclusion/decision;
- correction chain appears settled;
- unresolved reference becomes resolved;
- active span grows beyond safety bound;
- context compaction/eviction risk.

These signals only schedule a compiler call. They do not determine membership.

### Initial scheduler

Use a conservative hybrid:

~~~text
compile if:
  eviction_risk
  OR uncompiled_span >= bounded_max
  OR Body sidecar marks boundary_candidate
  OR explicit closure candidate
~~~

The Body may emit an operational compile hint, but it is not canonical semantic authority.

Example:

~~~text
compile_hint:
  NONE | BOUNDARY_CANDIDATE | CLOSURE_CANDIDATE
~~~

This enum is operational scheduling metadata, not a semantic ontology.

### Gate D

Track:
- unnecessary compile rate;
- late compile / eviction failures;
- average turns per Block;
- DEFER frequency;
- reopened/superseded Block frequency.

Do not tune for lowest call count at the expense of semantic loss.

## 5. Phase E — MR-Mem admission V2

### Objective

Make deterministic code verify and freeze the semantic result rather than invent it.

### Admission checks

- all member Point IDs exist;
- Points belong to authorized source turns;
- exact SourceRef revisions still match;
- source span ordering is valid;
- external context refs are activated and same-scope;
- CLOSED is rejected if unresolved_context remains;
- Body cannot set system authority fields;
- block compiler version is stored;
- accepted proposal replay is deterministic.

### Identity

Do not silently reinterpret existing immutable V1 memory IDs.

Recommended migration rule:
- SemanticBlock V2 gets a new semantic/compiler identity namespace;
- old V1 accepted compilations remain auditable;
- where old frozen material can be deterministically mapped without new semantic judgment, migration may replay;
- where contextual semantic judgment is required, use live/historical semantic compiler and create a new V2 identity.

## 6. Phase F — fidelity evaluation

### Objective

Prove that Block solves the original problem: decontextualization.

Do not reuse Issue #14's "candidate vs full Delta compression" as the primary test.

### Core fixture style

Each fixture is a dialogue, not a pre-made Point graph.

Example pattern:

~~~text
T1: initial statement
T2: assistant/user interpretation
T3: correction
T4: qualification
T5: final clarification
~~~

Then test:

1. What Points were emitted per turn?
2. What span did the compiler choose?
3. What Block was produced?
4. Can the Block answer the intended semantic questions without raw turns?
5. Does removing one required member change the answer?
6. Does retrieving one raw Point alone create the exact misreading Block was designed to prevent?

### Required fixture groups

- correction across 3+ turns;
- scope distinction that looks contradictory when split;
- partial plan update;
- uncertain -> confirmed transition;
- confirmed -> withdrawn transition;
- cross-turn reference;
- delayed clarification;
- topic diversion and return;
- two simultaneous semantic lines in overlapping turns;
- one-turn self-contained Block;
- span that must stay DEFER because closure never arrives.

### Independent evaluator

Use a fixed external/top model only as an evaluator, not as the production compiler.

For each fixture compare:
- raw conversation ground truth;
- naked latest Point;
- all Points without Block;
- compiled Block.

Evaluate:
- intended meaning preserved;
- false contradiction;
- lost qualification;
- incorrect certainty;
- incorrect temporal interpretation;
- incorrect partial supersession;
- unresolved reference hallucination.

### Gate F

Do not connect production Thread/LCE until Block materially outperforms naked Points on the decontextualization cases.

## 7. Phase G — projections after Block passes

Only after Gate F:

### Thread / Path A

~~~text
accepted Block
 -> incremental Thread update
 -> mature Thread
 -> LCE
~~~

Thread receives complete semantic material and should not reread Raw to repair meaning.

### Point Cloud / Path B

Point Cloud may retain member Point vectors, but projected nodes carry:
- block_id;
- member role/support;
- relevant internal relation;
- occurred/known time;
- lifecycle.

A naked turn Point that has not yet been compiled into an accepted Block must not be treated as a stable longitudinal assertion.

Path B then runs the existing structural/trajectory work over block-bound projections. No trajectory-algorithm redesign is part of this plan.

## 8. Production rollout order

~~~text
P0 docs/contract only
P1 Point V2 sidecar + fixtures
P2 ActiveSemanticBuffer
P3 live-window Block compiler experiment
P4 fidelity evaluator
P5 Block V2 admission/replay
P6 Thread consumer
P7 block-bound Point Cloud consumer
P8 production flag + real-data shadow run
P9 enable normal online route
~~~

Every step gets its own rollback point.

## 9. Real-data shadow run

Before production enablement, shadow on representative conversations.

Collect:
- Point output rate;
- Point sidecar token overhead;
- Block compile call frequency;
- cache hit / cached prompt tokens;
- Block span length;
- DEFER rate;
- semantic correction/supersession rate;
- context-eviction misses;
- false Block splits;
- false Block merges;
- Thread/LCE downstream differences.

Manual review must focus on whether a Block can safely leave the raw conversation context.

## 10. Hard stop rules

Do not add adjacent work while implementing this plan.

Specifically do not:
- redesign LCE;
- add another semantic provider;
- optimize CJK lexical retrieval;
- change AML;
- redesign Hot Start;
- add a new memory ontology;
- reopen MR vs MR-Mem ownership;
- optimize token compression before fidelity works.

If a downstream component fails because it needs information the accepted Block should contain, first determine whether Block compilation failed. Do not patch the downstream component to reinterpret raw turns.

## 11. Completion definition

Semantic compilation V2 is complete when all of the following are true:

1. Each normal Body turn can emit Point sidecar in the same inference.
2. Point is explicitly turn-local and may depend on context.
3. Relevant Points stay buffered until a Block decision.
4. Block compiler runs while source turns remain in the live conversation window.
5. Cache-aware execution is measured, not assumed.
6. Multi-turn correction/clarification fixtures produce context-complete Blocks.
7. The Block can be consumed without raw turns for the intended meaning.
8. MR-Mem admission is deterministic after semantic authority output.
9. Thread consumes accepted Blocks.
10. Path B consumes block-bound semantic projections.
11. No downstream component needs to recreate the lost conversation context as its normal behavior.

At that point the "wheat to flour" step is genuinely present in the runtime.
