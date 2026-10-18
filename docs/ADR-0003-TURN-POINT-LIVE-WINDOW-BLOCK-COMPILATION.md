# ADR-0003 — Turn-local SemanticPoint and live-window SemanticBlock compilation

- Status: ACCEPTED / FROZEN
- Date: 2026-10-03
- Scope: Body semantic sidecar, SemanticPoint, SemanticBlock, MR-Mem admission, Thread/LCE input
- Parent: ADR-0001-SIDECAR-MEMORY-BOUNDARY.md
- Supersedes: any interpretation of ADR-0002 in which deterministic COHABIT closure itself is treated as the semantic algorithm that discovers or creates a SemanticBlock

## Decision summary

The semantic architecture is fixed as follows:

~~~text
turn-based conversation
        |
        | Body normal inference
        | response + turn-local semantic sidecar
        v
SemanticPoint(s)
        |
        | points accumulate while the relevant conversation
        | context is still resident in the live window
        v
live-window contextual semantic compilation
        |
        | original turns + turn-local points + bounded active context
        v
SemanticBlock
        |
        +--> canonical semantic memory
        +--> Thread projection --> Path A --> LCE
        +--> block-bound semantic projection / Point Cloud --> Path B --> LCE
~~~

The central invariant is:

> SemanticPoint exists because the interaction is turn-based. SemanticBlock exists because a turn-local understanding is not safe to consume as a complete meaning outside the conversation that produced it.

A SemanticPoint is not an NLP atom and is not required to be context-free. A SemanticBlock is not a bag of similar points, not a newline join, not a summary, and not a deterministic quotient graph. It is the context-complete semantic result compiled from a bounded sequence of turns whose meaning emerged through continuation, correction, qualification, contradiction, clarification, reference and other cross-turn development.

Without SemanticBlock, the normal long-term semantic path is incomplete. Thread and LCE must not treat naked turn-local points as authoritative standalone memory.

## 1. Why SemanticPoint exists

The host LLM receives and produces conversation in turns. Therefore its understanding necessarily develops incrementally.

Example:

~~~text
Turn 1 -> current local understanding P1
Turn 2 -> P2 extends or corrects P1
Turn 3 -> P3 narrows the scope
Turn 4 -> P4 resolves what the earlier turns jointly meant
~~~

Each turn may contain a valid local semantic interpretation, but the complete meaning can remain underdetermined until later turns arrive.

Therefore:

~~~text
SemanticPoint = one turn-local semantic state/proposal produced by the Body while handling that turn.
~~~

It is not defined as:
- the smallest proposition in language;
- an AMR node;
- a sentence fragment that must stand alone forever;
- a context-free fact;
- a long-term canonical memory item by default.

A Point may contain multiple local propositions if that is the faithful turn-level interpretation. The protocol should not invent artificial granularity solely to create more nodes.

## 2. Why SemanticBlock exists

SemanticBlock exists to prevent later systems from consuming a locally correct turn in a globally wrong way.

For example, a discussion may develop as:

~~~text
P1: swapping itself can have value
P2: automakers directly owning a nationwide heavy swap-station network can have poor economics
P3: continuing to build swap-capable cars does not contradict exiting direct infrastructure ownership
~~~

Each Point can be locally correct. Any one of them can still be misleading when retrieved alone.

The Block is the compiled result of the sequence and must preserve the jointly expressed meaning, including the distinctions that make the local statements safe to consume.

Therefore:

~~~text
SemanticBlock = context-complete semantic compilation over a bounded turn span.
~~~

"Context-complete" means complete for the meaning being committed, not that the Block must copy the transcript.

The Block must preserve whatever cross-turn structure is necessary to avoid changing the intended meaning when the raw turns are no longer present in the consumer context.

## 3. SemanticPoint production is a Body sidecar

Online Point production MUST occur as a sibling output of the normal Body inference whenever the host supports structured sidecar output.

~~~text
Body inference
  -> assistant response
  -> SemanticPoint sidecar
~~~

The system MUST NOT add a second online model call merely to reinterpret the same current turn into a Point.

This is a cost and authority decision:
- the Body already understands the current turn;
- the Point is an attached semantic product of that inference;
- another model pass would add cost, latency and competing semantic authority.

MR-Mem validates and stores accepted semantic products. It does not reconstruct turn semantics with its own small LLM.

## 4. SemanticBlock compilation is a separate contextual operation

Block formation is not required to happen in every Body inference.

Unlike a Point, a Block may require rereading multiple turns and comparing the evolution of meaning across them. It is therefore a separate contextual semantic compilation operation.

The normal online route is:

~~~text
live conversation prefix
+ relevant original turns still in the active window
+ accepted turn-local Points
+ bounded adjacent canonical context when explicitly needed
        |
        v
contextual semantic compiler
        |
        v
SemanticBlock proposal
~~~

The compiler SHOULD execute while the relevant conversation prefix is still live/cacheable.

The intended cost model is:

~~~text
every turn:
  Body response + Point sidecar          high frequency / no extra inference

occasionally:
  contextual Block compilation          low frequency / reuse live prefix cache
~~~

When the host/model supports KV or prompt-prefix caching, the Block compiler request should preserve the same stable system prefix and live conversation prefix so that the contextual reread hits the existing cache as much as possible.

The normal path MUST NOT wait until the relevant turns have fallen out of the live context and then reconstruct the same meaning from a broad historical RAG query. That would reintroduce retrieval-selection error into the mechanism whose purpose is to prevent decontextualization.

Historical/backfill compilation remains possible, but it is a recovery/import path, not the preferred online path.

## 5. Points are intermediate semantic state, not canonical standalone truth

Turn-local Points may be retained for:
- Block compilation support;
- provenance;
- audit;
- semantic evolution traces;
- block-bound structural projections;
- future recompile after compiler-version changes.

They MUST NOT automatically become independently retrievable canonical claims merely because the Body emitted them.

A Point can be promoted to a one-Point Block only when the contextual compiler determines that its meaning is already complete and safe to consume alone.

This prevents the system from converting every conversational turn into a permanent isolated "fact."

## 6. Block formation is semantic, not deterministic graph compression

The following mechanism is explicitly rejected as the semantic Block algorithm:

~~~text
COHABIT edges
 -> union-find
 -> connected components
 -> newline join / deterministic graph normalization
 -> call result SemanticBlock
~~~

Deterministic closure remains useful for validation, replay, endpoint checking and canonicalization after a semantic Block proposal exists.

It does not own the semantic judgment:
- which turns jointly express one meaning;
- whether a later turn corrects or narrows an earlier one;
- whether a reference points to an earlier concept;
- whether several locally valid statements must remain together to avoid distortion;
- whether an apparent contradiction is actually a distinction between scopes.

Those judgments belong to the Semantic Authority operating over the live context.

## 7. Body Point protocol: minimum required contract

The Point sidecar should remain bounded. It exists to preserve the Body's current semantic understanding and provide useful intermediate state for later Block compilation.

A V2 Point proposal should minimally carry:

~~~text
SemanticPoint
- point_id                  system/turn-stable local identity
- interaction_id / turn_id
- meaning                   faithful local semantic interpretation
- context_links[]           optional links to prior Point/Block IDs
    - target_id
    - relation              open semantic relation text
- epistemic                 when materially present
- polarity                  when materially present
- temporal_scope/expression when materially present
- unresolved_refs[]         unresolved contextual references
- status                    RESOLVED / DEFER
~~~

Rules:
- meaning is local semantic meaning, not a requirement to restate all prior context.
- relation remains open semantic language; do not create a closed ontology for unlimited meanings.
- epistemic/polarity/time are retained when they materially change interpretation.
- unresolved references are not guessed; they remain DEFER or explicit unresolved context.
- SourceRef, Scope, occurred_at, known_at and host revision authority remain system-bound, not Body-authored.

The protocol may later be reduced or extended only based on real compiler needs. It must not grow into a complete duplicate of the conversation.

## 8. Active semantic buffer

The host integration should maintain a bounded active semantic buffer aligned to the live conversation window:

~~~text
ActiveSemanticBuffer
- ordered turn references
- Point proposals for those turns
- candidate span markers
- last compiled Block boundary
- unresolved references
- adjacent active Block IDs when needed
~~~

This is not another durable transcript.

Its purpose is to tell the Block compiler:
- which Points are still waiting for contextual compilation;
- which original turns remain available in the live window;
- where the last accepted Block ended;
- whether unresolved material prevents closure.

The buffer may be reconstructed from the native host source plus accepted Point receipts if the process restarts while the turns are still available.

## 9. Block compilation triggers

Block compilation is low frequency and event-driven. The implementation should support at least these trigger classes:

1. Semantic boundary candidate: the conversation clearly shifts away from the active meaning line.
2. Correction/clarification settlement: several turns have revised the same concept and the latest state appears stable enough to commit.
3. Explicit completion: a decision, explanation, plan, definition or argument has reached a locally complete form.
4. Window pressure: relevant turns are approaching compaction/eviction. A compile attempt MUST occur before they are lost from the live window.
5. Bounded safety threshold: too many uncompiled turns/Points have accumulated.

Triggers schedule compilation; they do not decide Block semantics. The semantic compiler may return:
- one or more Block proposals;
- no closed Block yet;
- a partial span that remains unresolved.

No fixed turn count is the semantic definition of a Block.

## 10. Block compiler input and output

The compiler receives only bounded material:

~~~text
Input:
- exact source turns for the candidate span
- accepted Point sidecars for those turns
- immediately adjacent accepted Block context if required
- explicit unresolved references / activated canonical context
- system authority envelope
~~~

It must not receive the user's complete lifetime history by default.

A candidate Block should minimally produce:

~~~text
SemanticBlockProposal
- block_local_id
- member_point_ids[]
- source_turn_refs[]
- compiled_meaning
- internal_relations[]          only where needed to preserve meaning
- external_context_refs[]       canonical references, not copied raw history
- unresolved_context[]
- closure_status                CLOSED / DEFER
~~~

compiled_meaning is the context-complete meaning of the span. It is not a prose summary optimized for readability and not a concatenation of Point texts.

The proposal does not own SourceRef authority, known_at, immutable canonical ID or lifecycle. MR-Mem binds and validates those.

## 11. Canonical admission

MR-Mem remains the canonical authority after semantic proposal.

Admission should perform deterministic checks such as:
- every member Point exists and belongs to the allowed source span;
- every source turn reference resolves to the exact current native revision;
- external context refs are within activated/authorized canonical scope;
- unresolved refs cannot be silently promoted to CLOSED;
- system-owned time/scope/provenance fields cannot be forged by the Body;
- replay of the same accepted proposal is deterministic;
- compiler version is frozen in the receipt;
- recompile under a new semantic compiler version produces a new immutable semantic identity rather than silently reinterpreting an old one.

Deterministic closure is a validator/canonicalizer after semantic compilation, not the source of semantic grouping authority.

## 12. Relationship to Point Cloud and LCE

LCE must not build longitudinal structure from naked, unbound turn Points as if each were a complete standalone belief.

The safe route is:

~~~text
Turn Points
   -> SemanticBlock compilation
   -> canonical Block
   -> block-bound structural projection
~~~

A Point Cloud may preserve the finer-grained trajectory of the member Points, but each projected Point MUST retain its Block identity/context boundary once that Block exists.

Conceptually:

~~~text
Block B1:
  P1 -> P2 -> P3

Block B2:
  P4 -> P5

block-bound Point Cloud:
  B1/P1  B1/P2  B1/P3  B2/P4  B2/P5
~~~

This allows Path B to observe semantic movement and recurring local structure without forgetting which Points require shared context to be interpreted correctly.

Path A remains:

~~~text
canonical SemanticBlock
 -> active Thread
 -> mature Thread handoff
 -> LCE
~~~

Path B remains:

~~~text
canonical SemanticBlock
 -> block-bound Point Cloud
 -> trajectory / Line discovery
 -> LCE
~~~

Thus:
- Block preserves complete meaning;
- Point-level projections preserve the evolution path;
- Thread tracks an already-visible line cheaply;
- Path B searches for latent longitudinal structure.

SemanticBlock is therefore a prerequisite of the normal LCE route, not a later optional optimization.

## 13. Relationship to Thread

Thread exists to avoid repeatedly scanning the entire memory store for already-visible lines.

It consumes accepted Block semantics and evolves incrementally as new Blocks arrive.

Thread is not responsible for repairing turn-local semantic incompleteness. If a Thread must repeatedly reread raw turns to understand a Point, the Block boundary failed upstream.

A mature Thread handoff may include:
- Block IDs;
- Block compiled meanings;
- relevant block-bound Point trajectory/support;
- temporal/provenance state.

A free-text Thread summary alone is not sufficient authority.

## 14. Cache-aware online execution

The preferred timing is:

~~~text
T1: Body response + Point
T2: Body response + Point
T3: Body response + Point
    boundary/closure signal
    -> Block compile while T1..T3 prefix is still hot
T4: continue with accepted Block available
~~~

Implementation requirements:
- keep the compiler instruction stable;
- keep the same conversation prefix ordering as the Body path;
- invoke before host context compaction;
- use the same model/cache-compatible route where possible;
- compile only bounded candidate spans, not the whole session;
- persist accepted Block receipts so downstream failure never forces semantic recompilation.

If a provider exposes no reusable KV/prompt cache, correctness is unchanged; only the cost optimization is unavailable.

## 15. Recovery and hot start

Online and historical compilation share the semantic contract but not necessarily the executor.

Online:
- current Body inference yields Points;
- live-window contextual compiler yields Blocks.

Historical:
- native turns are read in bounded windows;
- an external semantic authority may reconstruct Points/Blocks;
- accepted compilation receipts are frozen;
- downstream replay must reuse them.

The historical route must emulate the same semantic boundary, not redefine Point or Block.

## 16. Issue #14 interpretation

The SemanticBlock V1 experiment from Issue #14 remains useful as negative evidence about one implementation class:

> deterministic post-hoc graph normalization of already-emitted Points does not by itself solve contextual semantic compilation.

Its NO-GO result MUST NOT be interpreted as:
- Block being optional;
- Point being sufficient;
- SemanticBlock being postponed until after LCE;
- all Point-to-Block routes being impossible.

The experiment attacked the wrong layer for semantic grouping but correctly demonstrated that V0 public Block projection loses important structure and that deterministic lossless repackaging does not create the required contextual compilation.

## Frozen invariants

The following are architecture invariants until explicitly superseded by a later accepted ADR:

1. Point exists because conversation understanding is turn-local.
2. Point is not defined as a context-free NLP atom.
3. Point sidecar is produced by the normal Body inference without an extra online semantic-model call.
4. Block is the context-complete compilation across one or more related turns.
5. Block formation may reread the relevant live conversation context.
6. Block compilation should happen while that context remains in the active window so cache reuse is possible.
7. The semantic authority, not union-find or vector similarity, decides Block meaning and membership.
8. Deterministic code validates/canonicalizes accepted semantic proposals; it does not invent missing meaning.
9. Naked turn-local Points are not authoritative standalone long-term memory.
10. SemanticBlock is required before normal Thread/LCE semantic consumption.
11. Path A uses Block -> Thread -> LCE.
12. Path B may preserve Point-level trajectories only as block-bound projections.
13. Block exists to prevent decontextualization / 断章取义; any design that makes every Point artificially self-contained and thereby removes the need for Block violates the architecture.
14. Raw native turns remain the ultimate audit source and may be reread during live-window Block compilation; MR-Mem does not become the raw transcript authority.
15. No new feature work on LCE, RAG, AML or Hot Start may redefine these semantics implicitly.

## Immediate implementation consequence

The next engineering work is not another Block compression experiment.

The order is:

~~~text
A. freeze SemanticPoint V2 sidecar protocol
B. validate same-inference Body emission on real multi-turn conversations
C. implement bounded ActiveSemanticBuffer
D. implement cache-aware live-window Block compiler contract
E. evaluate Block fidelity on correction/clarification/multi-turn cases
F. only after Block semantics pass, wire canonical Block output into Thread and block-bound Path B
~~~

This ADR is the semantic-compilation architecture authority until explicitly superseded.
