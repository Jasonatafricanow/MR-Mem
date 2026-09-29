# Extraction boundary

MR-Mem is the standalone product boundary for the memory side of Mind Runtime.

## MR-Mem owns

### Admission and canonical authority

- structured `Scope`
- `SyncFields` identity and idempotency metadata
- provider-neutral `SourceEvidence` / `SourceObservation`
- read-only `DurableFactReader` proof boundary
- NEW / REPAIRED / REPLAY Memory eligibility
- bounded extraction and provenance enforcement
- immutable `CommittedMemory`
- canonical SQLite persistence
- rebuildable projection outbox

Admission is Memory governance. A host may use MR's existing factual plane or a
different source store; MR-Mem depends only on the narrow structural contracts.

### Read and working plane

- canonical retrieval authority revalidation
- BM25 lexical retrieval
- RRF hybrid fusion
- optional HyDE sparse-recall fallback
- provider-neutral semantic reranking seam
- embedding identity and vector validation
- optional FastEmbed / Qdrant derived indexes
- attention state
- non-durable Pending overlay
- bounded medium-term Thread
- automatic Thread maintenance from accepted semantic events

Thread consumes the structural `ThreadSemanticEvent` protocol instead of
Mind Runtime's concrete semantic-event type. It does not call a model.

## Canonical vs derived state

`CanonicalMemoryStore` is the durable factual Memory authority.

Attention, Pending, retrieval indexes, Thread, and future LCE state are derived
or working state. They cannot create factual authority merely by existing.

A mature Thread may be retired after an accepted higher longitudinal projection;
canonical Memory IDs retain the factual support chain.

## Host responsibilities

These do not belong to MR-Mem:

- Mind Runtime interaction lifecycle
- RuntimeBinding and host storage layout
- affect, appraisal, intent, policy, StateBar
- Body/Host online inference
- host-specific historical-context assembly

MR should be able to run its affect side against MR-Mem, another memory backend,
or no persistent memory backend through a thin historical-context port.

## Deferred semantic authority boundary

LCE requires semantic authority, but the authority implementation does not
belong inside the memory engine.

The intended next boundary is a provider-neutral Semantic Authority port:

```text
Body LLM / host model / offline annotator
                |
        Semantic Authority
                |
             MR-Mem
                |
          Thread -> LCE
```

MR-Mem/LCE consume structured semantic output. They do not require Mind
Runtime's Body and do not secretly introduce a second default semantic model.

## Source snapshot

The extraction started from:

- `Jasonatafricanow/Mind-Runtime@18d9275175ed6781962164e530efe4433e29ac7f`
- `refactor/independent-memory-core@5071955449aff079f11fb958948fbabb28050acb`

Historical `mr-memory-v1`, Thread, and projection identities are retained where
identity continuity matters.
