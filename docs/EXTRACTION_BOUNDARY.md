# Extraction boundary

MR-Mem is the standalone product boundary for the memory side of Mind Runtime.

## Canonical authority

`CanonicalMemoryStore` is the durable factual Memory authority. `MemoryCore`
owns connection lifetime and exact bounded selection; it does not create a
second write path.

The standalone package owns:

- structured `Scope`
- `SyncFields` identity and idempotency metadata
- provider-neutral source/admission contracts
- durable Evidence/Observation pair verification through `DurableFactReader`
- NEW / REPAIRED / REPLAY eligibility semantics
- bounded extraction and provenance enforcement
- immutable `CommittedMemory`
- canonical SQLite persistence
- projection outbox state
- derived attention state
- bounded medium-term `MemoryThread`
- `MemoryCore` ownership and exact selection

## Admission ownership

Admission is Memory governance, not affect-runtime behavior.

MR-Mem decides when a durably admitted source pair may become canonical Memory,
prevents historical backfill on repaired/replayed facts, constrains extractor
provenance to admitted source authority, and preserves stable Memory identity.

The package does not require Mind Runtime's concrete `FactBackend`. Hosts expose
only a read-only proof surface:

```
SourceEvidence
SourceObservation
DurableFactReader
Clock
```

Mind Runtime may adapt its existing Evidence / Observation / FactBackend types
to that surface. Other hosts may use different source stores.

## Derived state

Attention and Thread rows are projections over canonical Memory. They may be
rebuilt, suppressed, resolved, or retired without changing factual authority.

Future LCE integration must follow the same rule: LCE consumes canonical
MR-Mem state plus externally supplied semantic authority and owns only derived
longitudinal cognition.

## Outside this package boundary

These concerns remain host/integration responsibilities:

- Mind Runtime interaction lifecycle
- RuntimeBinding and host storage composition
- Body/Host online inference
- affect/state/intent/policy logic
- host-specific historical-context adapters
- semantic authority implementation used by LCE

Retrieval providers, Thread auto-update, LCE integration, and AML benchmarks are
Memory-owned or Memory-adjacent work and can be extracted without moving MR host
behavior into this package.

## Source snapshot

Initial source material:

- `Jasonatafricanow/Mind-Runtime@18d9275175ed6781962164e530efe4433e29ac7f`
- `refactor/independent-memory-core@5071955449aff079f11fb958948fbabb28050acb`
