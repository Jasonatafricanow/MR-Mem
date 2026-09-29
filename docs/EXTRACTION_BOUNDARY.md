# Extraction boundary

MR-Mem is the standalone product boundary for the memory side of Mind Runtime.

## Canonical authority

`CanonicalMemoryStore` is the durable factual authority. `MemoryCore` owns
connection lifetime and exact bounded selection; it does not create a second
write path.

The first extraction preserves:

- structured `Scope`
- `SyncFields` identity and idempotency metadata
- immutable `CommittedMemory`
- canonical SQLite persistence
- projection outbox state
- derived attention state
- bounded medium-term `MemoryThread`
- `MemoryCore` ownership and exact selection

## Derived state

Attention and Thread rows are projections over canonical Memory. They may be
rebuilt, suppressed, resolved, or retired without changing factual authority.

Future LCE integration must follow the same rule: LCE reads canonical MR-Mem
source state and owns only derived longitudinal projection state.

## Not part of the first core extraction

The following remain outside the package for now:

- evidence/observation admission
- Body/Host online semantic parsing
- retrieval providers and Decision Plane
- Thread auto-update policy
- LCE engine implementation
- AML/benchmark protocol
- Mind Runtime affect/state/intent logic

Those should attach through explicit adapters rather than becoming dependencies
of the canonical core.

## Source snapshot

Initial source material:

- `Jasonatafricanow/Mind-Runtime@18d9275175ed6781962164e530efe4433e29ac7f`
- `refactor/independent-memory-core@5071955449aff079f11fb958948fbabb28050acb`
