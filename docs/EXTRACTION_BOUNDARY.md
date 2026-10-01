# Extraction boundary

MR-Mem is the reusable **semantic-memory sidecar** boundary for Mind Runtime and
other Agents.

The frozen authority decision is defined by
[ADR-0001](ADR-0001-SIDECAR-MEMORY-BOUNDARY.md).

## Host-native source owns raw history

When a host already persists sessions/messages/tool records, that database is
the raw authority.

MR-Mem does not own another complete transcript and does not require full raw
history to be copied into canonical Memory before semantic processing.

Host source access is provider-neutral and should be expressed through stable
references plus adapter capabilities for delta discovery, resolution, optional
raw retrieval, and revision/deletion visibility.

## MR-Mem owns canonical semantic memory

MR-Mem owns the durable interpretation layer:

- canonical semantic identity
- semantic content/structure
- scope and subject boundaries
- SourceRef provenance
- occurred-at / known-at semantics
- support / contradiction / supersession state
- semantic retrieval indexes
- Thread product state
- LCE projection state and longitudinal identities
- accepted higher-order/Baseline products
- cursors, receipts and versions needed for idempotent recovery

Existing `CommittedMemory` / `CanonicalMemoryStore` names are implementation
history. They must not be interpreted as authority over the host's raw
conversation database.

## Semantic Authority

MR-Mem consumes accepted structured semantics. It does not require Mind
Runtime's Body and does not construct a hidden replacement model.

```text
Body LLM / host model / offline compiler
                |
        Semantic Authority
                |
       canonical semantic memory
                |
        Thread      LCE
```

The Body semantic protocol is therefore an integration seam, not an internal
small-LLM ownership boundary.

## Thread / LCE

Thread and LCE are projections over the same canonical semantic substrate.

Thread is medium-term active cognition. LCE is longitudinal structural
cognition. Both retain semantic support IDs; raw audit follows those IDs through
SourceRefs to the native host store.

They may persist stateful evolution, but neither may become a second semantic
authority or raw transcript store.

## Retrieval

The reusable retrieval model is three-lane:

```text
raw/source lane
+ semantic-memory lane
+ Thread/LCE structural lane
        -> fusion/rerank
        -> selective source hydration
        -> context assembly
```

Raw/source retrieval may use a rebuildable index, but the raw body remains
host-owned whenever possible.

## Recovery

Hot/warm start reconciles cursors, source revisions, semantic compile receipts
and projection state. It does not duplicate the entire native history as a
second canonical database.

Only missing, stale or changed work should be recomputed.

## Host without persistence

An optional RawSource adapter may provide durable source storage when the host
has none. This is also how benchmark-only runtimes such as AML may stage source
input.

This optional source implementation does not change MemoryCore's authority.

## Still outside MR-Mem

- host interaction lifecycle
- host-native storage schema
- affect/appraisal/intent/policy/StateBar
- concrete Body/Host inference
- concrete SourceAdapter implementations
- host-specific context composition
- AML protocol and HTTP surface

## Source snapshot

The extraction started from:

- `Jasonatafricanow/Mind-Runtime@18d9275175ed6781962164e530efe4433e29ac7f`
- `refactor/independent-memory-core@5071955449aff079f11fb958948fbabb28050acb`

Historical identifiers are retained only where they remain correct under the
sidecar semantic-memory authority boundary.
