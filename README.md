# MR-Mem

Standalone sidecar semantic-memory runtime extracted from Mind Runtime.

MR-Mem does **not** replace an Agent's native session/history database. When the
host already has durable conversation storage, that store remains the raw source
of truth. MR-Mem owns compiled semantic memory plus Thread/LCE-derived cognition
and retrieval products.

See [ADR-0001](docs/ADR-0001-SIDECAR-MEMORY-BOUNDARY.md) for the frozen
authority boundary.

## Current boundary

MR-Mem owns:

- provider-neutral source references and ingestion/recovery metadata
- semantic-memory admission and canonical identity
- structured provenance back to host-native records
- canonical semantic-memory persistence
- semantic retrieval and rebuildable indexes
- BM25 lexical retrieval
- RRF hybrid retrieval
- HyDE as an optional sparse-recall fallback
- provider-neutral semantic reranking seam
- embedding identity/contracts
- optional FastEmbed and Qdrant providers
- attention + bounded Thread product state
- automatic Thread maintenance from accepted semantic output
- the reusable MemoryCore boundary

MR-Mem must not require a second authoritative copy of the host's complete raw
conversation history.

## Source boundary

A host integrates through a narrow source adapter. The adapter may expose:

- incremental source discovery
- stable SourceRef identity
- native-record resolution when raw text is needed
- optional raw/source retrieval for RAG
- revision/deletion visibility

Raw bodies remain in the host-native store whenever that store exists.

For hosts without durable history, an optional RawSource implementation may
supply the missing source layer. That is an adapter concern, not a mandatory
MemoryCore substrate.

## Semantic boundary

Semantic Authority belongs to the host:

```text
host-native source
        |
Body LLM / host model / offline compiler
        |
canonical semantic memory
        |
   +----+----+
   |         |
 Thread     LCE
```

MR-Mem does not silently introduce a second default online semantic model.

The native admission entry point is `SemanticAdmissionService.admit`. A host
supplies an already accepted `SemanticMemoryCandidate` and a `SourceRefReader`
that verifies exact current, scoped source metadata. Neither capability requires
MR Evidence/Observation rows or transfers raw text. `MemoryAdmissionService`
and `DurableFactReader` remain a legacy compatibility adapter only.

`CommittedMemory.content` and `attributes` contain accepted semantic output.
Its provenance carries native SourceRefs; `occurred_at` is the earliest supporting
source event time, while `known_at` aliases the persisted semantic commit time.
Compiler version and source versions participate in immutable semantic identity.
Replaying a changed interpretation under the same identity fails closed.

After receiving a native revision or deletion, the host calls
`canonical.invalidate_source(scope, current_ref, deleted=...)`. An indexed
SourceRef lookup invalidates only affected semantic records. Old support and
contradiction/supersession links remain inspectable. Retrieval and Thread
surfacing revalidate canonical lifecycle, and projection intents are requeued.
Recompilation is host-owned; this API neither rescans nor reads raw history.
Hosts must serialize source deltas against their current metadata and pass
deletion tombstones explicitly.

Thread consumes committed semantic records directly. Independent support is
counted by native event identity (namespace/session/record), so revisions or
multiple semantic fragments of one event cannot manufacture maturity.

Thread and LCE consume the same semantic substrate and retain traceability
through semantic IDs to SourceRefs.

## Retrieval

The intended retrieval surface combines three lanes:

1. raw/source retrieval for exact evidence
2. canonical semantic retrieval for meaning
3. Thread/LCE structural retrieval for active and longitudinal context

Fusion/reranking happens before selective raw hydration.

A raw-search index may cache identifiers, embeddings, lexical features and
revision metadata, but it must not become a second authoritative transcript.

## Canonical vs projection state

Canonical MR-Mem state means **compiled semantic memory**, not raw chat history.

Thread and LCE are stateful materialized projections over that semantic
substrate. They may persist evolution state and stable identities, but they do
not become independent semantic authorities.

## Quick start

```python
from pathlib import Path

from mr_mem import MemoryCore

with MemoryCore(Path("memory.sqlite"), read_only=True) as memory:
    rows = memory.load_all()
```

## Outside MR-Mem core

These remain host/integration concerns:

- host interaction lifecycle
- host-native session/history storage
- Body/Host online inference
- affect, appraisal, intent, policy, StateBar
- concrete SourceAdapter implementations
- host-specific context assembly
- AML HTTP/benchmark protocol

## Extraction provenance

The standalone subsystem originated from Mind Runtime
`main@18d9275175ed6781962164e530efe4433e29ac7f`. Historical identities should
only be preserved where they remain semantically valid under ADR-0001.
