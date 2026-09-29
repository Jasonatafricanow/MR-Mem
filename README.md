# MR-Mem

Standalone Memory runtime extracted from Mind Runtime.

MR-Mem owns the reusable memory-side pipeline: admission governance, canonical
Memory authority, retrieval, derived attention/Thread state, and optional
vector projections. Mind Runtime can consume it as a memory backend instead of
owning those subsystems.

## Current boundary

MR-Mem now owns:

- structured Scope + sync identity
- provider-neutral source/admission contracts
- durable-pair admission governance
- bounded deterministic extraction
- canonical Memory contracts and SQLite store
- rebuildable projection outbox
- attention + bounded Thread product state
- non-durable Pending working overlay
- canonical retrieval authority revalidation
- BM25 lexical retrieval
- RRF hybrid retrieval
- HyDE as an optional sparse-recall fallback
- provider-neutral semantic reranking seam
- embedding identity/contracts
- optional FastEmbed and Qdrant providers
- Thread auto-update behind a provider-neutral semantic-event protocol
- runtime-independent `MemoryCore`

The package has no dependency on Mind Runtime's `RuntimeBinding`, affect
runtime, host composition, or concrete FactBackend.

## Admission

Admission is owned by Memory governance. Any host can provide:

```text
SourceEvidence
SourceObservation
DurableFactReader
Clock
```

MR-Mem then enforces durable source pairing, NEW/REPAIRED/REPLAY eligibility,
provenance bounds, idempotency, and stable canonical Memory identity.

## Semantic boundary

Thread maintenance consumes only the structural `ThreadSemanticEvent` protocol:
`scope + attributes + evidence_refs`. MR-Mem does not own or invoke the model
that produced those semantics.

The same principle will be used for LCE: a later Semantic Authority port will
accept structured semantic output from a Body LLM, another host model, or an
offline annotator without making MR-Mem depend on Mind Runtime's Body.

## Retrieval

BM25 and hybrid RRF are local ranking mechanisms over stable Memory IDs.
HyDE is available as a fallback when primary recall is sparse; it is not the
default path. Optional semantic reranking is injected through `MemoryReranker`.

Vector support is optional:

```bash
pip install "mr-mem[vector]"
```

Qdrant is opened with explicit `index_root`, `canonical_path`, and
`namespace`; there is no RuntimeBinding dependency.

## Quick start

```python
from pathlib import Path

from mr_mem import MemoryCore

with MemoryCore(Path("memory.sqlite"), read_only=True) as memory:
    rows = memory.load_all()
```

`MemoryCore` deliberately does not expose a second public canonical write
authority. Canonical writes flow through Memory admission governance.

## Still outside MR-Mem

These remain host or later integration concerns:

- Mind Runtime interaction lifecycle
- RuntimeBinding / host storage layout
- affect, appraisal, intent, policy, StateBar
- host-specific historical-context adapters
- LCE engine binding and Semantic Authority implementation
- AML benchmark protocol

## Extraction provenance

The standalone subsystem originated from Mind Runtime
`main@18d9275175ed6781962164e530efe4433e29ac7f`. Stable historical identity
prefixes are intentionally preserved during package extraction.
