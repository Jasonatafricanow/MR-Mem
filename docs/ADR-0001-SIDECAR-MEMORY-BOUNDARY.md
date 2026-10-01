# ADR-0001 — Sidecar memory boundary and native-source integration

- Status: ACCEPTED / FROZEN
- Date: 2026-10-01
- Scope: MR-Mem, Mind Runtime integration, Thread/LCE projections, retrieval, AML adapter

## Decision

MR-Mem is a **sidecar semantic-memory system**, not a replacement conversation/session database.

When a host Agent already owns durable session/history storage, that native store remains the sole authority for raw messages, tool records, session chronology, and original text. MR-Mem MUST NOT require a second full copy of that history and MUST NOT introduce bidirectional synchronization with it.

The invariant is:

```text
Host native store
raw/session authority
        |
        | one-way incremental source feed
        v
Semantic Authority
Body LLM / host model / offline compiler
        |
        v
MR-Mem canonical semantic state
        |
        +--> Thread projection
        +--> LCE projection
        +--> Baseline / retrieval products
```

## 1. Authority layers

### Host native store owns

- raw message/session/tool history
- original text and role ordering
- host record identity and revision/deletion state
- event/session chronology

MR-Mem never becomes authoritative for those records merely because it has observed them.

### MR-Mem owns

- compiled semantic memory records
- semantic identity and provenance
- occurred-at / known-at semantics
- support, contradiction, supersession and validity state
- semantic retrieval indexes
- Thread product state
- LCE state and longitudinal identities
- accepted Baseline / long-term semantic products
- ingestion cursors, receipts and compiler/index version metadata needed for idempotence and recovery

The current `CommittedMemory` / `CanonicalMemoryStore` implementation is transitional naming. Its authority is **compiled semantic memory**, not raw conversation history.

## 2. Source abstraction

MR-Mem consumes host history through a provider-neutral source boundary.

A stable source reference must be able to identify the native record without copying the entire body into MR-Mem. The contract should carry the minimum identity necessary for replay, invalidation and audit, for example:

```text
SourceRef
- source_namespace / provider
- conversation or session identity
- record identity
- occurred_at
- revision or fingerprint
```

Source integration is split into capabilities rather than one database assumption:

- incremental discovery / delta feed
- resolve a SourceRef to native content when needed
- optional raw-search capability for RAG
- revision/deletion visibility

Host-specific schemas stay behind adapters.

## 3. Semantic compilation boundary

Raw history is not canonical MR memory.

The host's Semantic Authority compiles source material into structured semantic output. MR-Mem does not silently create a second default online LLM.

```text
Source span(s)
   -> Semantic Authority
   -> SemanticBlock / canonical semantic memory
```

Compilation may classify, split, merge, bind subjects/scopes, extract propositions, preserve time, and attach source refs in one pass.

Excluded/noisy source text may remain only in the host-native store.

## 4. SemanticBlock is the common substrate

Thread and LCE consume the same canonical semantic substrate.

They MUST NOT independently re-own raw chat text or independently create competing semantic authorities.

```text
SemanticBlock
   |\
   | +--> Thread
   +----> LCE
```

Projection support resolves through:

```text
Thread / LCE / Baseline
        -> semantic memory IDs
        -> SourceRef
        -> host-native record
```

A projection may retain its own durable state and identity, but its factual/semantic support chain remains traceable to canonical semantic memory and then to native evidence.

## 5. Thread and LCE are stateful materialized projections

Thread represents active medium-term lines: unresolved topics, ongoing work, changing plans and current progress.

LCE represents longitudinal structure: point-cloud relations, trajectories, persistent Line DAG identity, support evolution, rejection/contradiction, maturity, Worktree/branch and Baseline transitions.

They are not independent memory databases and they are not disposable caches. Their structure is derived from canonical semantic memory, while their **evolution state** may itself need durable persistence.

A mature Thread may retire after an accepted higher longitudinal projection covers the same logical line.

## 6. Retrieval: three lanes, one fusion surface

MR-Mem may not own raw text, but raw evidence remains a first-class retrieval lane.

The retrieval surface is:

```text
A. Raw/source retrieval
   "What exactly was said?"

B. Canonical semantic retrieval
   "What does the history mean?"

C. Structural retrieval
   Thread / LCE
   "What ongoing or longitudinal structure exists?"

A + B + C
   -> fusion / rerank
   -> selective source hydration
   -> context assembly
```

A rebuildable raw-search index MAY persist identifiers, lexical/vector features, timestamps and revision metadata. It SHOULD NOT become a second authoritative copy of the complete host transcript.

## 7. One-way synchronization only

Authority flows one way:

```text
Host native source -> semantic compilation -> MR-Mem -> projections
```

There is no general MR-Mem -> host-history writeback and no bidirectional source synchronization.

Source revision or deletion invalidates only affected semantic support and projections. Recovery is local and idempotent.

## 8. Hot start and recovery

Hot/warm start MUST NOT re-import the host's complete transcript into another canonical raw store.

Recovery should use:

- source cursor/checkpoint
- SourceRef revision/fingerprint
- compile/version receipt
- semantic-memory identity
- projection receipt/state

Normal newly committed source material should become semantically searchable and update required Thread/LCE projections in the same processing lifecycle expected by the integration.

Restart reconciliation processes only missing, stale or changed work.

## 9. Hosts without native persistence

If a host has no durable session/history store, an optional RawSource implementation may provide one.

That implementation is a **source provider**, not a mandatory layer of Memory Core.

This also covers benchmark adapters such as AML, where Add supplies source material but Search does not expose a separately queryable host-native session database.

## 10. AML boundary

AML protocol belongs outside MR-Mem core.

Competition composition may use:

```text
AML Add
  -> AMLSourceStore
  -> normal SourceAdapter
  -> Semantic Authority
  -> canonical semantic memory
  -> Thread / LCE
  -> normal fusion retrieval
  -> AML Search
```

AMLSourceStore exists because the benchmark host does not supply a native readback store during Search. It must not redefine production authority boundaries.

AML-specific request journals, HTTP types, run namespaces and benchmark constraints remain adapter/runtime concerns.

## 11. Physical storage is not the logical boundary

MR-Mem semantic tables may live in:

- a standalone SQLite/Postgres database
- a separate schema in the host database
- another persistence backend

This does not change authority.

Two physical persistence systems do not imply two copies of the same memory. The forbidden design is two authoritative raw histories.

## Frozen non-goals

The following designs are explicitly rejected:

- copying every host session/message into MR-Mem as a prerequisite for use
- treating MR-Mem as the authoritative raw transcript when the host already has one
- bidirectional synchronization between host history and MR-Mem
- requiring every native record to become canonical semantic memory
- letting LCE create its own competing SemanticBlock store
- letting Thread own another semantic summary database independent of canonical semantic memory
- forcing host-specific session schemas into Memory Core
- making AML protocol concepts part of the reusable memory core

## Implementation consequence

Existing code that assumes:

```text
durable raw/factual Evidence -> canonical MR Memory -> LCE
```

must be reviewed and narrowed to:

```text
native source -> semantic compilation -> canonical semantic memory -> Thread/LCE
```

The existing recovery, idempotence, projection and retrieval work remains useful, but its authority must be attached to semantic-memory identities and SourceRefs rather than to a mirrored host transcript.

This ADR is the architecture authority for the sidecar-memory boundary until explicitly superseded by another accepted ADR.
