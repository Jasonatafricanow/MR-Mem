# MR-Mem

[中文说明](README.zh-CN.md)

**A memory system that keeps what was said as the authority and what it understood as a
traceable projection.** MR-Mem stores every message as raw evidence, compiles the
conversation into semantic blocks (**SB**s — one thing being talked about), links the blocks
into **Lines** (how a matter moved on) and cuts Lines into **stretches**, each summarized
once as cause → who → when → what → outcome. A question is answered from that structure:
raw hits lead to their SBs, the SBs to their stretches, and each SB brings back the exact
words it was compiled from. Nothing the compiler wrote replaces the source; every projection
can be followed back to it.

```text
raw messages ──BM25 + vectors (RRF)──▶ raw evidence      (P1 baseline)
   │ compile in windows, closing at natural breakpoints
   ▼
SB: one matter talked about ── carries pointers to its words
   │ links: "this step carries that one on"
   ▼
Line ──▶ stretches with cause → who → when → what → outcome   (LCE)
```

| | |
|---|---|
| **Where it was tested** | AML 2026 Round 2 (official run 2026-11-02, average **54.37**), then public BEAM, PersonaMem-v2 and CLBench sets locally |
| **What is on `main`** | The submitted version plus the post-competition changes measured as gains: SBs carried with their linked steps, message times written into the readout, and precise readout (only the SBs the search found carry their words; dev set 37.1 → 43.1, stable over two answer passes) |
| **How it got here** | [Research log](docs/DEVLOG.md) — what was built, what was dropped and why |
| **Design** | [Frozen definitions](docs/LCE_FROZEN.md) · [LCE design](docs/LCE_DESIGN.md) · [technical report](docs/TECHNICAL_REPORT.md) · [configuration guide](docs/CONFIG_GUIDE.md) |

| Branch | What it holds |
|---|---|
| `main` | The version above |
| `feat/aml-20261013` | The AML submission, frozen |
| `dev` | Ongoing work; experimental switches default off |

Scores are development measurements on public data and one official run, not independent
validation.


## Start here

Python 3.12+ and Git are required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[aml,lce]"
python scripts/smoke_offline.py
```

Installation downloads dependencies. The smoke test uses synthetic messages, temporary
storage and loopback HTTP only: **no model calls or model charges**. It checks Add, Search
and user isolation, not semantic quality. Core-only users can install with `pip install -e .`.

To connect real providers, copy `examples/runtime.aml-lce.json` to `runtime.json`, edit the
endpoints/models for your account and set the named environment variables. `.env.example`
lists names only; `.env` files are not loaded automatically. Set `MRMEM_AML_TOKEN` to your
own service access token, then run `mr-mem-aml serve --config runtime.json`. Clients use
`Authorization: Bearer <service-token>`. Default binding is loopback.

Provider-backed Add/Search and `check` may incur charges. `check --check-decision` also
probes the optional decision provider. `/health` itself is passive. Decision failures and
the default 15-second total deadline leave the ordinary retrieval response available.
Remove `decision` from the configuration to disable that provider.

See [deployment](docs/AML_SUBMIT_LCE.md), [HTTP/configuration](docs/AML.md),
[frozen definitions](docs/LCE_FROZEN.md), and [privacy](SECURITY.md).
Reported benchmark scores are development measurements, not independent validation.

For development: `pip install -e ".[dev,aml,lce]"`, then `ruff check src tests`,
`python -m pytest tests`, and `python scripts/smoke_offline.py`.

## Architecture

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

with MemoryCore(Path("memory.sqlite")) as memory:
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

## Path A

`mr_mem.integrations` (optional, `pip install -e ".[lce]"`) connects mature
Threads, first-appearance complete cognition and lineage revisions to LCE Path A,
routes compiled Block signals in one place, and provides one unified read path
(accepted cognition with its evidence, then active Threads, then canonical
Memory). It is host-neutral: the host keeps raw text (ADR-0001) and supplies the
semantic compiler and retrieval stack. See [docs/PATH_A.md](docs/PATH_A.md).
The AML evaluation adapter (`mr_mem_aml`) is described in
[docs/AML.md](docs/AML.md).

For a server deployment, use [Docker Compose deployment](docs/DEPLOYMENT.md).
