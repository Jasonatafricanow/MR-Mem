# MR-Mem Path A canonical projection — independent implementation and Claude review contract

Status: isolated PR, not merged; implementation began from MR-Mem public main 0685a8a2498cc828c8d500be6f4673e53fa6dc24. **No Grok Path A patches, precomputed harness, or prior private worktree were imported.**

## Original architectural authority

1. Raw/source history remains in host. MR-Mem stores immutable, admitted Canonical SemanticBlocks with native versioned SourceRefs. Raw payload must not appear in Thread or LCE.
2. Thread is a *temporary projection* of Canonical memory IDs; it does not own independent fact truth. A semantic producer supplies stable thread_key plus explicit track/mature/abandon judgment. No question-string equivalence and no forced SemanticPoint intermediary.
3. LCE Path A consumes a mature semantic understanding and exact Canonical IDs, writes a Draft and immutable Baseline revision. It is another projection, and does **not** recompile raw text.
4. Readback revalidates Canonical scope, lifecycle and original versioned SourceRefs. If a source is invalidated, a derived Baseline cannot surface.
5. Path B/Point Cloud/Line discovery is not imported or invoked by this adapter. Path B can be added as a separate projection consumer later.

## What has been coded

- `src/mr_mem/memory/product.py`: opt-in explicit semantic maturity basis; old multi-source rule preserved for legacy callers. A fully understood batch SemanticBlock may mature without fictional multi-turn support.
- `path_a.py`: explicit semantic updates over committed Memory IDs, bounded support ID projection, replay checks, durable pending handoff and recovery; successful LCE publication retires active Thread.
- `path_a_lce.py`: direct read-only LCE MemorySubstratePort against MR-Mem Canonical; LCE PrecomputedDraftIntake + SqliteBaselineStore + DraftRevisionStore; query of Baseline HEAD with active-source revalidation.
- `path_a_runtime.py`: semantic batch Add seam (real native admission before projection), and Search exposing D0 canonical results **separately** from Baseline projections. No K=5 fan-out, gold cues, rerank parameter sweep or provider-prompt manipulation.
- `tests/test_path_a_projection.py`, `tests/test_path_a_runtime.py`, `tests/test_path_a_lce_integration.py`; CI also checks pinned LCE master 5228475d4cce7403078963695e2a98776f4fec24.

## What is NOT claimed, and must be audited

- **No agent/LLM semantic producer is added**: caller must supply valid SemanticMemoryCandidate + ThreadVerdict. That boundary is intentional to keep semantic authority at the host. Claude should assess whether the chosen interface gives enough stable identity and maturity evidence.
- Public MR-Mem repository contains **no published AML adapter**. An `/add` HTTP endpoint, official AML `/search` JSON schema, Answer model and rubric scorer have NOT been run in this PR. This is a canonical Path A test base, not an AML score.
- Baseline Search currently uses conservative lexical matching of Baseline + authorized support content. It is not a benchmark-tuned high-recall ranker and gives **no claim of +15–20pt**.
- LCE integration is pinned to the public LCE master, not Grok's 5 unpublished commits. No dependency on the earlier private worktree.
- Cross-connection atomicity of Canonical admission, Thread projection and LCE commit needs failure-injection review; one idempotent event journal exists, but commit ordering, concurrency and multi-process locking require hostile testing before production.
- `MemoryProductStore` still exposes historical `ThreadAutoUpdateService` lexical auto-matching for legacy users. The new Path A composition **does not call it**. Claude should confirm no accidental dual projector is activated by an actual adapter.
- SourceRefs are versioned pointers, not copied raw; confirm end-to-end ability to resolve them through the host source system when that adapter is connected.
- Potential lifecycle edge: same thread_key reopening after mature handoff, support-reference cleanup, stale snapshot and revision recheck require dedicated adversarial tests.
- No benchmarks or large-scale full-cycle tests have been run on this newly created implementation. Previous Grok-derived 22-query scores MUST NOT be attributed to it.

## Claude review checklist

1. Trace `SemanticMemoryCandidate -> SemanticAdmissionService.admit -> CanonicalMemoryStore -> PathASemanticWriter -> PathAProjector -> MemoryProductStore -> LcePathAProjection -> LCE Draft -> LCE Baseline -> PathASearch`.
2. Check whether every derived claim/ID has a valid provenance chain to native SourceRef and whether invalidation fails closed.
3. Verify batch vs per-turn semantics, explicit maturity without synthetic support, stable Thread identity across batches, and lifecycle changes.
4. Verify crash windows, duplicate/reordered updates, source invalidation between validate and publish, and scope isolation.
5. Verify none of the old Grok harness, prefilled gold evidence, Path B runtime, or fake baseline source corpus is used.
6. Decide whether official AML Add/Search and Answer are really connected; they are currently **not**. Treat missing connections as unfinished work, not test passes.
7. Require cost accounting for actual semantic compilation/update/search before judging engineering value.

Review without automatically making fixes. Findings should distinguish demonstrated bugs, missing integration, and design decisions.
