# ADR-0002 — Semantic Delta V1 native source admission
- Status: ACCEPTED / FROZEN
- Date: 2026-10-02 (Asia/Shanghai)
- Authority: explicit user decision in the Semantic Delta V1 implementation assignment
- Supersedes: the earlier PROPOSED source-contract draft; SOURCE_PROVENANCE_AND_ADMISSION_CONTRACT is resolved
- Parent: ADR-0001-SIDECAR-MEMORY-BOUNDARY.md

## Decision
1. Semantic Delta V1 uses native SourceRef. Native hosts need no literal Evidence/Observation pair.
2. The system binds source identity and verifies exact current source, scope, revision/version, and interaction association. Body cannot supply this authority.
3. SemanticAdmissionService owns native Semantic Delta admission.
4. MemoryAdmissionService remains the legacy factual pair adapter. Native sources must neither pass through it nor fabricate observation/evidence identities.
5. CanonicalMemoryStore remains the sole canonical Memory writer/store. Host, MR Core, and semantic closure have no bypass writer.
6. The only approved prerequisite from fix/source-ref-semantic-admission is 1ec6518fa63410969a0887af0be1684ade9583b2 (native-backed semantic state without raw mirrors). Do not import 0b887ff9 (Thread handoff replay) or a79993a (Thread question identity). The 1711fad typed-package metadata change is permitted only as an independent mechanical build prerequisite if necessary.

## Boundaries retained
- One Body inference produces response and current semantic delta as sibling outputs.
- Strict validation, DEFER exclusion, deterministic cohabit closure, context, and supersede retain the frozen V1 meanings.
- Only committed SemanticBlocks enter embedding.
- Canonical commit, lifecycle mutation, relations, and receipts remain atomic; accepted compilation is durable and reused on replay/restart.
- MindRuntimeHostPort and the MR cognition pipeline are not widened.
- Thread/LCE remain downstream projections; LCE algorithms are unchanged.
- CommittedMemory and the existing identity/job mechanism remain canonical. No new identity scheme, model call, ontology, or research route is authorized.

## Implementation order
Refresh main and apply only the approved prerequisite. Then implement and verify separately:
PR-1 contracts/validator -> PR-2 closure -> PR-3 persistence/admission -> PR-4 MemoryCore/replay/recovery -> PR-5 host one-pass composition -> PR-6 feature flag/integration regression.

Source-route selection is closed. Do not reopen native versus factual-pair comparisons.

## Provenance
The original frozen work order SHA-256 is F271014A998A9BB90AAED837694350BBDDD62DC0BDEF6D5E73EE266883E9733E. This decision narrowly replaces its mandatory factual pair/native admission ownership assumptions in sections 22-24. All other frozen requirements remain in force.
