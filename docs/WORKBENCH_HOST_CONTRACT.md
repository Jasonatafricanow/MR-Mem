# WorkBench host/user-context adapter contract

Status: feature-branch integration preparation (not production-wired).

The user profile and task records belong to **Agent-WorkBench** SQL. This module
adds a narrow, optional adapter for existing MR-Mem host integrations. MR-Mem
does not own WorkBench tasks, Telegram sessions, credentials, user ACLs, or a
second raw-message ledger.

## Authority and isolation

- The host authenticates a principal (Telegram external ID is mapped server-side).
- The host's authorizer decides whether that principal may query a target
  user's memory inside a workspace. LLM text MUST NOT supply this identity.
- `UserContextBinder` returns an immutable `Scope(USER, user_id=...)` only
  after the authorizer grants `memory.read`. Default-deny.
- Each WorkBench event uses a SourceRef with:
  `source_namespace="agent-workbench.task-event"`,
  `session_id=<workspace_id>`, `record_id=<event_id>`, an aware UTC event
  time and durable source revision/fingerprint.
- `WorkbenchEventSourceReader.current_ref` rejects mismatched scope,
  namespace, workspace, deleted or revised events before admitting semantics.
  `render_source` hydrates host-owned content on demand.
- A public requester may create/read only permitted WorkBench tasks but has
  **no implicit right to the owner's MR-Mem USER scope**.
- If a host needs user-shared context, it must grant that access explicitly
  under its own ACL and with auditable consent. No owner impersonation.

## Integration path

1. WorkBench saves the authorized mutation and event in its transaction.
2. A durable outbox references the event and source revision.
3. Host resolves the authenticated actor and binds authorized user scope.
4. Host semantic authority compiles any eligible event using the already
   configured model. The resulting candidate carries the precise SourceRef.
5. Existing `SemanticAdmissionService.admit` validates the SourceRefReader
   and writes semantic memory. Thread/LCE behavior is unchanged.
6. Retry uses durable event ID, revision, compiler version and outbox receipt.
   Task SQL remains query authority regardless of memory projection lag.
7. For task-status questions, the host queries WorkBench SQL first; MR-Mem
   contributes optional semantic context, never a replacement for current state.

## Out of scope

No new model, no automatic admission of every task update, no AML /add reliance,
no change to frozen ADR-0001, no business database in MR-Mem, and no cross-user
search via untrusted agent input.

## Verification

`python -m unittest tests.test_workbench_host` (repository environment).

Contract tests cover default-deny authorization, user scope, cross-workspace
references, stale revisions and source hydration. Full live host wiring must be
verified in the separate WorkBench repository before deployment.
