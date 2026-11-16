# WorkBench host/user-context adapter contract

> Detailed rationale, state authority and full flow: [WORKBENCH_DESIGN_RATIONALE.md](WORKBENCH_DESIGN_RATIONALE.md).

Status: draft integration adapter, **not connected to production Agent Host**.
Aligns with MR-Mem ADR-0001 (source authority stays with the host).

## Three-state ownership

- **Agent-WorkBench SQL** owns business tasks, durable business-event inbox,
  timers, approvals, outgoing delivery intents and task history.
- **Agent Host** owns Sessions/Runs, tool invocation/suspension/resumption and
  execution credentials. MR's cognitive WakeSignal is not a host Run.
- **Mind Runtime** owns cognition, state/intent/policy projections and
  cognitive wake semantics; it does NOT become the transport or Run scheduler.
- **MR-Mem** owns accepted semantic memory, retrieval, Thread/LCE and provenance;
  neither business task state nor Agent Session can be restored from memory.

Reference: Anthropic, *Building always-on agents*, 2026-11-13,
https://claude.com/resources/articles/building-always-on-agents

## Scoped host adapter

- Host must authenticate the principal from verified transport metadata;
  an LLM-supplied ID/role/token is not an identity claim.
- `UserContextBinder.bind` authorizes exactly one `Scope(USER, user_id)` and
  one `memory.read` or `memory.write` capability. It does not grant a
  permanent token.
- `WorkbenchEventSourceReader` takes an explicit current `authorizer` and
  rechecks its authorization on **every** `current_ref` and `render_source`.
  Revocation after context creation must fail closed.
- Its `WorkbenchEventResolver` receives the verified `BoundUserContext`
  and MUST check task/event-level visibility, retention and deletion at SQL
  access time; a shared workspace does not imply full visibility.
- SourceRef namespace: `agent-workbench.task-event`, session ID is the
  WorkBench workspace, record ID is the task event ID, revision/fingerprint
  denotes the exact durable event. No second raw-transcript storage.
- Eligible, host-accepted semantic candidates may use
  `SemanticAdmissionService.admit` and existing Path A. Raw task updates
  and operational receipts do **not** automatically become semantic memories.
- Memory lag or failure does not block SQL task update, and a stale memory
  result does not override SQL current status.

## Event / Run handshake

WorkBench may submit a stable `event_id` with an `origin` and optional
`task_id`. The Agent Host returns an opaque `run_ref`/`session_ref`
and a **Run outcome**. WorkBench stores only references and business-visible
outcomes, never serialized Host Session or tool call internals.

For an approval-deferred tool call: Host persists the Session and deferred
tool token, WorkBench records an approval request and notification, and the
authorized response creates a new event. Only the Host can resume the saved
Session and re-check tool permission. A business `WAITING` state does not
mean the Agent Host is itself in a suspended tool call.

**Operational feedback is not experiential authority.** Successful tool
delivery/acknowledgement alone must not be admitted as a user emotion,
preference, changed relationship or long-term user commitment.

## Tests

`python -m unittest tests.test_workbench_host`

Offline tests include default-deny, read/write isolation, cross-workspace
source checks, stale revision rejection and grant revocation after binding.
No live Agent Host, model or LCE integration is claimed.


## Progressive Delegation boundary

Normative WorkBench business contract: [ADR-0003 — Progressive Delegation](https://github.com/Jasonatafricanow/Agent-WorkBench/blob/feat/modular-backend-20261116/docs/ADR-0003-PROGRESSIVE-DELEGATION.md).

Agent-WorkBench owns **business responsibility transfer**: the accountable
owner, current executor, work class, versioned autonomy ceiling, action
allowlist, mandatory human decisions, promotion/demotion history and Human
Action Inbox. The human is a first-class participant in the WorkBench work
loop, but is not required to be online for work already inside an approved
execution envelope.

MR-Mem may retrieve historical preferences and evidence that help the Host
interpret a task or recommend a work-class policy. It MUST NOT infer, create,
widen, approve or promote execution authority. Memory is contextual evidence,
not a capability or consent token. An old semantic memory that says "the user
usually lets the Agent do this" cannot substitute for a current WorkBench
delegation profile or authenticated approval.

The Agent Host remains responsible for actual tool permissions and must
revalidate them at each tool call and deferred resume. Mind Runtime may
influence cognitive attention or recommend a review, but cannot modify
WorkBench delegation policy. WorkBench SQL is the authority for the current
responsibility assignment and policy version.
