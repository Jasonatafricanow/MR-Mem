"""Host-side WorkBench bindings; no business state or authentication lives in MR-Mem.

The host authenticates the calling human and authorizes access before creating a
context. Agent/LLM arguments never select a memory owner or workspace scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from mr_mem.contracts import Scope, ScopeDomain
from mr_mem.memory.source import SourceRef, SourceRefReader


def _nonblank(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty")
    return value


class UserContextAuthorizer(Protocol):
    """Implemented by the host's current ACL/identity service."""

    def assert_allowed(
        self, *, principal_id: str, workspace_id: str, user_id: str, action: str
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class BoundUserContext:
    principal_id: str
    workspace_id: str
    user_id: str
    scope: Scope


class UserContextBinder:
    """Fail closed unless the host approves this principal and target user."""

    def __init__(self, authorizer: UserContextAuthorizer) -> None:
        self._authorizer = authorizer

    def bind(
        self, *, authenticated_principal_id: str, workspace_id: str, target_user_id: str
    ) -> BoundUserContext:
        principal = _nonblank(authenticated_principal_id, "authenticated_principal_id")
        workspace = _nonblank(workspace_id, "workspace_id")
        target = _nonblank(target_user_id, "target_user_id")
        self._authorizer.assert_allowed(
            principal_id=principal,
            workspace_id=workspace,
            user_id=target,
            action="memory.read",
        )
        return BoundUserContext(
            principal_id=principal, workspace_id=workspace,
            user_id=target, scope=Scope(ScopeDomain.USER, user_id=target),
        )


class WorkbenchEventResolver(Protocol):
    """Host-owned SQL source adapter. Never reconstruct from MR memory."""

    def current_event_ref(self, *, workspace_id: str, event_id: str) -> SourceRef | None: ...

    def render_event(self, *, workspace_id: str, event_id: str) -> str | None: ...


class WorkbenchEventSourceReader(SourceRefReader):
    """Scoped SourceRef verification and optional on-demand event hydration.

    Only accepted, current WorkBench event metadata may be admitted. The
    caller's user context is already authorized by UserContextBinder.
    """

    NAMESPACE = "agent-workbench.task-event"

    def __init__(self, context: BoundUserContext, resolver: WorkbenchEventResolver):
        self._context, self._resolver = context, resolver

    def _matches(self, scope: Scope, ref: SourceRef) -> bool:
        return (
            scope == self._context.scope
            and ref.source_namespace == self.NAMESPACE
            and ref.session_id == self._context.workspace_id
            and bool(ref.record_id)
        )

    def current_ref(self, scope: Scope, ref: SourceRef) -> SourceRef | None:
        if not self._matches(scope, ref):
            return None
        current = self._resolver.current_event_ref(
            workspace_id=self._context.workspace_id, event_id=ref.record_id
        )
        return current if current == ref else None

    def render_source(self, scope: Scope, ref: SourceRef) -> str | None:
        if self.current_ref(scope, ref) is None:
            return None
        return self._resolver.render_event(
            workspace_id=self._context.workspace_id, event_id=ref.record_id
        )
