"""Optional WorkBench host/source adapter; no business state or Agent Run ownership.

The host authenticates principals and authorizes scopes. A cached BoundUserContext
is not an irrevocable grant: every source read rechecks current permissions.
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
    capability: str = "memory.read"


class UserContextBinder:
    """Bind the authenticated principal to a specific USER scope, fail closed."""

    def __init__(self, authorizer: UserContextAuthorizer) -> None:
        self._authorizer = authorizer

    def bind(
        self,
        *,
        authenticated_principal_id: str,
        workspace_id: str,
        target_user_id: str,
        capability: str = "memory.read",
    ) -> BoundUserContext:
        principal = _nonblank(authenticated_principal_id, "authenticated_principal_id")
        workspace = _nonblank(workspace_id, "workspace_id")
        target = _nonblank(target_user_id, "target_user_id")
        if capability not in ("memory.read", "memory.write"):
            raise PermissionError("unsupported memory capability")
        self._authorizer.assert_allowed(
            principal_id=principal,
            workspace_id=workspace,
            user_id=target,
            action=capability,
        )
        return BoundUserContext(
            principal_id=principal,
            workspace_id=workspace,
            user_id=target,
            scope=Scope(ScopeDomain.USER, user_id=target),
            capability=capability,
        )


class WorkbenchEventResolver(Protocol):
    """Host-owned SQL resolver; it MUST enforce task/event-level visibility.

    The context is a verified, immutable identity binding. Host SQL must
    still check event ownership/visibility against it, including at hydration.
    """

    def current_event_ref(
        self, *, context: BoundUserContext, event_id: str
    ) -> SourceRef | None: ...

    def render_event(self, *, context: BoundUserContext, event_id: str) -> str | None: ...


class WorkbenchEventSourceReader(SourceRefReader):
    """Check exact provenance AND current host ACL before serving event evidence.

    A previously authorized context becomes invalid immediately after grant
    revocation. Permission errors fail closed, not as an empty source override.
    """

    NAMESPACE = "agent-workbench.task-event"

    def __init__(
        self,
        context: BoundUserContext,
        resolver: WorkbenchEventResolver,
        authorizer: UserContextAuthorizer,
    ) -> None:
        self._context = context
        self._resolver = resolver
        self._authorizer = authorizer

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
        self._authorizer.assert_allowed(
            principal_id=self._context.principal_id,
            workspace_id=self._context.workspace_id,
            user_id=self._context.user_id,
            action=self._context.capability,
        )
        current = self._resolver.current_event_ref(
            context=self._context, event_id=ref.record_id
        )
        return current if current == ref else None

    def render_source(self, scope: Scope, ref: SourceRef) -> str | None:
        if self.current_ref(scope, ref) is None:
            return None
        # A second check prevents a grant revoked during ref resolution from
        # allowing event hydration. SQL event visibility is checked by resolver.
        self._authorizer.assert_allowed(
            principal_id=self._context.principal_id,
            workspace_id=self._context.workspace_id,
            user_id=self._context.user_id,
            action=self._context.capability,
        )
        return self._resolver.render_event(
            context=self._context, event_id=ref.record_id
        )
