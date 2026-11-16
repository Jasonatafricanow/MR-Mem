"""Contract tests for the user/workspace host adapter. No SQL or LLM calls."""
from __future__ import annotations

import unittest
from datetime import UTC, datetime

from mr_mem.contracts import Scope, ScopeDomain
from mr_mem.integrations.workbench_host import (
    UserContextBinder,
    WorkbenchEventSourceReader,
)
from mr_mem.memory.source import SourceRef


class Authz:
    def __init__(self):
        self.grants = {("owner", "personal", "owner")}

    def assert_allowed(self, *, principal_id, workspace_id, user_id, action):
        assert action == "memory.read"
        if (principal_id, workspace_id, user_id) not in self.grants:
            raise PermissionError("denied")


class Resolver:
    def __init__(self, ref):
        self.ref = ref
        self.reads = []

    def current_event_ref(self, *, workspace_id, event_id):
        self.reads.append((workspace_id, event_id))
        return self.ref

    def render_event(self, *, workspace_id, event_id):
        return "Status updated"


class HostContractTests(unittest.TestCase):
    def setUp(self):
        self.binder = UserContextBinder(Authz())
        self.ctx = self.binder.bind(
            authenticated_principal_id="owner", workspace_id="personal",
            target_user_id="owner",
        )
        self.ref = SourceRef(
            source_namespace="agent-workbench.task-event",
            session_id="personal", record_id="evt-1",
            occurred_at=datetime(2026, 11, 16, tzinfo=UTC),
            revision="1",
        )
        self.resolver = Resolver(self.ref)
        self.reader = WorkbenchEventSourceReader(self.ctx, self.resolver)

    def test_owner_context_is_user_scope(self):
        self.assertEqual(self.ctx.scope, Scope(ScopeDomain.USER, user_id="owner"))

    def test_requester_cannot_select_owner_memory(self):
        with self.assertRaises(PermissionError):
            self.binder.bind(
                authenticated_principal_id="requester",
                workspace_id="personal", target_user_id="owner",
            )

    def test_invalid_identity_rejected(self):
        with self.assertRaises(ValueError):
            self.binder.bind(
                authenticated_principal_id="", workspace_id="personal",
                target_user_id="owner",
            )

    def test_exact_provenance_and_hydration(self):
        self.assertEqual(self.reader.current_ref(self.ctx.scope, self.ref), self.ref)
        self.assertEqual(self.reader.render_source(self.ctx.scope, self.ref), "Status updated")

    def test_cross_workspace_and_scope_rejected_before_resolve(self):
        foreign = SourceRef(
            source_namespace=self.ref.source_namespace, session_id="elsewhere",
            record_id="evt-1", occurred_at=self.ref.occurred_at, revision="1",
        )
        self.assertIsNone(self.reader.current_ref(self.ctx.scope, foreign))
        self.assertIsNone(
            self.reader.current_ref(Scope(ScopeDomain.USER, user_id="other"), self.ref)
        )
        self.assertEqual(self.resolver.reads, [])

    def test_stale_revision_and_foreign_namespace_rejected(self):
        stale = SourceRef(
            source_namespace=self.ref.source_namespace, session_id="personal",
            record_id="evt-1", occurred_at=self.ref.occurred_at, revision="0",
        )
        self.assertIsNone(self.reader.current_ref(self.ctx.scope, stale))
        foreign = SourceRef(
            source_namespace="telegram", session_id="personal",
            record_id="evt-1", occurred_at=self.ref.occurred_at,
        )
        reads_before = len(self.resolver.reads)
        self.assertIsNone(self.reader.current_ref(self.ctx.scope, foreign))
        self.assertEqual(len(self.resolver.reads), reads_before)


if __name__ == "__main__":
    unittest.main()
