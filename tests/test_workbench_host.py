"""Contract tests for WorkBench user/source bindings (offline, no LLM)."""
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
        self.grants = {("owner", "personal", "owner", "memory.read")}

    def assert_allowed(self, *, principal_id, workspace_id, user_id, action):
        if (principal_id, workspace_id, user_id, action) not in self.grants:
            raise PermissionError("denied")


class Resolver:
    def __init__(self, ref):
        self.ref = ref
        self.reads = []

    def current_event_ref(self, *, context, event_id):
        self.reads.append(("ref", context, event_id))
        return self.ref

    def render_event(self, *, context, event_id):
        self.reads.append(("render", context, event_id))
        return "Status updated"


class HostContractTests(unittest.TestCase):
    def setUp(self):
        self.authz = Authz()
        self.binder = UserContextBinder(self.authz)
        self.ctx = self.binder.bind(
            authenticated_principal_id="owner",
            workspace_id="personal",
            target_user_id="owner",
        )
        self.ref = SourceRef(
            source_namespace="agent-workbench.task-event",
            session_id="personal",
            record_id="evt-1",
            occurred_at=datetime(2026, 11, 16, tzinfo=UTC),
            revision="1",
        )
        self.resolver = Resolver(self.ref)
        self.reader = WorkbenchEventSourceReader(self.ctx, self.resolver, self.authz)

    def test_owner_context_is_user_scope(self):
        self.assertEqual(self.ctx.scope, Scope(ScopeDomain.USER, user_id="owner"))

    def test_requester_cannot_select_owner_memory(self):
        with self.assertRaises(PermissionError):
            self.binder.bind(
                authenticated_principal_id="requester",
                workspace_id="personal",
                target_user_id="owner",
            )

    def test_invalid_identity_and_capability_rejected(self):
        with self.assertRaises(ValueError):
            self.binder.bind(
                authenticated_principal_id="",
                workspace_id="personal",
                target_user_id="owner",
            )
        with self.assertRaises(PermissionError):
            self.binder.bind(
                authenticated_principal_id="owner",
                workspace_id="personal",
                target_user_id="owner",
                capability="admin",
            )

    def test_exact_provenance_and_hydration(self):
        self.assertEqual(self.reader.current_ref(self.ctx.scope, self.ref), self.ref)
        self.assertEqual(self.reader.render_source(self.ctx.scope, self.ref), "Status updated")
        self.assertTrue(all(call[1] == self.ctx for call in self.resolver.reads))

    def test_cross_workspace_and_scope_rejected_before_resolve(self):
        foreign = SourceRef(
            source_namespace=self.ref.source_namespace,
            session_id="elsewhere",
            record_id="evt-1",
            occurred_at=self.ref.occurred_at,
            revision="1",
        )
        self.assertIsNone(self.reader.current_ref(self.ctx.scope, foreign))
        self.assertIsNone(
            self.reader.current_ref(Scope(ScopeDomain.USER, user_id="other"), self.ref)
        )
        self.assertEqual(self.resolver.reads, [])

    def test_stale_revision_and_foreign_namespace_rejected(self):
        stale = SourceRef(
            source_namespace=self.ref.source_namespace,
            session_id="personal",
            record_id="evt-1",
            occurred_at=self.ref.occurred_at,
            revision="0",
        )
        self.assertIsNone(self.reader.current_ref(self.ctx.scope, stale))
        foreign = SourceRef(
            source_namespace="telegram",
            session_id="personal",
            record_id="evt-1",
            occurred_at=self.ref.occurred_at,
        )
        reads_before = len(self.resolver.reads)
        self.assertIsNone(self.reader.current_ref(self.ctx.scope, foreign))
        self.assertEqual(len(self.resolver.reads), reads_before)

    def test_revoke_after_bind_blocks_all_subsequent_reads(self):
        self.assertEqual(self.reader.current_ref(self.ctx.scope, self.ref), self.ref)
        self.authz.grants.clear()
        with self.assertRaises(PermissionError):
            self.reader.current_ref(self.ctx.scope, self.ref)
        with self.assertRaises(PermissionError):
            self.reader.render_source(self.ctx.scope, self.ref)
        self.assertEqual([x[0] for x in self.resolver.reads], ["ref"])

    def test_write_binding_is_separate_permission(self):
        with self.assertRaises(PermissionError):
            self.binder.bind(
                authenticated_principal_id="owner",
                workspace_id="personal",
                target_user_id="owner",
                capability="memory.write",
            )
        self.authz.grants.add(("owner", "personal", "owner", "memory.write"))
        writable = self.binder.bind(
            authenticated_principal_id="owner",
            workspace_id="personal",
            target_user_id="owner",
            capability="memory.write",
        )
        writer_reader = WorkbenchEventSourceReader(writable, self.resolver, self.authz)
        self.assertEqual(writer_reader.current_ref(writable.scope, self.ref), self.ref)
        self.authz.grants.remove(("owner", "personal", "owner", "memory.write"))
        with self.assertRaises(PermissionError):
            writer_reader.current_ref(writable.scope, self.ref)


if __name__ == "__main__":
    unittest.main()
