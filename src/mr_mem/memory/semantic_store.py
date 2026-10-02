"""System-bound source and semantic persistence values, never Body output."""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Protocol

from mr_mem.contracts import Scope, ScopeDomain
from mr_mem.contracts.common import require_non_empty
from mr_mem.memory.source import SourceRef, SourceRefReader

SEMANTIC_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS semantic_compilations (
    source_key TEXT PRIMARY KEY REFERENCES admission_jobs(source_key),
    accepted TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'frozen');
CREATE TABLE IF NOT EXISTS semantic_block_metadata (
    memory_id TEXT PRIMARY KEY REFERENCES canonical_memory(memory_id),
    payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS semantic_relations (
    from_memory_id TEXT NOT NULL REFERENCES canonical_memory(memory_id),
    to_memory_id TEXT NOT NULL REFERENCES canonical_memory(memory_id),
    relation TEXT NOT NULL, boundary_policy TEXT NOT NULL,
    lifecycle_effect TEXT NOT NULL, interaction_id TEXT NOT NULL,
    PRIMARY KEY(from_memory_id,to_memory_id,relation,boundary_policy,
                lifecycle_effect,interaction_id));
"""


@dataclass(frozen=True, slots=True)
class SemanticSourceBinding:
    scope: Scope
    interaction_id: str
    source_ref: SourceRef

    def __post_init__(self):
        require_non_empty(self.interaction_id, "interaction_id")
        if not isinstance(self.scope, Scope) or not isinstance(self.source_ref, SourceRef):
            raise ValueError("system Scope and SourceRef required")
        if not (self.source_ref.revision or self.source_ref.fingerprint):
            raise ValueError("native delta source requires revision or fingerprint")


class SemanticDeltaSourceReader(SourceRefReader, Protocol):
    def current_user_source(self, scope: Scope, interaction_id: str) -> SourceRef | None:
        """Resolve the exact current durable USER record associated with this turn."""
        ...


@dataclass(frozen=True, slots=True)
class SemanticBlockMetadata:
    memory_id: str
    schema_version: str
    compiler_version: str
    member_point_ids: tuple[str, ...]
    context_memory_ids: tuple[str, ...]
    source_interaction_id: str
    source_refs: tuple[SourceRef, ...]


@dataclass(frozen=True, slots=True)
class SemanticDeltaReceipt:
    interaction_id: str
    source_ref: SourceRef
    memory_ids: tuple[str, ...]
    status: str


def binding_payload(binding: SemanticSourceBinding) -> dict:
    data = asdict(binding)
    data["source_ref"]["occurred_at"] = binding.source_ref.occurred_at.isoformat()
    return data


def read_binding(data: dict) -> SemanticSourceBinding:
    scope = dict(data["scope"])
    scope["domain"] = ScopeDomain(scope["domain"])
    ref = dict(data["source_ref"])
    ref["occurred_at"] = datetime.fromisoformat(ref["occurred_at"])
    return SemanticSourceBinding(Scope(**scope), data["interaction_id"], SourceRef(**ref))


def read_metadata(payload: str) -> SemanticBlockMetadata:
    data = json.loads(payload)
    refs = []
    for ref in data["source_refs"]:
        ref["occurred_at"] = datetime.fromisoformat(ref["occurred_at"])
        refs.append(SourceRef(**ref))
    return SemanticBlockMetadata(
        data["memory_id"],
        data["schema_version"],
        data["compiler_version"],
        tuple(data["member_point_ids"]),
        tuple(data["context_memory_ids"]),
        data["source_interaction_id"],
        tuple(refs),
    )
