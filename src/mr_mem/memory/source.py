"""Native source identity and legacy factual-admission compatibility contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol, runtime_checkable

from mr_mem.contracts import Scope
from mr_mem.contracts.common import require_aware_utc, require_non_empty


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Metadata only. Source revisions share one independent support identity."""

    source_namespace: str
    session_id: str
    record_id: str
    occurred_at: datetime
    revision: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_namespace", "session_id", "record_id"):
            require_non_empty(getattr(self, name), name)
        require_aware_utc(self.occurred_at, "occurred_at")
        for name in ("revision", "fingerprint"):
            if getattr(self, name) is not None:
                require_non_empty(getattr(self, name), name)

    @property
    def source_key(self) -> str:
        return json.dumps(
            [self.source_namespace, self.session_id, self.record_id], ensure_ascii=False
        )

    @property
    def version_key(self) -> str:
        data = [self.source_key, self.occurred_at.isoformat(), self.revision, self.fingerprint]
        return hashlib.sha256(json.dumps(data, ensure_ascii=False).encode()).hexdigest()


class SourceRefReader(Protocol):
    """Verify exact current metadata within scope without reading/copying raw."""

    def current_ref(self, scope: Scope, ref: SourceRef) -> SourceRef | None: ...


@runtime_checkable
class SourceEvidence(Protocol):
    """Minimum factual source shape required by MR-Mem admission."""

    @property
    def id(self) -> str: ...

    @property
    def scope(self) -> Scope: ...

    @property
    def payload(self) -> object: ...


@runtime_checkable
class SourceObservation(Protocol):
    """Minimum evidence-backed observation shape required by MR-Mem admission."""

    @property
    def id(self) -> str: ...

    @property
    def interaction_id(self) -> str: ...

    @property
    def scope(self) -> Scope: ...

    @property
    def evidence_refs(self) -> tuple[str, ...]: ...


class AdmissionDisposition(StrEnum):
    """Durable source admission outcome."""

    NEW = "new"
    REPAIRED = "repaired"
    REPLAY = "replay"


@dataclass(frozen=True, slots=True)
class MemoryAdmissionResult:
    """Provider-neutral admission result consumed by Memory governance."""

    observation: SourceObservation
    disposition: AdmissionDisposition


@runtime_checkable
class DurableFactReader(Protocol):
    """Legacy adapter, not a prerequisite for native semantic admission."""

    def find_evidence(
        self, scope: Scope, evidence_id: str
    ) -> tuple[SourceEvidence, str] | None: ...

    def find_observation(
        self, scope: Scope, observation_id: str
    ) -> SourceObservation | None: ...


@runtime_checkable
class Clock(Protocol):
    """Provider-neutral wall clock used to timestamp admission eligibility."""

    def now(self) -> datetime: ...
