"""Provider-neutral source records consumed by Memory admission."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from mr_mem.contracts import Scope


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
    """Read-only proof that source Evidence/Observation are durably admitted."""

    def find_evidence(
        self, scope: Scope, evidence_id: str
    ) -> tuple[SourceEvidence, str] | None: ...

    def find_observation(
        self, scope: Scope, observation_id: str
    ) -> SourceObservation | None: ...


@runtime_checkable
class Clock(Protocol):
    """Provider-neutral wall clock used to timestamp admission eligibility."""

    def now(self): ...
