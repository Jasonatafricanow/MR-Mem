from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from mr_mem import Scope, ScopeDomain
from mr_mem.memory.admission import MemoryAdmissionService
from mr_mem.memory.extraction import DeterministicExtractor
from mr_mem.memory.source import AdmissionDisposition, MemoryAdmissionResult
from mr_mem.memory.store import CanonicalMemoryStore


NOW = datetime(2026, 9, 29, 9, 30, tzinfo=UTC)


@dataclass(frozen=True)
class Evidence:
    id: str
    scope: Scope
    payload: object


@dataclass(frozen=True)
class Observation:
    id: str
    interaction_id: str
    scope: Scope
    evidence_refs: tuple[str, ...]


class FixedClock:
    def now(self):
        return NOW


class Facts:
    def __init__(self, evidence: Evidence, observation: Observation):
        self.evidence = evidence
        self.observation = observation

    def find_evidence(self, scope, evidence_id):
        if scope == self.evidence.scope and evidence_id == self.evidence.id:
            return self.evidence, self.observation.interaction_id
        return None

    def find_observation(self, scope, observation_id):
        if scope == self.observation.scope and observation_id == self.observation.id:
            return self.observation
        return None


def fixture(tmp_path):
    scope = Scope(ScopeDomain.USER, user_id="user-1")
    evidence = Evidence("evidence-1", scope, {"text": "hello"})
    observation = Observation(
        "observation-evidence-1", "interaction-1", scope, ("evidence-1",)
    )
    store = CanonicalMemoryStore(tmp_path / "memory.sqlite")
    service = MemoryAdmissionService(
        store=store,
        facts=Facts(evidence, observation),
        clock=FixedClock(),
        origin_runtime_id="runtime-1",
        enabled=True,
    )
    return evidence, observation, store, service


def test_new_source_admission_commits_canonical_memory(tmp_path):
    evidence, observation, store, service = fixture(tmp_path)
    service.after_admission(
        evidence, MemoryAdmissionResult(observation, AdmissionDisposition.NEW)
    )
    rows = store.load_all()
    assert len(rows) == 1
    assert rows[0].content == "hello"
    assert rows[0].provenance.evidence_refs == ("evidence-1",)
    assert rows[0].provenance.interaction_id == "interaction-1"


def test_replay_is_idempotent_after_registered_new(tmp_path):
    evidence, observation, store, service = fixture(tmp_path)
    service.after_admission(
        evidence, MemoryAdmissionResult(observation, AdmissionDisposition.NEW)
    )
    original = store.load_all()
    service.after_admission(
        evidence, MemoryAdmissionResult(observation, AdmissionDisposition.REPLAY)
    )
    assert store.load_all() == original


def test_replay_does_not_backfill_without_original_new(tmp_path):
    evidence, observation, store, service = fixture(tmp_path)
    service.after_admission(
        evidence, MemoryAdmissionResult(observation, AdmissionDisposition.REPLAY)
    )
    assert store.load_all() == ()


def test_non_durable_or_mismatched_pair_fails_closed(tmp_path):
    evidence, observation, store, service = fixture(tmp_path)
    forged = replace(observation, interaction_id="forged")
    with pytest.raises(ValueError, match="exact durable admitted"):
        service.after_admission(
            evidence, MemoryAdmissionResult(forged, AdmissionDisposition.NEW)
        )
    assert store.load_all() == ()


def test_extractor_cannot_exceed_admitted_source_authority(tmp_path):
    evidence, observation, store, _ = fixture(tmp_path)

    class ForgedExtractor:
        def extract(self, evidence, observation):
            candidate = DeterministicExtractor().extract(evidence, observation)[0]
            return (
                replace(
                    candidate,
                    provenance=replace(
                        candidate.provenance, evidence_refs=("unknown-evidence",)
                    ),
                ),
            )

    service = MemoryAdmissionService(
        store=store,
        facts=Facts(evidence, observation),
        clock=FixedClock(),
        origin_runtime_id="runtime-1",
        enabled=True,
        extractor=ForgedExtractor(),
    )
    with pytest.raises(ValueError, match="source authority"):
        service.after_admission(
            evidence, MemoryAdmissionResult(observation, AdmissionDisposition.NEW)
        )
    assert store.load_all() == ()


def test_upstream_result_can_be_structurally_adapted(tmp_path):
    evidence, observation, store, service = fixture(tmp_path)

    @dataclass(frozen=True)
    class ExistingHostResult:
        observation: Observation
        disposition: str

    service.after_admission(evidence, ExistingHostResult(observation, "new"))  # type: ignore[arg-type]
    assert len(store.load_all()) == 1
