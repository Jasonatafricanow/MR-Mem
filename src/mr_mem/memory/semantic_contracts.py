"""Body proposal values. Construction grants no source or canonical authority."""

from dataclasses import dataclass
from enum import StrEnum

SCHEMA_VERSION = "semantic_delta_v1"
COMPILER_VERSION = "semantic-delta-compiler-v1"


class SemanticPointStatus(StrEnum):
    RESOLVED = "resolved"
    DEFER = "defer"


class SpeechAct(StrEnum):
    ASSERTION = "assertion"
    QUESTION = "question"
    DIRECTIVE = "directive"
    OTHER = "other"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"


class EpistemicStatus(StrEnum):
    ASSERTED = "asserted"
    UNCERTAIN = "uncertain"
    HYPOTHETICAL = "hypothetical"
    COUNTERFACTUAL = "counterfactual"
    PLANNED = "planned"
    REPORTED = "reported"
    UNKNOWN = "unknown"


class TemporalScope(StrEnum):
    PAST = "past"
    CURRENT = "current"
    FUTURE = "future"
    ATEMPORAL = "atemporal"
    UNKNOWN = "unknown"


class DependencyTargetKind(StrEnum):
    POINT = "point"
    MEMORY = "memory"


class BoundaryPolicy(StrEnum):
    COHABIT = "cohabit"
    CONTEXT = "context"


class LifecycleEffect(StrEnum):
    NONE = "none"
    SUPERSEDE = "supersede"


class SemanticDeltaErrorCode(StrEnum):
    INVALID_SCHEMA = "SEMANTIC_DELTA_INVALID_SCHEMA"
    UNKNOWN_POINT = "SEMANTIC_DELTA_UNKNOWN_POINT"
    UNKNOWN_MEMORY = "SEMANTIC_DELTA_UNKNOWN_MEMORY"
    SCOPE_MISMATCH = "SEMANTIC_DELTA_SCOPE_MISMATCH"
    INVALID_DEPENDENCY = "SEMANTIC_DELTA_INVALID_DEPENDENCY"
    DEFERRED = "SEMANTIC_DELTA_DEFERRED"
    COMMIT_FAILED = "SEMANTIC_DELTA_COMMIT_FAILED"
    REPLAY = "SEMANTIC_DELTA_REPLAY"


class SemanticDeltaError(ValueError):
    def __init__(self, code: SemanticDeltaErrorCode, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


@dataclass(frozen=True, slots=True)
class DeltaSemanticPoint:
    point_id: str
    meaning: str
    status: SemanticPointStatus
    speech_act: SpeechAct
    polarity: Polarity
    epistemic_status: EpistemicStatus
    temporal_scope: TemporalScope
    temporal_expression: str | None = None
    unresolved_refs: tuple[str, ...] = ()

    @property
    def eligible(self) -> bool:
        return self.status is SemanticPointStatus.RESOLVED and not self.unresolved_refs


@dataclass(frozen=True, slots=True)
class DeltaSemanticDependency:
    from_point_id: str
    target_kind: DependencyTargetKind
    target_id: str
    relation: str
    boundary_policy: BoundaryPolicy
    lifecycle_effect: LifecycleEffect


@dataclass(frozen=True, slots=True)
class SemanticDeltaV1:
    schema_version: str
    points: tuple[DeltaSemanticPoint, ...]
    dependencies: tuple[DeltaSemanticDependency, ...]
