"""Scope, authority, and ownership contracts."""

from dataclasses import dataclass
from enum import StrEnum


class ScopeDomain(StrEnum):
    """Supported runtime scope domains."""

    USER = "user"
    AGENT = "agent"
    RELATIONSHIP = "relationship"
    INTERACTION = "interaction"
    WORLD = "world"


_SCOPE_IDENTITY_FIELDS = (
    "user_id",
    "agent_id",
    "persona_id",
    "relationship_id",
    "world_id",
    "interaction_id",
)

_REQUIRED_SCOPE_FIELDS = {
    ScopeDomain.USER: frozenset({"user_id"}),
    ScopeDomain.AGENT: frozenset({"agent_id", "persona_id"}),
    ScopeDomain.RELATIONSHIP: frozenset({"relationship_id", "persona_id"}),
    ScopeDomain.INTERACTION: frozenset({"interaction_id"}),
    ScopeDomain.WORLD: frozenset({"world_id"}),
}


def _is_non_empty(value: str | None) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True, slots=True)
class Scope:
    """Structured, fail-closed identity for a runtime domain."""

    domain: ScopeDomain
    user_id: str | None = None
    agent_id: str | None = None
    persona_id: str | None = None
    relationship_id: str | None = None
    world_id: str | None = None
    interaction_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ScopeDomain):
            raise ValueError("domain must be a ScopeDomain")

        required = _REQUIRED_SCOPE_FIELDS[self.domain]
        for field_name in _SCOPE_IDENTITY_FIELDS:
            value = getattr(self, field_name)
            if field_name in required:
                if not _is_non_empty(value):
                    raise ValueError(f"{self.domain.value} scope requires non-empty {field_name}")
            elif value is not None:
                raise ValueError(f"{self.domain.value} scope forbids {field_name}")


class AuthorityLevel(StrEnum):
    """Strength of authority attached to a scope."""

    NONE = "none"
    ASSERTED = "asserted"
    OBSERVED = "observed"
    VERIFIED = "verified"
    SYSTEM = "system"


@dataclass(frozen=True, slots=True)
class Authority:
    """Authority level and its provenance for a scope."""

    scope: Scope
    level: AuthorityLevel
    source_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.level, AuthorityLevel):
            raise ValueError("level must be an AuthorityLevel")
        if self.level is AuthorityLevel.NONE:
            if self.source_id is not None:
                raise ValueError("NONE forbids source_id")
        elif not _is_non_empty(self.source_id):
            raise ValueError(f"{self.level.value} requires non-empty source_id")


class WritePolicy(StrEnum):
    """Declared write policy for a scope."""

    READ_ONLY = "read_only"
    SINGLE_WRITER = "single_writer"
    AUTHORIZED_WRITER = "authorized_writer"


@dataclass(frozen=True, slots=True)
class Ownership:
    """Static ownership declaration without runtime authorization behavior."""

    scope: Scope
    write_policy: WritePolicy
    owner_runtime_id: str | None = None
    owner_persona_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.write_policy, WritePolicy):
            raise ValueError("write_policy must be a WritePolicy")

        if self.scope.domain in {ScopeDomain.AGENT, ScopeDomain.RELATIONSHIP}:
            if self.owner_persona_id != self.scope.persona_id:
                raise ValueError("owner_persona_id must match scope.persona_id")
        elif self.owner_persona_id is not None:
            raise ValueError(f"{self.scope.domain.value} scope forbids owner_persona_id")

        if self.write_policy is not WritePolicy.READ_ONLY and not _is_non_empty(
            self.owner_runtime_id
        ):
            raise ValueError(f"{self.write_policy.value} requires non-empty owner_runtime_id")
        if self.owner_runtime_id is not None and not _is_non_empty(self.owner_runtime_id):
            raise ValueError("non-empty owner_runtime_id required")
