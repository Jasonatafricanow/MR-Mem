"""Shared synchronizable-object identity fields."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from mr_mem.contracts.scope import Scope


@dataclass(frozen=True, slots=True)
class SyncFields:
    """Identity fields every synchronizable object must carry."""

    scope: Scope
    origin_runtime_id: str
    object_id: str
    version: int
    idempotency_key: str


@runtime_checkable
class Syncable(Protocol):
    """A synchronizable object exposing its SyncFields identity."""

    def sync_fields(self) -> SyncFields:
        """Return the object's sync identity."""
        ...


type ImmutableValue = (
    None
    | bool
    | int
    | float
    | str
    | bytes
    | tuple[ImmutableValue, ...]
    | frozenset[ImmutableValue]
    | FrozenMapping
)


@dataclass(frozen=True, slots=True, eq=False)
class FrozenMapping(Mapping[str, ImmutableValue]):
    """A read-only, hashable mapping in the D1.2 immutable value algebra."""

    _entries: tuple[tuple[str, ImmutableValue], ...]

    def __getitem__(self, key: str) -> ImmutableValue:
        for entry_key, value in self._entries:
            if entry_key == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return False
        return dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(self._entries)


def require_non_empty(value: str, field_name: str) -> None:
    """Reject an absent or whitespace-only opaque identifier or label."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def require_aware_utc(value: datetime, field_name: str) -> None:
    """Reject naive or non-UTC timestamps at the contract boundary."""

    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{field_name} must be aware UTC")


def freeze_mapping(value: object, field_name: str) -> FrozenMapping:
    """Convert a structured value to a deterministic, hashable mapping."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    frozen: list[tuple[str, ImmutableValue]] = []
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        frozen.append((key, freeze_value(item, field_name)))
    return FrozenMapping(tuple(sorted(frozen)))


def freeze_value(value: object, field_name: str) -> ImmutableValue:
    """Accept only explicitly immutable leaves and normalized containers."""

    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        return freeze_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_value(item, field_name) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(freeze_value(item, field_name) for item in value)
    raise ValueError(f"{field_name} contains unsupported value type {type(value).__name__}")


def freeze_refs(value: object, field_name: str) -> tuple[str, ...]:
    """Convert reference collections to immutable, non-empty opaque IDs."""

    if isinstance(value, str):
        raise ValueError(f"{field_name} must not be a string")
    try:
        refs: tuple[str, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{field_name} must be iterable") from error
    for ref in refs:
        require_non_empty(ref, f"{field_name} entries")
    return refs


def validate_sync_fields(
    sync: SyncFields,
    *,
    scope: Scope,
    origin_runtime_id: str,
    object_id: str,
    expected_version: int | None = None,
) -> None:
    """Ensure a SyncFields value describes the enclosing factual object."""

    if not isinstance(sync, SyncFields):
        raise ValueError("sync must be SyncFields")
    if sync.scope != scope:
        raise ValueError("sync.scope must match scope")
    if sync.origin_runtime_id != origin_runtime_id:
        raise ValueError("sync.origin_runtime_id must match origin_runtime_id")
    if sync.object_id != object_id:
        raise ValueError("sync.object_id must match object id")
    require_non_empty(sync.origin_runtime_id, "sync.origin_runtime_id")
    require_non_empty(sync.object_id, "sync.object_id")
    require_non_empty(sync.idempotency_key, "sync.idempotency_key")
    if isinstance(sync.version, bool) or sync.version < 1:
        raise ValueError("sync.version must be at least 1")
    if expected_version is not None and sync.version != expected_version:
        raise ValueError("sync.version must match version")
