"""The one rejection ledger: derived cognition the user denied.

Ported from LCE ``cognition/rejection.py`` (origin/master 5228475). Design v6
3.3: when a derived understanding E (from support A + B) is denied, E is
withdrawn and this record keeps the same support from producing E again; new
evidence C changes the support signature and allows re-evaluation; history is
kept. MR-Mem is the compile authority, so this is the only ledger.

Audit changes against the LCE original:

1. An exact content match alone is a weak guard against a language model, which
   can phrase the same denied inference differently. The exact match stays as the
   last gate, and ``relevant`` returns the active records for a set of regions or
   sources so the compiler sees them next to KNOWN_LINES (design 4.3).
2. Times are explicit and identifiers deterministic, so replay is reproducible.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _utc(value: datetime, name: str) -> datetime:
    if value.tzinfo != UTC:
        raise ValueError(f"{name} must be an aware UTC datetime")
    return value


def content_hash(content: str) -> str:
    return hashlib.sha256(re.sub(r"\s+", " ", content).strip().encode("utf-8")).hexdigest()


def normalized_sources(source_refs: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(source_refs, tuple) or not source_refs:
        raise ValueError("source_refs must be a nonempty tuple")
    return tuple(sorted({_text(item, "source_ref") for item in source_refs}))


def support_signature(source_refs: tuple[str, ...]) -> str:
    payload = json.dumps(normalized_sources(source_refs), ensure_ascii=False,
                         separators=(",", ":"))
    return "support_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


@dataclass(frozen=True, slots=True)
class Rejection:
    rejection_id: str
    region_id: str
    content_hash: str
    support_signature: str
    source_refs: tuple[str, ...]
    rejected_content: str
    rejected_at: datetime
    authority_ref: str
    reason: str | None = None
    reopened_at: datetime | None = None
    reopening_authority_ref: str | None = None

    @property
    def active(self) -> bool:
        return self.reopened_at is None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS derived_rejections (
    rejection_id TEXT PRIMARY KEY,
    region_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    support_signature TEXT NOT NULL,
    source_refs_json TEXT NOT NULL,
    rejected_content TEXT NOT NULL,
    rejected_at TEXT NOT NULL,
    authority_ref TEXT NOT NULL,
    reason TEXT,
    reopened_at TEXT,
    reopening_authority_ref TEXT
);
CREATE INDEX IF NOT EXISTS idx_derived_rejection_lookup
    ON derived_rejections(region_id, content_hash, support_signature, reopened_at);
"""
_COLUMNS = ("rejection_id, region_id, content_hash, support_signature, source_refs_json, "
            "rejected_content, rejected_at, authority_ref, reason, reopened_at, "
            "reopening_authority_ref")


class RejectionStore:
    """Audit-preserving suppression of denied derived cognition."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), timeout=30.0, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @staticmethod
    def _row(row) -> Rejection:
        return Rejection(
            rejection_id=str(row[0]), region_id=str(row[1]), content_hash=str(row[2]),
            support_signature=str(row[3]), source_refs=tuple(json.loads(str(row[4]))),
            rejected_content=str(row[5]), rejected_at=datetime.fromisoformat(str(row[6])),
            authority_ref=str(row[7]), reason=None if row[8] is None else str(row[8]),
            reopened_at=None if row[9] is None else datetime.fromisoformat(str(row[9])),
            reopening_authority_ref=None if row[10] is None else str(row[10]),
        )

    def _select(self, where: str, args: tuple) -> tuple[Rejection, ...]:
        rows = self._conn.execute(
            f"SELECT {_COLUMNS} FROM derived_rejections {where}", args
        ).fetchall()
        return tuple(self._row(row) for row in rows)

    def active_match(
        self, *, region_id: str, content: str, source_refs: tuple[str, ...]
    ) -> Rejection | None:
        """Exact gate: the same region, the same content, the same support."""
        found = self._select(
            "WHERE region_id=? AND content_hash=? AND support_signature=? AND reopened_at IS NULL "
            "ORDER BY rejected_at DESC, rejection_id DESC LIMIT 1",
            (_text(region_id, "region_id"), content_hash(_text(content, "content")),
             support_signature(source_refs)),
        )
        return found[0] if found else None

    def relevant(
        self, *, region_ids: tuple[str, ...] = (), source_refs: tuple[str, ...] = ()
    ) -> tuple[Rejection, ...]:
        """Active records on these regions or sharing a source: compiler context."""
        regions, sources = set(region_ids), set(source_refs)
        return tuple(
            item for item in self._select(
                "WHERE reopened_at IS NULL ORDER BY rejected_at, rejection_id", ()
            )
            if item.region_id in regions or sources & set(item.source_refs)
        )

    def reject(
        self, *, region_id: str, content: str, source_refs: tuple[str, ...], authority_ref: str,
        rejected_at: datetime, reason: str | None = None,
    ) -> Rejection:
        region, proposal = _text(region_id, "region_id"), _text(content, "content")
        authority = _text(authority_ref, "authority_ref")
        reason = None if reason is None else _text(reason, "reason")
        at = _utc(rejected_at, "rejected_at")
        existing = self.active_match(region_id=region, content=proposal, source_refs=source_refs)
        if existing is not None:
            return existing
        sources = normalized_sources(source_refs)
        signature = support_signature(sources)
        digest = content_hash(proposal)
        rejection_id = "reject_" + hashlib.sha256(
            "|".join((region, digest, signature, at.isoformat())).encode("utf-8")
        ).hexdigest()[:32]
        with self._conn:
            self._conn.execute(
                f"INSERT OR IGNORE INTO derived_rejections({_COLUMNS}) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                (rejection_id, region, digest, signature, json.dumps(sources), proposal,
                 at.isoformat(), authority, reason),
            )
        return self.get(rejection_id)

    def get(self, rejection_id: str) -> Rejection:
        found = self._select("WHERE rejection_id=?", (_text(rejection_id, "rejection_id"),))
        if not found:
            raise KeyError(rejection_id)
        return found[0]

    def reopen(self, rejection_id: str, *, authority_ref: str, reopened_at: datetime) -> Rejection:
        """Lift the suppression explicitly; the record itself is kept."""
        current = self.get(rejection_id)
        if current.reopened_at is None:
            with self._conn:
                self._conn.execute(
                    "UPDATE derived_rejections SET reopened_at=?, reopening_authority_ref=? "
                    "WHERE rejection_id=?",
                    (_utc(reopened_at, "reopened_at").isoformat(),
                     _text(authority_ref, "authority_ref"), current.rejection_id),
                )
        return self.get(rejection_id)

    def list(self, *, region_id: str | None = None) -> tuple[Rejection, ...]:
        if region_id is None:
            return self._select("ORDER BY rejected_at, rejection_id", ())
        return self._select("WHERE region_id=? ORDER BY rejected_at, rejection_id",
                            (_text(region_id, "region_id"),))
