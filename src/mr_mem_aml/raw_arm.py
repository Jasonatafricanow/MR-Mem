"""Raw-message retrieval: the P1 baseline inside the Path A runtime.

Every Add message becomes one searchable unit ("User: ..." / "Assistant: ...",
split at 16,000 UTF-8 bytes, de-duplicated within the request), indexed by BM25
(k1=1.5, b=0.75) and a dense embedding, fused by MR-Mem's core
``HybridRRFProvider`` with a 200-candidate pool per arm and rrf_k=60. This is the
protocol of the P1 baseline run (mr-mem-aml-adapter 07914e7), so Path A reads on
top of exactly the evidence list P1 returns and the difference between the two
is the projections alone.

Units keep the record id of their message: that is the upward link from a raw
hit to the SemanticBlocks compiled from the same message (and from there to
Threads and LCE lines).
"""

from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from mr_mem.memory.providers.bm25 import lexical_tokens
from mr_mem.memory.providers.hybrid import HybridRRFProvider, RetrievalArm
from mr_mem.memory.retrieval import MemoryRetrievalQuery, RetrievedMemoryCandidate

MAX_CONTENT_BYTES = 16_000
DENSE_MAX_CHARS = 32_000


def split_content(text: str, *, max_bytes: int = MAX_CONTENT_BYTES) -> list[str]:
    """Same split as the P1 adapter (mr_mem_aml.chunking.split_content there)."""
    if not text.strip():
        return []
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        while end > start and (encoded[end - 1] & 0xC0) == 0x80:
            end -= 1
        if end <= start:
            end = min(start + max_bytes, len(encoded))
        piece = encoded[start:end].decode("utf-8", errors="ignore").strip()
        if piece:
            chunks.append(piece)
        start = end
    return chunks


@dataclass(frozen=True, slots=True)
class RawUnit:
    unit_id: str
    user_id: str
    text: str
    record_id: str
    occurred_at: datetime


def units_for(request_id: str, messages) -> list[RawUnit]:
    """P1 unit construction for one Add request (messages in request order)."""
    out: list[RawUnit] = []
    seen: set[str] = set()
    for message in messages:
        label = "User" if message.role == "user" else "Assistant"
        for piece in split_content(f"{label}: {message.content}"):
            if piece in seen:
                continue
            seen.add(piece)
            out.append(RawUnit(f"raw:{request_id}:{len(out)}", message.user_id, piece,
                               message.record_id, message.occurred_at))
    return out


class _UserIndex:
    def __init__(self) -> None:
        self.units: dict[str, RawUnit] = {}
        self.lengths: dict[str, int] = {}
        self.postings: dict[str, dict[str, int]] = defaultdict(dict)
        self.ids: list[str] = []
        self.vectors: np.ndarray | None = None


class RawArm:
    def __init__(self, embedding=None, *, rrf_k: int = 60, pool: int = 200,
                 k1: float = 1.5, b: float = 0.75) -> None:
        self.embedding = embedding
        self.pool = pool
        self.k1, self.b = k1, b
        self._lock = threading.RLock()
        self._users: dict[str, _UserIndex] = defaultdict(_UserIndex)
        arms = [RetrievalArm("bm25", _Arm(self, self._bm25))]
        if embedding is not None:
            arms.append(RetrievalArm("dense", _Arm(self, self._dense)))
        self._fused = HybridRRFProvider(tuple(arms), rrf_k=rrf_k) if len(arms) > 1 else None

    # -- write ------------------------------------------------------------

    def add(self, units: list[RawUnit]) -> None:
        fresh = []
        with self._lock:
            for unit in units:
                index = self._users[unit.user_id]
                if unit.unit_id in index.units:
                    continue
                freqs = Counter(lexical_tokens(unit.text))
                index.units[unit.unit_id] = unit
                length = sum(freqs.values())
                if length:
                    index.lengths[unit.unit_id] = length
                    for term, tf in freqs.items():
                        index.postings[term][unit.unit_id] = tf
                fresh.append(unit)
        if self.embedding is None or not fresh:
            return
        texts = [u.text[:DENSE_MAX_CHARS] for u in fresh]
        if hasattr(self.embedding, "embed_many"):
            matrix = np.asarray(self.embedding.embed_many(texts), dtype=np.float32)
        else:
            matrix = np.asarray([self.embedding.embed(t) for t in texts], dtype=np.float32)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        with self._lock:
            for unit, row in zip(fresh, matrix, strict=True):
                index = self._users[unit.user_id]
                index.ids.append(unit.unit_id)
                index.vectors = (row[None, :] if index.vectors is None
                                 else np.vstack([index.vectors, row]))

    # -- read -------------------------------------------------------------

    def unit(self, user_id: str, unit_id: str) -> RawUnit | None:
        return self._users[user_id].units.get(unit_id)

    def _bm25(self, user_id: str, text: str, limit: int) -> list[tuple[str, float]]:
        index = self._users[user_id]
        terms = tuple(dict.fromkeys(lexical_tokens(text)))
        if not terms or not index.lengths:
            return []
        n = len(index.lengths)
        avg = sum(index.lengths.values()) / n
        scores: dict[str, float] = defaultdict(float)
        for term in terms:
            posting = index.postings.get(term)
            if not posting:
                continue
            idf = math.log(1.0 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for unit_id, tf in posting.items():
                denom = tf + self.k1 * (1.0 - self.b + self.b * index.lengths[unit_id] / avg)
                scores[unit_id] += idf * (tf * (self.k1 + 1.0)) / denom
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]

    def _dense(self, user_id: str, text: str, limit: int) -> list[tuple[str, float]]:
        index = self._users[user_id]
        if index.vectors is None or not index.ids:
            return []
        q = np.asarray(self.embedding.embed(text[:DENSE_MAX_CHARS]), dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        scores = index.vectors @ q
        order = np.argsort(-scores, kind="stable")[:limit]
        return [(index.ids[int(i)], float(scores[int(i)])) for i in order]

    def search(self, user_id: str, text: str, limit: int) -> list[tuple[RawUnit, float]]:
        with self._lock:
            index = self._users[user_id]
            if self._fused is None:
                hits = self._bm25(user_id, text, limit)
            else:
                fused = self._fused.search(_query(user_id, text, limit))
                hits = [(c.memory_id, float(c.score or 0.0)) for c in fused]
            return [(index.units[uid], score) for uid, score in hits[:limit]]


def _query(user_id: str, text: str, limit: int) -> MemoryRetrievalQuery:
    from mr_mem_aml.source_store import user_scope

    return MemoryRetrievalQuery(scope=user_scope(user_id), text=text, limit=limit)


class _Arm:
    """One retrieval arm with the P1 fixed candidate pool (max(limit, pool))."""

    def __init__(self, owner: RawArm, search) -> None:
        self._owner = owner
        self._search = search

    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        if query.limit == 0:
            return ()
        user_id = query.scope.user_id or ""
        hits = self._search(user_id, query.text, max(query.limit, self._owner.pool))
        return tuple(
            RetrievedMemoryCandidate(memory_id=uid, provider="raw", provider_ref=f"raw:{uid}",
                                     score=score)
            for uid, score in hits
        )
