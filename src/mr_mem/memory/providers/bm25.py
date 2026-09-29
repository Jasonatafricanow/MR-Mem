"""Dependency-free lexical BM25 discovery over canonical Memory snapshots.

The snapshot is derived read state. It owns no Memory authority and can be
discarded/rebuilt from canonical Memory at any time.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from mr_mem.memory.contracts import CommittedMemory, MemoryLifecycle
from mr_mem.memory.retrieval import MemoryRetrievalQuery, RetrievedMemoryCandidate
from mr_mem.memory.store import scope_json


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def lexical_tokens(text: str) -> tuple[str, ...]:
    """Tokenize Latin/digits as words and CJK runs as full terms plus bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    word: list[str] = []
    cjk: list[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    def flush_cjk() -> None:
        if not cjk:
            return
        run = "".join(cjk)
        if len(run) == 1:
            tokens.append(run)
        else:
            tokens.append(run)
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
        cjk.clear()

    for char in normalized:
        if _is_cjk(char):
            flush_word()
            cjk.append(char)
        elif char.isalnum():
            flush_cjk()
            word.append(char)
        else:
            flush_word()
            flush_cjk()
    flush_word()
    flush_cjk()
    return tuple(tokens)


@dataclass(frozen=True, slots=True)
class _Document:
    memory_id: str
    scope_key: str
    length: int
    frequencies: Counter[str]


class BM25RetrievalProvider:
    """Immutable lexical snapshot; rebuild after canonical Memory changes."""

    def __init__(
        self,
        memories: Iterable[CommittedMemory],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be finite and positive")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be finite and in [0, 1]")
        self._k1, self._b = k1, b
        self._documents: dict[str, _Document] = {}
        self._scope_ids: dict[str, tuple[str, ...]] = {}
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        scope_ids: dict[str, list[str]] = defaultdict(list)

        for memory in memories:
            if not isinstance(memory, CommittedMemory):
                raise ValueError("BM25 snapshot requires canonical CommittedMemory values")
            if memory.lifecycle is not MemoryLifecycle.ACTIVE:
                continue
            if memory.memory_id in self._documents:
                raise ValueError(f"duplicate Memory identity: {memory.memory_id}")
            frequencies = Counter(lexical_tokens(memory.content))
            length = sum(frequencies.values())
            if not length:
                continue
            scope_key = scope_json(memory.scope)
            doc = _Document(memory.memory_id, scope_key, length, frequencies)
            self._documents[memory.memory_id] = doc
            scope_ids[scope_key].append(memory.memory_id)
            for term, frequency in frequencies.items():
                postings[term][memory.memory_id] = frequency

        self._scope_ids = {key: tuple(ids) for key, ids in scope_ids.items()}
        self._postings = {term: dict(values) for term, values in postings.items()}

    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        if query.limit == 0:
            return ()
        terms = tuple(dict.fromkeys(lexical_tokens(query.text)))
        if not terms:
            return ()
        scope_key = scope_json(query.scope)
        scoped_ids = self._scope_ids.get(scope_key, ())
        if not scoped_ids:
            return ()
        scoped = set(scoped_ids)
        document_count = len(scoped_ids)
        average_length = sum(self._documents[mid].length for mid in scoped_ids) / document_count
        scores: dict[str, float] = defaultdict(float)

        for term in terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            matches = [(mid, tf) for mid, tf in posting.items() if mid in scoped]
            if not matches:
                continue
            document_frequency = len(matches)
            idf = math.log(
                1.0 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for memory_id, tf in matches:
                length = self._documents[memory_id].length
                denominator = tf + self._k1 * (
                    1.0 - self._b + self._b * length / average_length
                )
                scores[memory_id] += idf * (tf * (self._k1 + 1.0)) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return tuple(
            RetrievedMemoryCandidate(
                memory_id=memory_id,
                provider="bm25",
                provider_ref=f"bm25:{memory_id}",
                score=score,
            )
            for memory_id, score in ranked[: query.limit]
        )
