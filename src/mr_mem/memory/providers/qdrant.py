"""Explicit vectors only: Qdrant owns derived points, never canonical Memory."""

import json
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from mr_mem.memory.contracts import CommittedMemory
from mr_mem.memory.embedding import (
    EmbeddingProvider,
    projection_target,
    validate_vector,
)
from mr_mem.memory.projection import ProjectionIntent, ProjectionQueue
from mr_mem.memory.providers.errors import (
    ProviderPackageMissing,
    ProviderStorageUnavailable,
)
from mr_mem.memory.retrieval import (
    MemoryRetrievalQuery,
    RetrievalProviderUnavailable,
    RetrievedMemoryCandidate,
)
from mr_mem.memory.store import CanonicalMemoryStore, scope_json


class QdrantIndex:
    """One explicitly opened client, shared by separate read/write capabilities."""

    collection = "memory"

    def __init__(
        self,
        client: Any,
        models: Any,
        embedding: EmbeddingProvider,
        namespace: str,
        canonical_path: Path,
    ):
        self._client, self._models, self._embedding = client, models, embedding
        self.identity = embedding.identity
        self.target = projection_target(self.identity)
        self._namespace = namespace
        self._canonical_path = canonical_path.resolve()
        self._metadata = {
            "mr_namespace": namespace,
            "mr_target": self.target,
            "mr_embedding": asdict(self.identity),
            "mr_schema": "qdrant-cosine-v1",
        }
        # one scope's points (ids, memory ids, unit vectors), the most recently searched scopes:
        # Qdrant's local mode scores every point of every scope and filters the payloads in
        # Python, so a query took ~2 s at 150k points (AML Full, 2026-11-01) and a gateway cut
        # the queued Searches at 20 s. The same exact cosine search, over this scope alone
        self._scopes: OrderedDict[str, tuple[list[str], list[str], Any]] = OrderedDict()
        self._scopes_lock = threading.Lock()
        self._scopes_kept = 32

    def _scope_filter(self, scope: str) -> Any:
        return self._models.Filter(
            must=[
                self._models.FieldCondition(key=key, match=self._models.MatchValue(value=value))
                for key, value in {
                    "mr_namespace": self._namespace,
                    "mr_target": self.target,
                    "mr_scope": scope,
                }.items()
            ]
        )

    def _scope_points(self, scope: str) -> tuple[list[str], list[str], Any]:
        """The scope's points, read from Qdrant once and kept up to date by the writer."""
        import numpy as np

        with self._scopes_lock:
            if scope in self._scopes:
                self._scopes.move_to_end(scope)
                return self._scopes[scope]
        ids, memories, rows, offset = [], [], [], None
        while True:
            points, offset = self._client.scroll(
                self.collection,
                scroll_filter=self._scope_filter(scope),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for point in points:
                payload = point.payload if isinstance(point.payload, dict) else {}
                memory_id = payload.get("mr_memory_id")
                if (
                    isinstance(memory_id, str)
                    and memory_id.strip()
                    and all(payload.get(k) == v for k, v in self._metadata.items())
                    and payload.get("mr_scope") == scope
                    and str(point.id) == self.point_id(memory_id)
                ):
                    ids.append(str(point.id))
                    memories.append(memory_id)
                    rows.append(point.vector)
            if offset is None:
                break
        matrix = np.asarray(rows, dtype=np.float32).reshape(len(rows), self.identity.dimension)
        matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
        with self._scopes_lock:
            self._scopes[scope] = (ids, memories, matrix)
            while len(self._scopes) > self._scopes_kept:
                self._scopes.popitem(last=False)
        return ids, memories, matrix

    def _scope_upserted(self, scope: str, point_id: str, memory_id: str, vector) -> None:
        import numpy as np

        with self._scopes_lock:
            if scope not in self._scopes:
                return  # read whole from Qdrant when it is first searched
            ids, memories, matrix = self._scopes[scope]
            row = np.asarray(vector, dtype=np.float32)
            row /= max(float(np.linalg.norm(row)), 1e-12)
            if point_id in ids:
                matrix[ids.index(point_id)] = row
            else:
                self._scopes[scope] = (
                    [*ids, point_id],
                    [*memories, memory_id],
                    np.vstack([matrix, row[None, :]]),
                )

    def _check(self) -> None:
        if self._embedding.identity != self.identity:
            raise ProviderStorageUnavailable("embedding identity changed on an open index")
        try:
            config = self._client.get_collection(self.collection).config
            if (
                config.metadata != self._metadata
                or config.params.vectors.size != self.identity.dimension
                or config.params.vectors.distance != self._models.Distance.COSINE
            ):
                raise ProviderStorageUnavailable("incompatible semantic index identity/geometry")
        except ProviderStorageUnavailable:
            raise
        except Exception as exc:
            raise ProviderStorageUnavailable("semantic index missing or unavailable") from exc

    def _create(self) -> None:
        self._client.create_collection(
            self.collection,
            vectors_config=self._models.VectorParams(
                size=self.identity.dimension,
                distance=self._models.Distance.COSINE,
            ),
            metadata=self._metadata,
        )

    def point_id(self, memory_id: str) -> str:
        """Derived upsert key; canonical identity is always retained separately."""
        key = json.dumps(["mr-point-v1", self._namespace, self.target, memory_id])
        return str(uuid5(NAMESPACE_URL, key))

    def writer(self) -> "QdrantProjectionWriter":
        return QdrantProjectionWriter(self)

    def retrieval(self) -> "QdrantRetrievalProvider":
        return QdrantRetrievalProvider(self)

    def close(self) -> None:
        self._client.close()

    def wipe(self) -> None:
        """Explicitly remove this derived collection only."""
        if self._client.collection_exists(self.collection):
            self._check()
            self._client.delete(
                self.collection,
                points_selector=self._models.FilterSelector(
                    filter=self._models.Filter(),
                ),
                wait=True,
            )
            if self._client.count(self.collection, exact=True).count != 0:
                raise ProviderStorageUnavailable("semantic index wipe incomplete")
            self._client.delete_collection(self.collection)

    def rebuild(self, queue: ProjectionQueue) -> int:
        """Explicit destructive index rebuild; canonical rows are only read."""
        self.writer().validate_queue(queue)
        count = queue.rebuild(target=self.target, reset=True)
        self.wipe()
        self._create()
        return count


class QdrantProjectionWriter:
    def __init__(self, index: QdrantIndex):
        self._index = index

    def validate_queue(self, queue: ProjectionQueue) -> None:
        if queue.database_path != self._index._canonical_path:
            raise ValueError("projection queue belongs to a different Runtime database")

    def upsert(self, memory: CommittedMemory, *, intent: ProjectionIntent) -> str:
        index = self._index
        if (
            intent.target != index.target
            or intent.memory_id != memory.memory_id
            or intent.operation != "upsert"
        ):
            raise ValueError("projection intent does not match configured target/Memory")
        canonical = CanonicalMemoryStore(index._canonical_path, read_only=True)
        try:
            if canonical.get(memory.memory_id) != memory:
                raise ValueError("Memory is not canonical in the configured Runtime")
        finally:
            canonical.close()
        if not index._client.collection_exists(index.collection):
            index._create()
        index._check()
        vector = validate_vector(index._embedding.embed(memory.content), index.identity)
        point_id = index.point_id(memory.memory_id)
        scope = scope_json(memory.scope)
        payload = index._metadata | {
            "mr_memory_id": memory.memory_id,
            "mr_scope": scope,
        }
        try:
            index._client.upsert(
                index.collection,
                points=[
                    index._models.PointStruct(
                        id=point_id,
                        vector=list(vector),
                        payload=payload,
                    )
                ],
                wait=True,
            )
        except Exception as exc:
            raise RetrievalProviderUnavailable("Qdrant projection failed") from exc
        index._scope_upserted(scope, point_id, memory.memory_id, vector)
        return point_id


class QdrantRetrievalProvider:
    """Read capability: no admission, persistence, wipe or rebuild methods."""

    def __init__(self, index: QdrantIndex):
        self._index = index

    def search(self, query: MemoryRetrievalQuery) -> tuple[RetrievedMemoryCandidate, ...]:
        if query.limit == 0:
            return ()
        index = self._index
        index._check()
        try:
            vector = validate_vector(index._embedding.embed(query.text), index.identity)
        except Exception as exc:
            raise RetrievalProviderUnavailable("Qdrant semantic query failed") from exc
        try:
            return self._scoped(query, vector)
        except Exception:  # noqa: BLE001 - Qdrant's own query below gives the same ranking
            pass
        try:
            filters = {
                "mr_namespace": index._namespace,
                "mr_target": index.target,
                "mr_scope": scope_json(query.scope),
            }
            raw = index._client.query_points(
                index.collection,
                query=list(vector),
                limit=query.limit,
                with_vectors=False,
                query_filter=index._models.Filter(
                    must=[
                        index._models.FieldCondition(
                            key=key,
                            match=index._models.MatchValue(value=value),
                        )
                        for key, value in filters.items()
                    ]
                ),
            ).points
        except Exception as exc:
            raise RetrievalProviderUnavailable("Qdrant semantic query failed") from exc
        result = []
        for hit in raw[: query.limit]:
            payload = hit.payload
            if not isinstance(payload, dict):
                continue
            memory_id = payload.get("mr_memory_id")
            if (
                not isinstance(memory_id, str)
                or not memory_id.strip()
                or any(payload.get(k) != v for k, v in index._metadata.items())
                or payload.get("mr_scope") != scope_json(query.scope)
                or str(hit.id) != index.point_id(memory_id)
            ):
                continue
            try:
                result.append(
                    RetrievedMemoryCandidate(
                        memory_id,
                        "qdrant",
                        provider_ref=str(hit.id),
                        score=float(hit.score),
                    )
                )
            except (ValueError, TypeError):
                continue
        return tuple(result)

    def _scoped(self, query: MemoryRetrievalQuery, vector) -> tuple[RetrievedMemoryCandidate, ...]:
        import numpy as np

        ids, memories, matrix = self._index._scope_points(scope_json(query.scope))
        if not ids:
            return ()
        q = np.asarray(vector, dtype=np.float32)
        q /= max(float(np.linalg.norm(q)), 1e-12)
        scores = matrix @ q
        k = min(query.limit, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return tuple(
            RetrievedMemoryCandidate(
                memories[i], "qdrant", provider_ref=ids[i], score=float(scores[i])
            )
            for i in top
        )


def open_qdrant_index(
    *,
    index_root: Path | str,
    canonical_path: Path | str,
    namespace: str,
    embedding: EmbeddingProvider,
    create: bool = False,
) -> QdrantIndex:
    """Open a standalone derived Qdrant index for one MR-Mem namespace."""
    if type(create) is not bool:
        raise ValueError("create must be bool")
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be nonempty")
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise ProviderPackageMissing("install mr-mem[vector] for Qdrant") from exc

    canonical = Path(canonical_path).resolve()
    root = Path(index_root).resolve()
    path = root / "qdrant" / projection_target(embedding.identity)
    if not create and not (path / "meta.json").is_file():
        raise ProviderStorageUnavailable("configured Qdrant index is absent")

    client = None
    try:
        client = QdrantClient(path=str(path))
        index = QdrantIndex(client, models, embedding, namespace, canonical)
        if create and not client.collection_exists(index.collection):
            index._create()
        index._check()
        return index
    except Exception as exc:
        if client is not None:
            client.close()
        if isinstance(exc, RetrievalProviderUnavailable):
            raise
        raise ProviderStorageUnavailable("cannot open configured Qdrant index") from exc
