"""The SB search over one scope's cached points ranks as Qdrant's own filtered query does."""

import numpy as np
import pytest

pytest.importorskip("qdrant_client")


def test_scoped_search_matches_qdrant_and_follows_upserts():
    from qdrant_client import QdrantClient, models

    from mr_mem.memory.embedding import EmbeddingIdentity
    from mr_mem.memory.providers.qdrant import QdrantIndex
    from mr_mem.memory.retrieval import MemoryRetrievalQuery
    from mr_mem.memory.store import scope_json
    from mr_mem_aml.source_store import user_scope

    rng = np.random.default_rng(0)
    vectors = {}

    class Fixed:
        identity = EmbeddingIdentity("test", "m", 8, "r1")

        def embed(self, text):
            return vectors.setdefault(text, tuple(float(v) for v in rng.standard_normal(8)))

    index = QdrantIndex(QdrantClient(":memory:"), models, Fixed(), "ns", __import__("pathlib").Path("."))
    index._create()

    def put(user, memory_id):
        index._client.upsert(
            index.collection,
            points=[
                models.PointStruct(
                    id=index.point_id(memory_id),
                    vector=list(index._embedding.embed(memory_id)),
                    payload=index._metadata
                    | {"mr_memory_id": memory_id, "mr_scope": scope_json(user_scope(user))},
                )
            ],
            wait=True,
        )
        index._scope_upserted(scope_json(user_scope(user)), index.point_id(memory_id))

    for n in range(60):
        put("a" if n % 3 else "b", f"m{n}")
    search = index.retrieval()
    plain = type(search)._scoped
    query = MemoryRetrievalQuery(scope=user_scope("a"), text="question", limit=10)

    def qdrant_only():
        type(search)._scoped = lambda self, q, v: (_ for _ in ()).throw(RuntimeError())
        try:
            return [c.memory_id for c in search.search(query)]
        finally:
            type(search)._scoped = plain

    assert [c.memory_id for c in search.search(query)] == qdrant_only()
    put("a", "late")  # written after the scope was cached
    vectors["question2"] = vectors["late"]
    query = MemoryRetrievalQuery(scope=user_scope("a"), text="question2", limit=10)
    got = [c.memory_id for c in search.search(query)]
    assert got[0] == "late" and got == qdrant_only()
    assert all(m in {f"m{n}" for n in range(60) if n % 3} | {"late"} for m in got)
