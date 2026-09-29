from dataclasses import replace

from mr_mem.memory.providers.bm25 import BM25RetrievalProvider, lexical_tokens
from mr_mem.memory.providers.hybrid import (
    HybridRRFProvider,
    HyDEFallbackProvider,
    RetrievalArm,
)
from mr_mem.memory.retrieval import MemoryRetrievalQuery, RetrievedMemoryCandidate
from tests.conftest import memory


class Provider:
    def __init__(self, responses):
        self.responses = responses
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return self.responses.get(query.text, self.responses.get("*", ()))


class Expander:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def expand(self, query):
        self.calls.append(query)
        return self.value


def hit(memory_id):
    return RetrievedMemoryCandidate(memory_id, "fixture")


def test_bm25_keeps_existing_cjk_and_exact_term_behavior():
    assert lexical_tokens("MacBook M4 / ORA-00942") == (
        "macbook", "m4", "ora", "00942"
    )
    assert lexical_tokens("电脑越来越卡") == (
        "电脑越来越卡", "电脑", "脑越", "越来", "来越", "越卡"
    )
    base = memory("m1", content="MacBook M4 price is 9999")
    unrelated = replace(
        memory("m2", content="lunch"),
        sync=replace(memory("m2").sync, idempotency_key="m2"),
    )
    provider = BM25RetrievalProvider((base, unrelated))
    hits = provider.search(MemoryRetrievalQuery(base.scope, "M4 price", limit=5))
    assert [item.memory_id for item in hits] == ["m1"]


def test_rrf_fuses_rank_without_comparing_raw_scores():
    first = Provider({"*": (
        RetrievedMemoryCandidate("m1", "a", score=1000),
        RetrievedMemoryCandidate("m2", "a", score=1),
    )})
    second = Provider({"*": (
        RetrievedMemoryCandidate("m2", "b", score=0.01),
        RetrievedMemoryCandidate("m3", "b", score=0.99),
    )})
    provider = HybridRRFProvider(
        (RetrievalArm("lexical", first), RetrievalArm("dense", second))
    )
    result = provider.search(
        MemoryRetrievalQuery(memory().scope, "computer", limit=2)
    )
    assert [item.memory_id for item in result] == ["m2", "m1"]


def test_hyde_is_only_used_as_sparse_recall_fallback():
    enough = Provider({"q": (hit("m1"), hit("m2"), hit("m3"))})
    expander = Expander("expanded")
    provider = HyDEFallbackProvider(enough, expander, min_results=3)
    result = provider.search(MemoryRetrievalQuery(memory().scope, "q", limit=5))
    assert [item.memory_id for item in result] == ["m1", "m2", "m3"]
    assert expander.calls == []

    sparse = Provider({
        "q": (hit("m1"),),
        "expanded": (hit("m2"), hit("m1"), hit("m3")),
    })
    expander = Expander("expanded")
    provider = HyDEFallbackProvider(sparse, expander, min_results=3)
    result = provider.search(MemoryRetrievalQuery(memory().scope, "q", limit=3))
    assert [item.memory_id for item in result] == ["m1", "m2", "m3"]
    assert expander.calls == ["q"]
