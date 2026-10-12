"""True Canonical semantic batch Add and Search channel smoke test."""
from mr_mem.memory.core import MemoryCore
from mr_mem.memory.path_a import PathAAction, PathAProjector
from mr_mem.memory.path_a_runtime import (
    PathASearch,
    PathASemanticWriter,
    SemanticBlockInput,
    ThreadVerdict,
)
from mr_mem.memory.retrieval import (
    MemoryRetrievalQuery,
    MemoryRetrievalService,
    RetrievedMemoryCandidate,
)
from tests.test_native_semantics import KNOWN, SCOPE, Clock, Host, candidate


class Publisher:
    def __init__(self):
        self.received = []

    def publish(self, projection):
        self.received.append(projection)
        return "baseline-from-lce"


class CandidateProvider:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query):
        return tuple(RetrievedMemoryCandidate(mid, "D0") for mid in self.ids)


class BaselineReader:
    def __init__(self, scope):
        self.scope = scope
        self.queries = []

    def search(self, text, *, limit):
        self.queries.append((text, limit))
        return ()


def test_canonical_batch_add_then_search(tmp_path):
    host, publisher = Host(), Publisher()
    ref1 = host.put("t1", "RAW original first")
    ref2 = host.put("t2", "RAW original next")
    with MemoryCore(tmp_path / "memory.sqlite") as core:
        projector = PathAProjector(core, publisher=publisher)
        writer = PathASemanticWriter(
            core, projector, sources=host, clock=Clock(), origin_runtime_id="host",
        )
        receipt = writer.add_blocks(
            scope=SCOPE,
            blocks=(
                SemanticBlockInput(
                    candidate(ref1, "b1", "The decision is under review"),
                    ThreadVerdict("e1", "decision", PathAAction.TRACK,
                                  "Should we proceed?", "Pending decision"),
                ),
                SemanticBlockInput(
                    candidate(ref2, "b2", "New evidence completes the decision"),
                    ThreadVerdict("e2", "decision", PathAAction.MATURE,
                                  "Should we proceed?", "Proceed conditionally"),
                ),
            ),
            at=KNOWN,
        )
        assert len(receipt.committed) == 2
        assert tuple(r.status for r in receipt.thread_receipts) == ("active", "published")
        assert publisher.received[0].supporting_memory_ids == tuple(
            memory.memory_id for memory in receipt.committed
        )
        retrieval = MemoryRetrievalService(
            store=core.canonical,
            provider=CandidateProvider(tuple(memory.memory_id for memory in receipt.committed)),
        )
        baseline_reader = BaselineReader(SCOPE)
        search = PathASearch(retrieval, baseline_reader)
        result = search.search(MemoryRetrievalQuery(SCOPE, "decision", limit=5))
        assert len(result.canonical) == 2
        assert result.baselines == ()
        assert baseline_reader.queries == [("decision", 10)]
        projector.close()
