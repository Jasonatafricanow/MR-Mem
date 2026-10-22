"""Unified Path A read over canonical Memory, Threads and real LCE (no host adapter)."""

from datetime import timedelta

import pytest

from mr_mem import MemoryCore
from mr_mem.memory.providers.bm25 import BM25RetrievalProvider
from mr_mem.memory.retrieval import MemoryRetrievalService
from tests.test_lce_path_a import NOW, SCOPE, T0, Host, admit, service, track

pytest.importorskip("lce")

from mr_mem.integrations.lce_path_a import LcePathA  # noqa: E402
from mr_mem.integrations.path_a_read import PathAReader, ReadPolicy  # noqa: E402


class Env:
    """A canonical store, its LCE root and a host that can render source text."""

    def __init__(self, tmp_path):
        self.path = tmp_path / "m.sqlite"
        self.host = Host()
        self.core = MemoryCore(self.path)
        self.lce = LcePathA(canonical_path=self.path, lce_root=tmp_path / "lce", scope=SCOPE)

    def admit(self, record, minutes, content):
        return admit(self.core, self.host, record, minutes, content)

    def render(self, ref):
        if ref.source_key not in self.host.refs:
            return None
        return f"[Original | {ref.record_id}] raw text"

    def read(self, query, *, mode="path_a", limit=20, embed=None, **policy):
        retrieval = MemoryRetrievalService(
            store=self.core.canonical, provider=BM25RetrievalProvider(self.core.load_all()),
        )
        reader = PathAReader(
            canonical_path=self.path, retrieval=retrieval, lce=self.lce,
            policy=ReadPolicy(**policy), render_source=self.render, embed=embed,
        )
        return reader.search(SCOPE, query, limit=limit, mode=mode)

    def close(self):
        self.lce.close()
        self.core.close()


@pytest.fixture
def env(tmp_path):
    value = Env(tmp_path)
    yield value
    value.close()


def kinds(items):
    return [item.kind for item in items]


def test_path_a_reads_cognition_then_support_then_source_baseline_reads_memory_only(env):
    a = env.admit("r1", 0, "laptop battery dies after two hours")
    env.admit("r2", 5, "weekend hiking trip to the lake")
    env.lce.commit(question="Replace the laptop?", summary="Replace it because of the battery",
                   support=(a.memory_id,))
    items = env.read("laptop battery")
    assert kinds(items)[:3] == ["lce", "support", "source"]
    assert items[0].content.startswith("[Accepted understanding | revision 1 | as of ")
    assert items[0].content.endswith("Replace it because of the battery")
    assert items[1].item_id == f"memory:{a.memory_id}" and items[1].parent_id == items[0].item_id
    assert items[2].content == "[Original | r1] raw text"
    assert f"memory:{a.memory_id}" not in [i.item_id for i in items[3:]]  # no duplicate
    baseline = env.read("laptop battery", mode="baseline")
    assert set(kinds(baseline)) == {"memory"}
    assert baseline[0].item_id == f"memory:{a.memory_id}"


def test_projection_without_relevant_support_is_not_read(env):
    a = env.admit("r1", 0, "laptop battery dies after two hours")
    env.admit("r2", 5, "weekend hiking trip to the lake")
    env.lce.commit(question="Replace the laptop?", summary="Replace it", support=(a.memory_id,))
    assert "lce" not in kinds(env.read("hiking lake", support_rank_window=1))


def test_open_thread_is_read_and_hidden_once_its_lineage_head_is_newer(env):
    a = env.admit("r1", 0, "thinking about moving to Lisbon")
    svc = service(env.core, env.lce)
    svc.apply(scope=SCOPE, accepted_events=(
        track(a, "Move to Lisbon?", "Considering it"),), at=NOW)
    items = env.read("moving Lisbon")
    assert kinds(items)[:2] == ["thread", "support"]
    assert "Ongoing matter, not concluded" in items[0].content
    assert "Move to Lisbon? | So far: Considering it" in items[0].content
    # The same question is settled at first appearance elsewhere: the newer HEAD
    # supersedes the open Thread, so only the accepted understanding is served.
    b = env.admit("r2", 9, "decided to move to Lisbon in spring")
    env.lce.commit(question="Move to Lisbon?", summary="Moving in spring",
                   support=(b.memory_id,))
    later = env.read("moving Lisbon spring")
    assert "thread" not in kinds(later) and kinds(later)[0] == "lce"


def bag_of_words(text):
    vocabulary = ("penicillin", "allergy", "allergic", "drug", "laptop")
    words = text.lower().replace("?", " ").split()
    return [float(sum(w.startswith(v) for w in words)) for v in vocabulary]


def test_lost_evidence_cognition_is_kept_and_read_with_a_marker(env):
    a = env.admit("r1", 0, "allergic to penicillin")
    env.lce.commit(question="Drug allergies?", summary="Penicillin allergy",
                   support=(a.memory_id,))
    ref = a.provenance.source_refs[0]
    env.host.refs.pop(ref.source_key)  # the host lost the original text
    env.core.canonical.invalidate_source(SCOPE, ref, deleted=True)
    (kept,) = env.lce.accepted()
    assert kept.content == "Penicillin allergy" and not kept.evidence_complete
    # The support is no longer ACTIVE, so retrieval cannot rank it; only the
    # cognition's own similarity to the query can surface it.
    assert "lce" not in kinds(env.read("penicillin allergy"))
    items = env.read("penicillin allergy", embed=bag_of_words)
    assert kinds(items) == ["lce", "support"]  # no source text: it is gone
    assert all("| original evidence unavailable]" in item.content for item in items)


def test_source_text_for_hits_applies_to_both_modes(env):
    env.admit("r1", 0, "my sister lives in Osaka")
    for mode in ("baseline", "path_a"):
        items = env.read("sister Osaka", mode=mode, source_text_for_hits=True)
        assert kinds(items)[:2] == ["memory", "source"], mode


def test_limit_caps_output(env):
    for i in range(6):
        env.admit(f"r{i}", i, f"note number {i} about gardening")
    assert len(env.read("gardening", limit=3)) == 3
    with pytest.raises(ValueError):
        env.read("gardening", limit=0)


def test_revision_timeline_is_read_from_head(env):
    a = env.admit("r1", 0, "considering a new laptop with 16GB")
    b = env.admit("r2", 5, "not buying the laptop until after the trip")
    env.lce.commit(question="Replace laptop?", summary="Considering 16GB", support=(a.memory_id,))
    env.lce.revise(question="Replace laptop?", summary="Postponed until after the trip",
                   support=(b.memory_id,))
    head = env.read("laptop")[0]
    assert head.content.startswith("[Accepted understanding | revision 2 | as of 2026-08-01 00:05]")
    assert head.content.endswith("Postponed until after the trip")
    assert head.occurred_at == T0 + timedelta(minutes=5)


def test_raw_base_reaches_projections_upward_from_raw_hits(env):
    """Host raw evidence is the base list; a projection is read when a raw hit's
    record was compiled into one of its Blocks (raw -> Block -> LCE line)."""
    from mr_mem.integrations.path_a_read import RawHit

    a = env.admit("r1", 0, "The user decided not to buy the laptop: no budget")
    env.admit("r2", 5, "The user went hiking at the lake")
    env.lce.commit(question="Buy the laptop?", summary="Not buying it: no budget",
                   support=(a.memory_id,))
    raw = [
        RawHit("raw:1", "User: no money, so I am not buying that laptop", 0.9, T0, ("r1",)),
        RawHit("raw:2", "User: lake hike was great", 0.5, T0 + timedelta(minutes=5), ("r2",)),
    ]
    blocks = {"r1": (a.memory_id,)}

    def read(mode, **policy):
        reader = PathAReader(
            canonical_path=env.path, retrieval=None, lce=env.lce,
            policy=ReadPolicy(include_source_text=False, **policy),
            raw_search=lambda scope, query, limit: raw[:limit],
            blocks_for_record=lambda record_id: blocks.get(record_id, ()),
        )
        return reader.search(SCOPE, "did I buy the laptop", limit=10, mode=mode)

    items = read("path_a")
    assert kinds(items) == ["lce", "support", "raw", "raw"]
    assert [i.item_id for i in items[2:]] == ["raw:1", "raw:2"]  # base order kept
    baseline = read("baseline", stamp_raw_hits=True)
    assert kinds(baseline) == ["raw", "raw"]
    assert baseline[0].content.startswith("[") and "no money" in baseline[0].content


def test_matter_shape_reads_one_item_per_line_with_steps_in_order(env):
    """Governance design v5 5.2-5.3: a line is read as one matter, its steps
    inside in order, not as a header followed by loose support items."""
    from mr_mem.integrations.path_a_read import RawHit

    a = env.admit("r1", 0, "The user wants to buy the laptop")
    b = env.admit("r2", 5, "The user has no budget, so does not buy it")
    env.lce.commit(question="Buy the laptop?", summary="Wanted it, then decided not to buy",
                   support=(b.memory_id, a.memory_id))
    raw = [RawHit("raw:1", "User: no budget", 0.9, None, ("r2",))]

    def read(evidence_time):
        reader = PathAReader(
            canonical_path=env.path, retrieval=None, lce=env.lce,
            policy=ReadPolicy(include_source_text=False, projection_shape="matter",
                              stamp_raw_hits=True),
            raw_search=lambda scope, query, limit: raw[:limit],
            blocks_for_record=lambda record_id: {"r2": (b.memory_id,)}.get(record_id, ()),
            evidence_time=evidence_time,
        )
        return reader.search(SCOPE, "did I buy the laptop", limit=10, mode="path_a")

    items = read(None)
    assert kinds(items) == ["lce", "raw"]
    lines = items[0].content.split("\n")
    assert lines[0].startswith("[Accepted understanding | revision 1 | as of ")
    assert lines[1] == "Steps in order:"
    assert "wants to buy" in lines[2] and "no budget" in lines[3]  # time order
    assert items[1].content == "User: no budget"  # no source time: no stamp

    untimed = read(lambda memory: None)  # the source gave no message times
    head, *steps = untimed[0].content.split("\n")
    assert head == "[Accepted understanding | revision 1] Wanted it, then decided not to buy"
    assert steps[1] == "1. The user wants to buy the laptop"
    assert untimed[0].occurred_at is None
