"""LCE Thread + Path A tree driven by compiler signals (design v6 4.4-4.5)."""

from datetime import UTC, datetime, timedelta

import pytest

from mr_mem import MemoryCore, SemanticAdmissionService, SemanticMemoryCandidate, SourceRef
from mr_mem.lce.path_a import Attachment, BlockLines, PathATree
from mr_mem.memory.product import ThreadStatus
from tests.test_native_semantics import SCOPE

T0 = datetime(2026, 5, 1, 8, tzinfo=UTC)
TRIP = "Trip to Portugal"
DINNER = "Dinner with Ana in Lisbon"


class Host:
    def __init__(self):
        self.refs = {}

    def current_ref(self, scope, ref):
        return self.refs.get(ref.source_key) if scope == SCOPE else None


class Clock:
    def now(self):
        return datetime(2026, 10, 1, tzinfo=UTC)


class Env:
    def __init__(self, tmp_path):
        self.core = MemoryCore(tmp_path / "m.sqlite")
        self.host = Host()
        self.tree = PathATree(tmp_path / "lce", SCOPE, canonical=self.core.canonical,
                              products=self.core.products, fingerprint="test")
        self.hour = 0

    def block(self, content, hours):
        ref = SourceRef("host", "s1", f"r{hours}-{len(self.host.refs)}",
                        T0 + timedelta(hours=hours), "1")
        self.host.refs[ref.source_key] = ref
        admission = SemanticAdmissionService(store=self.core.canonical, sources=self.host,
                                             clock=Clock(), origin_runtime_id="t")
        return admission.admit(SemanticMemoryCandidate(
            semantic_id=ref.record_id, scope=SCOPE, content=content, source_refs=(ref,),
            compiler_version="test",
        ))

    def step(self, hours, *items, refs=None):
        """items: (content, [Attachment, ...]) compiled in one batch at ``hours``."""
        blocks = [BlockLines(self.block(c, hours).memory_id, tuple(lines)) for c, lines in items]
        return self.tree.apply(blocks, refs or {}, T0 + timedelta(hours=hours))

    def thread(self, question):
        threads = self.core.products.list_threads(SCOPE)
        return next(t for t in threads if t.open_question == question)

    def close(self):
        self.tree.close()
        self.core.close()


@pytest.fixture
def env(tmp_path):
    value = Env(tmp_path)
    yield value
    value.close()


def A(thread, after=None, committed=True, summary=None, done=False):  # noqa: N802
    return Attachment(thread, after, committed, summary or thread, done)


def test_portugal_trip_grows_a_tree_with_parallel_branches(env):
    # an intention: nothing has happened yet, so no tree, only a Thread
    env.step(0, ("The user wants to go to Portugal.", [A(TRIP, committed=False)]))
    trip = env.thread(TRIP)
    assert trip.anchor_node_id is None and env.tree.lines.list_lines() == ()
    # booked: the first committed step roots a Line; the Thread hangs from it
    env.step(1, ("The user booked a flight to Lisbon.", [A(TRIP)]))
    booked = env.thread(TRIP).anchor_node_id
    assert booked is not None and len(env.tree.lines.list_lines()) == 1
    # traffic happened (committed), missing the flight is still open (uncommitted)
    env.step(5, ("The user is stuck in traffic on the way to the airport.", [A(TRIP)]),
             ("The user may miss the flight.", [A(TRIP, after="b1", committed=False)]))
    traffic = env.thread(TRIP).anchor_node_id
    assert env.tree.lines.parents_at(traffic, T0 + timedelta(hours=6)) == (booked,)
    assert len(env.thread(TRIP).current_support_ids) == 1  # the open step waits on the Thread
    env.step(6, ("The user made the flight; it took off.", [A(TRIP)]))
    env.step(9, ("The user arrived in Lisbon.", [A(TRIP)]))
    arrived = env.thread(TRIP).anchor_node_id
    # "agreed to have dinner tomorrow": the agreement happened; the dinner has not
    known = env.tree.known(batch_text="dinner with Ana", now=T0 + timedelta(hours=10),
                           threads_policy={"recent": 5, "cued": 5, "due": 5},
                           steps_policy={"recent": 5, "cued": 5})
    n_arrived = next(ref for ref, node in known.refs.items() if node == arrived)
    env.step(10, ("Ana and the user agreed to have dinner tomorrow.",
                  [A(DINNER, after=n_arrived)]),
             ("The user and Ana will have dinner tomorrow evening.",
              [A(DINNER, after="b1", committed=False)]),
             ("The user went shopping on Rua Augusta.", [A("Shopping in Lisbon", after=n_arrived)]),
             refs=known.refs)
    agreed = env.thread(DINNER).anchor_node_id
    shopping = env.thread("Shopping in Lisbon").anchor_node_id
    now = T0 + timedelta(hours=12)
    # dinner and shopping are parallel branches under "arrived", unrelated to each other
    assert set(env.tree.lines.children_at(arrived, now)) == {agreed, shopping}
    assert env.tree.view.ancestors(agreed, cutoff=now)[0] == arrived  # background, by index
    env.step(30, ("The user and Ana had dinner; the food was great.",
                  [A(DINNER, done=True)]))
    dinner = env.core.products.get_thread(env.thread(TRIP).thread_id)
    assert dinner.status is ThreadStatus.OPEN  # the trip goes on
    assert not [t for t in env.core.products.list_threads(SCOPE, status=ThreadStatus.OPEN)
                if t.open_question == DINNER]
    frontier = set(env.tree.view.frontier(env.tree.lines.list_lines()[0].line_id,
                                          cutoff=T0 + timedelta(hours=31)))
    assert shopping in frontier and len(frontier) == 2


def test_one_block_meets_two_trees_of_different_kinds(env):
    env.step(0, ("The user arrived in Lisbon.", [A(TRIP)]))
    env.step(1, ("The user likes trying local food.", [A("Food tastes")]))
    known = env.tree.known(batch_text="bacalhau Lisbon local food", now=T0 + timedelta(hours=2),
                           threads_policy={"recent": 5, "cued": 5, "due": 5},
                           steps_policy={"recent": 5, "cued": 5})
    result = env.step(2, ("The user ate bacalhau at a local tasca in Lisbon.",
                          [A(TRIP, after="T1" if known.refs["T1"] == env.thread(TRIP).thread_id
                             else "T2"),
                           A("Food tastes")]), refs=known.refs)
    assert result.committed_nodes == 2 and not result.failures
    lines = env.tree.lines.list_lines()
    assert len(lines) == 2
    meal = env.thread(TRIP).anchor_node_id
    block_id = env.tree.lines.get_node(meal).block_id
    assert len(env.tree.lines.lines_for_block(block_id)) == 2  # the trees meet here


def test_taking_up_an_ended_matter_opens_a_new_thread_anchored_on_the_old_step(env):
    env.step(0, ("The user decided not to buy the laptop: no budget.",
                 [A("Buy the laptop?", done=True)]))
    decided = env.tree.lines.nodes_for_line(env.tree.lines.list_lines()[0].line_id)[0].node_id
    known = env.tree.known(batch_text="laptop again", now=T0 + timedelta(days=30),
                           threads_policy={"recent": 5, "cued": 5, "due": 5},
                           steps_policy={"recent": 5, "cued": 5})
    old = next(ref for ref, node in known.refs.items() if node == decided)
    env.step(24 * 30, ("The user wants the laptop again after a raise.",
                       [A("Buy the laptop?", after=old, committed=False)]), refs=known.refs)
    again = env.thread("Buy the laptop?")
    assert again.anchor_node_id == decided  # index to the old step, which is unchanged
    assert len(env.tree.lines.nodes_for_line(env.tree.lines.list_lines()[0].line_id)) == 1


def test_known_shows_threads_steps_and_rejections_with_refs(env):
    env.step(0, ("The user arrived in Lisbon.", [A(TRIP)]))
    line_id = env.tree.lines.list_lines()[0].line_id
    env.tree.rejections.reject(region_id=line_id, content="The user dislikes Lisbon.",
                               source_refs=("x",), authority_ref="user",
                               rejected_at=T0 + timedelta(hours=1))
    known = env.tree.known(batch_text="Lisbon", now=T0 + timedelta(hours=2),
                           threads_policy={"recent": 5, "cued": 5, "due": 5},
                           steps_policy={"recent": 5, "cued": 5})
    assert known.payload["threads"][0]["question"] == TRIP
    assert known.payload["threads"][0]["after"] == "N1"
    assert known.payload["steps"][0]["text"] == "The user arrived in Lisbon."
    assert known.payload["rejected"] == ["The user dislikes Lisbon."]
    assert known.refs["T1"] == env.thread(TRIP).thread_id
