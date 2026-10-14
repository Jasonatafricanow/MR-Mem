"""LCE Thread + Path A tree driven by compiler signals (design v6 4.4-4.5).

A thread has no name: its identity is a code id the compiler reaches through a
ref ("T1", or "b<k>" for a thread opened earlier in the same batch); what it is
about lives in its summary, which follows the matter as it changes.
"""

from datetime import UTC, datetime, timedelta

import pytest

from mr_mem import MemoryCore, SemanticAdmissionService, SemanticMemoryCandidate, SourceRef
from mr_mem.lce.path_a import Attachment, BlockLines, PathATree
from mr_mem.memory.product import ThreadStatus
from tests.test_native_semantics import SCOPE

T0 = datetime(2026, 5, 1, 8, tzinfo=UTC)
POLICY = {"threads_policy": {"recent": 9, "cued": 9, "due": 9},
          "steps_policy": {"recent": 20, "cued": 20}}


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
        self.known = None

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

    def look(self, hours, text=""):
        """What the compiler would see before a batch at ``hours``."""
        self.known = self.tree.known(batch_text=text, now=T0 + timedelta(hours=hours), **POLICY)
        return self.known

    def ref(self, thread_id=None, node_id=None):
        target = thread_id or node_id
        return next(r for r, v in self.known.refs.items() if v == target)

    def step(self, hours, *items):
        """items: (content, [Attachment, ...]) compiled in one batch at ``hours``."""
        blocks = [BlockLines(self.block(c, hours).memory_id, tuple(lines)) for c, lines in items]
        refs = self.known.refs if self.known else {}
        self.known = None
        return self.tree.apply(blocks, refs, T0 + timedelta(hours=hours))

    def thread(self, thread_id):
        return self.core.products.get_thread(thread_id)

    def close(self):
        self.tree.close()
        self.core.close()


@pytest.fixture
def env(tmp_path):
    value = Env(tmp_path)
    yield value
    value.close()


def A(thread=None, after=None, committed=True, summary="s", done=False):  # noqa: N802
    return Attachment(thread, after, committed, summary, done)


def test_portugal_trip_grows_a_tree_with_parallel_branches(env):
    # an intention: nothing has happened yet, so no tree, only a Thread
    trip = env.step(0, ("The user wants to go to Portugal.",
                        [A(committed=False, summary="Wants to go to Portugal")])).opened[0]
    assert env.thread(trip).anchor_node_id is None and env.tree.lines.list_lines() == ()
    # booked: the first committed step roots a Line; the Thread hangs from it
    env.look(1)
    env.step(1, ("The user booked a flight to Lisbon.", [A(env.ref(trip))]))
    booked = env.thread(trip).anchor_node_id
    assert booked is not None and len(env.tree.lines.list_lines()) == 1
    # traffic happened (committed), missing the flight is still open (uncommitted)
    env.look(5)
    t = env.ref(trip)
    env.step(5, ("The user is stuck in traffic on the way to the airport.", [A(t)]),
             ("The user may miss the flight.", [A(t, after="b1", committed=False)]))
    traffic = env.thread(trip).anchor_node_id
    assert env.tree.lines.parents_at(traffic, T0 + timedelta(hours=6)) == (booked,)
    assert len(env.thread(trip).current_support_ids) == 1  # the open step waits on the Thread
    env.look(6)
    env.step(6, ("The user made the flight; it took off.", [A(env.ref(trip))]))
    env.look(9)
    env.step(9, ("The user arrived in Lisbon.", [A(env.ref(trip))]))
    arrived = env.thread(trip).anchor_node_id
    # "agreed to have dinner tomorrow": the agreement happened; the dinner has not
    env.look(10, "dinner with Ana")
    at_arrival = env.ref(node_id=arrived)
    result = env.step(
        10,
        ("Ana and the user agreed to have dinner tomorrow.", [A(after=at_arrival)]),
        ("The user and Ana will have dinner tomorrow evening.",
         [A("b1", after="b1", committed=False)]),  # same new thread, opened by block 1
        ("The user went shopping on Rua Augusta.", [A(after=at_arrival)]),
    )
    dinner, shopping_thread = result.opened
    agreed = env.thread(dinner).anchor_node_id  # the open dinner hangs from "agreed"
    shopping = env.thread(shopping_thread).anchor_node_id
    now = T0 + timedelta(hours=12)
    # dinner and shopping are parallel branches under "arrived", unrelated to each other
    assert set(env.tree.lines.children_at(arrived, now)) == {agreed, shopping}
    assert env.tree.view.ancestors(shopping, cutoff=now)[0] == arrived  # background, by index
    env.look(30)
    env.step(30, ("The user and Ana had dinner; the food was great.", [A(env.ref(dinner))]))
    # the dinner happened: committed onto its branch, nothing open left on it; lines never
    # end, so both the dinner branch and the trip stay available to grow further
    assert env.thread(dinner).status is ThreadStatus.OPEN
    assert env.thread(trip).status is ThreadStatus.OPEN
    had = env.thread(dinner).anchor_node_id
    assert env.tree.lines.parents_at(had, T0 + timedelta(hours=31)) == (agreed,)


def test_thread_identity_is_a_code_id_and_its_summary_follows_the_matter(env):
    """Portugal became Spain and North Africa: same thread, the summary moved with it."""
    trip = env.step(0, ("The user plans a trip to Portugal.",
                        [A(committed=False, summary="Trip: Portugal planned")])).opened[0]
    env.look(5)
    env.step(5, ("The user decided to go on to Spain and Morocco as well.",
                 [A(env.ref(trip), summary="Trip: Portugal, then Spain and Morocco")]))
    assert env.thread(trip).working_summary == "Trip: Portugal, then Spain and Morocco"
    known = env.look(6, "Morocco")
    assert known.payload["threads"] == [
        {"ref": "T1", "summary": "Trip: Portugal, then Spain and Morocco", "after": "N1"}]
    # a null thread is a new matter even with similar words
    result = env.step(6, ("The user plans a trip to Portugal next year with family.",
                          [A(committed=False, summary="Trip: Portugal with family")]))
    assert result.opened and result.opened[0] != trip


def test_one_block_meets_two_trees_of_different_kinds(env):
    trip = env.step(0, ("The user arrived in Lisbon.", [A()])).opened[0]
    env.look(1)
    food = env.step(1, ("The user likes trying local food.", [A()])).opened[0]
    env.look(2, "bacalhau Lisbon local food")
    result = env.step(2, ("The user ate bacalhau at a local tasca in Lisbon.",
                          [A(env.ref(trip)), A(env.ref(food))]))
    assert result.committed_nodes == 2 and not result.failures
    assert len(env.tree.lines.list_lines()) == 2
    meal = env.thread(trip).anchor_node_id
    block_id = env.tree.lines.get_node(meal).block_id
    assert len(env.tree.lines.lines_for_block(block_id)) == 2  # the trees meet here


def test_developing_a_frozen_decision_continues_from_the_old_step(env):
    """No end: the decision stays a frozen step; later development grows from it."""
    laptop = env.step(0, ("The user decided not to buy the laptop: no budget.",
                          [A(summary="Laptop: decided not to buy, no budget")])).opened[0]
    decided = env.thread(laptop).anchor_node_id
    env.look(24 * 30, "laptop again")
    env.step(24 * 30, ("The user wants the laptop again after a raise.",
                       [A(env.ref(laptop), after=env.ref(node_id=decided), committed=False,
                          summary="Laptop: wants it again after a raise (earlier: no budget)")]))
    again = env.thread(laptop)
    assert again.anchor_node_id == decided  # grows from the old step, which is unchanged
    assert len(env.tree.lines.nodes_for_line(env.tree.lines.list_lines()[0].line_id)) == 1
    assert again.working_summary.startswith("Laptop: wants it again")


def test_known_shows_threads_steps_and_rejections_with_refs(env):
    trip = env.step(0, ("The user arrived in Lisbon.", [A(summary="Trip to Lisbon")])).opened[0]
    line_id = env.tree.lines.list_lines()[0].line_id
    env.tree.rejections.reject(region_id=line_id, content="The user dislikes Lisbon.",
                               source_refs=("x",), authority_ref="user",
                               rejected_at=T0 + timedelta(hours=1))
    known = env.look(2, "Lisbon")
    assert known.payload["threads"] == [{"ref": "T1", "summary": "Trip to Lisbon", "after": "N1"}]
    assert known.payload["steps"][0]["text"] == "The user arrived in Lisbon."
    assert known.payload["rejected"] == ["The user dislikes Lisbon."]
    assert known.refs["T1"] == trip
