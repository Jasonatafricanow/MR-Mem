"""LCE Lines driven by compiler signals (design v6 4.4-4.5).

Every Block that connects is on its line from the moment it is said; Thread and
Path A are the two sides of a terminator that moves with time. A matter has no
name: a code id, reached through a ref; its meaning lives in its summary.
"""

from datetime import UTC, datetime, timedelta

import pytest

from mr_mem import MemoryCore, SemanticAdmissionService, SemanticMemoryCandidate, SourceRef
from mr_mem.lce.path_a import DAY, PAST, Attachment, BlockLines, PathATree
from tests.test_native_semantics import SCOPE

T0 = datetime(2026, 5, 1, 8, tzinfo=UTC)
POLICY = {"threads_policy": {"recent": 9, "cued": 9, "due": 9},
          "steps_policy": {"recent": 20, "cued": 20}}


def at(hours):
    return T0 + timedelta(hours=hours)


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
        self.memory_of = {}

    def block(self, content, hours):
        ref = SourceRef("host", "s1", f"r{hours}-{len(self.host.refs)}", at(hours), "1")
        self.host.refs[ref.source_key] = ref
        admission = SemanticAdmissionService(store=self.core.canonical, sources=self.host,
                                             clock=Clock(), origin_runtime_id="t")
        memory = admission.admit(SemanticMemoryCandidate(
            semantic_id=ref.record_id, scope=SCOPE, content=content, source_refs=(ref,),
            compiler_version="test",
        ))
        self.memory_of[content] = memory.memory_id
        return memory

    def look(self, hours, text=""):
        """What the compiler would see before a batch at ``hours``."""
        self.known = self.tree.known(batch_text=text, now=at(hours), **POLICY)
        return self.known

    def ref(self, thread_id=None, node_id=None):
        target = thread_id or node_id
        return next(r for r, v in self.known.refs.items() if v == target)

    def step(self, hours, *items, settles=()):
        """items: (content, [Attachment, ...]) compiled in one batch at ``hours``."""
        blocks = [BlockLines(self.block(c, hours).memory_id, tuple(lines),
                             tuple(settles) if n == 0 else ())
                  for n, (c, lines) in enumerate(items)]
        refs = self.known.refs if self.known else {}
        self.known = None
        return self.tree.apply(blocks, refs, at(hours))

    def node(self, content):
        block_id = self.memory_of[content]
        return next(n.node_id for line in self.tree.lines.list_lines()
                    for n in self.tree.lines.nodes_for_line(line.line_id)
                    if n.block_id == block_id)

    def matter(self, thread_id):
        return self.core.products.get_thread(thread_id)

    def close(self):
        self.tree.close()
        self.core.close()


@pytest.fixture
def env(tmp_path):
    value = Env(tmp_path)
    yield value
    value.close()


def A(thread=None, after=None, happened=True, summary="s"):  # noqa: N802
    return Attachment(thread, after, happened, summary)


WANT = "The user wants to go to Portugal."
BOOKED = "The user booked a flight to Lisbon for 2026-05-02 09:00."
TRAFFIC = "The user is stuck in traffic on the way to the airport."
MAY_MISS = "The user may miss the 2026-05-02 09:00 flight."
ARRIVED = "The user arrived in Lisbon."
AGREED = "Ana and the user agreed to have dinner tomorrow."
DINNER = "The user and Ana will have dinner tomorrow (2026-05-03 19:00)."
SHOPPING = "The user went shopping on Rua Augusta."


def test_every_connected_block_is_on_its_line_and_time_moves_the_terminator(env):
    trip = env.step(0, (WANT, [A(happened=False, summary="Trip to Portugal wanted")])).opened[0]
    assert env.tree.side(env.node(WANT), at(0)) == DAY  # an intention: on the line, day side
    env.look(1)
    env.step(1, (BOOKED, [A(env.ref(trip))]))
    assert env.tree.lines.parents_at(env.node(BOOKED), at(2)) == (env.node(WANT),)
    env.look(24)
    t = env.ref(trip)
    env.step(24, (TRAFFIC, [A(t)]),
             (MAY_MISS, [A(t, after="b1", happened=False)]))
    assert env.tree.side(env.node(MAY_MISS), at(24.5)) == DAY
    # the time stated in the content passed (two time axes only: the stated time is
    # content): frozen is "this was feared then", whatever happened next
    assert env.tree.side(env.node(MAY_MISS), at(25.5)) == PAST
    env.look(30)
    env.step(30, (ARRIVED, [A(env.ref(trip), after=env.ref(node_id=env.node(MAY_MISS)))]))
    env.look(31, "dinner with Ana")
    arrived = env.ref(node_id=env.node(ARRIVED))
    env.step(31, (AGREED, [A(after=arrived)]),
             (DINNER, [A("b1", after="b1", happened=False)]),
             (SHOPPING, [A(after=arrived)]))
    # dinner and shopping are parallel branches under "arrived"
    assert set(env.tree.lines.children_at(env.node(ARRIVED), at(32))) == {
        env.node(AGREED), env.node(SHOPPING)}
    assert env.tree.side(env.node(DINNER), at(40)) == DAY
    assert env.tree.side(env.node(DINNER), at(60)) == PAST  # the terminator moved on
    assert len(env.tree.lines.list_lines()) == 1  # one trip, one tree


def test_an_intention_without_time_grows_several_branches_and_is_settled_by_meaning(env):
    want = "The user wants to buy a new laptop."
    laptop = env.step(0, (want, [A(happened=False, summary="Wants a laptop")])).opened[0]
    env.look(5, "laptop")
    w = env.ref(node_id=env.node(want))
    branches = ("Laptop prices went up this month.", "The user's salary is not paid yet.",
                "The user saw nothing appealing.", "The user's old laptop still works fine.")
    env.step(5, *[(b, [A(env.ref(laptop), after=w)]) for b in branches])
    assert set(env.tree.lines.children_at(env.node(want), at(6))) == {
        env.node(b) for b in branches}
    # no stated time: still on the day side however much time passes...
    assert env.tree.side(env.node(want), at(24 * 60)) == DAY
    # ...until a block settles it by meaning
    env.look(24 * 61, "laptop")
    env.step(24 * 61, ("The user decided not to buy a laptop: the old one is enough.",
                       [A(env.ref(laptop), after=env.ref(node_id=env.node(branches[3])))]),
             settles=(env.ref(node_id=env.node(want)),))
    assert env.tree.side(env.node(want), at(24 * 61)) == PAST
    assert env.tree.side(env.node(want), at(24 * 60)) == DAY  # history is kept


def test_a_block_that_connects_to_nothing_stays_alone(env):
    env.step(0, ("The user is refactoring the Flask routes.", [A()]))
    egg = "The user said the egg they just ate was tasty."
    result = env.step(1, (egg, []))
    assert result.attached == 0
    assert all(n.block_id != env.memory_of[egg]
               for line in env.tree.lines.list_lines()
               for n in env.tree.lines.nodes_for_line(line.line_id))


def test_a_matter_has_no_name_and_its_summary_follows_it(env):
    trip = env.step(0, (WANT, [A(happened=False, summary="Trip: Portugal")])).opened[0]
    env.look(5)
    env.step(5, ("The user decided to go on to Spain and Morocco as well.",
                 [A(env.ref(trip), summary="Trip: Portugal, then Spain and Morocco")]))
    assert env.matter(trip).working_summary == "Trip: Portugal, then Spain and Morocco"
    known = env.look(6, "Morocco")
    assert known.payload["threads"][0]["summary"] == "Trip: Portugal, then Spain and Morocco"
    assert {s["state"] for s in known.payload["steps"]} == {DAY, PAST}
    # null is a new matter even with similar words
    other = env.step(6, ("The user plans a Portugal trip next year with family.",
                         [A(happened=False, summary="Trip: Portugal with family")]))
    assert other.opened and other.opened[0] != trip


def test_one_block_meets_two_trees_of_different_kinds(env):
    trip = env.step(0, (ARRIVED, [A()])).opened[0]
    env.look(1)
    food = env.step(1, ("The user likes trying local food.", [A()])).opened[0]
    env.look(2, "bacalhau Lisbon local food")
    meal = "The user ate bacalhau at a local tasca in Lisbon."
    result = env.step(2, (meal, [A(env.ref(trip)), A(env.ref(food))]))
    assert result.attached == 2 and not result.failures
    assert len(env.tree.lines.lines_for_block(env.memory_of[meal])) == 2  # the trees meet


def test_known_shows_rejections(env):
    env.step(0, (ARRIVED, [A(summary="Trip to Lisbon")]))
    line_id = env.tree.lines.list_lines()[0].line_id
    env.tree.rejections.reject(region_id=line_id, content="The user dislikes Lisbon.",
                               source_refs=("x",), authority_ref="user", rejected_at=at(1))
    known = env.look(2, "Lisbon")
    assert known.payload["threads"] == [{"ref": "T1", "summary": "Trip to Lisbon", "latest": "N1"}]
    assert known.payload["rejected"] == ["The user dislikes Lisbon."]
