"""Path A Line tree ported from LCE, with the v6 audit changes."""

from datetime import UTC, datetime, timedelta

import pytest

from mr_mem.lce.line_graph import (
    LineGraphStore,
    LineGraphView,
    LineTraversalLimitExceeded,
    StateView,
)

T0 = datetime(2026, 5, 1, 8, tzinfo=UTC)


def at(hours: float) -> datetime:
    return T0 + timedelta(hours=hours)


class Tree:
    """A store plus canonical states ``{state_id: (content, occurred, evidence_complete)}``."""

    def __init__(self, tmp_path):
        self.store = LineGraphStore(tmp_path / "lines.sqlite")
        self.store.set_derivation_fingerprint("compiler-test")
        self.states: dict[str, tuple[str, datetime, bool]] = {}
        self.view = LineGraphView(store=self.store, read_state=self.read)

    def read(self, state_id):
        if state_id not in self.states:
            return None
        content, occurred, complete = self.states[state_id]
        return StateView(state_id, content, occurred, (f"src:{state_id}",), complete)

    def line(self, seed):
        return self.store.create_line((seed,), created_at=T0).line_id

    def node(self, line_id, block, hours, *, state=None, parent=None, known=None):
        state = state or block
        self.states.setdefault(state, (block, at(hours), True))
        known = known or at(hours)
        node, _, _ = self.store.ensure_node(
            line_id, block_id=block, state_id=state, occurred_start=at(hours),
            occurred_end=at(hours), knowledge_at=known,
        )
        if parent is not None:
            self.store.add_edge(line_id, parent, node.node_id, knowledge_at=known)
        return node.node_id


@pytest.fixture
def tree(tmp_path):
    value = Tree(tmp_path)
    yield value
    value.store.close()


def test_one_experience_branches_under_a_shared_step(tree):
    """Dinner, shopping and Porto hang under "arrived": parallel, unrelated branches."""
    line = tree.line("booked")
    booked = tree.node(line, "booked", 0)
    flew = tree.node(line, "flew", 5, parent=booked)
    arrived = tree.node(line, "arrived", 9, parent=flew)
    dinner = tree.node(line, "dinner", 30, parent=arrived)
    shop = tree.node(line, "shopping", 31, parent=arrived)
    porto = tree.node(line, "porto", 50, parent=dinner)
    now = at(60)
    assert set(tree.view.frontier(line, cutoff=now)) == {shop, porto}
    assert set(tree.view.paths_to_frontier(line, cutoff=now)) == {
        (booked, flew, arrived, shop), (booked, flew, arrived, dinner, porto),
    }
    assert tree.view.ancestors(shop, cutoff=now) == (arrived, flew, booked)  # background
    assert set(tree.view.descendants(arrived, cutoff=now)) == {dinner, shop, porto}
    assert tree.view.descendants(shop, cutoff=now) == ()
    # before the dinner was known, the tree ended at "arrived"
    assert tree.view.frontier(line, cutoff=at(20)) == (arrived,)


def test_corrected_block_is_a_new_state_of_the_same_node(tree):
    line = tree.line("a")
    node = tree.node(line, "a", 0)
    tree.states["a-fixed"] = ("a, understood correctly", at(0), True)
    again = tree.node(line, "a", 0, state="a-fixed", known=at(10))
    assert again == node and len(tree.store.nodes_for_line(line)) == 1
    assert tree.view.state_for_node_at_cutoff(node, cutoff=at(5)).content == "a"
    later = tree.view.state_for_node_at_cutoff(node, cutoff=at(11))
    assert later.content == "a, understood correctly"
    assert [n.node_id for n in tree.store.nodes_for_state("a-fixed")] == [node]


def test_retired_relation_keeps_history(tree):
    line = tree.line("a")
    a = tree.node(line, "a", 0)
    b = tree.node(line, "b", 1, parent=a)
    assert tree.store.retire_edge_from(line, a, b, at(5))
    assert tree.store.parents_at(b, at(3)) == (a,)  # what was known then
    assert tree.store.parents_at(b, at(6)) == ()


def test_lost_evidence_greys_the_node_but_never_hides_it(tree):
    """Audit 1: LCE hid a node and its whole subtree; raw is authority, projections persist."""
    line = tree.line("a")
    a = tree.node(line, "a", 0)
    b = tree.node(line, "b", 1, parent=a)
    tree.states["a"] = ("a", at(0), False)  # source evidence of "a" is gone
    now = at(5)
    assert set(tree.view.visible_node_ids(line, cutoff=now)) == {a, b}
    keys, complete = tree.view.raw_closure(b, cutoff=now)
    assert keys == ("src:a", "src:b") and complete is False
    assert tree.view.raw_closure(b, cutoff=now)[1] is False
    tree.states["a"] = ("a", at(0), True)
    assert tree.view.raw_closure(b, cutoff=now)[1] is True


def test_cycles_are_rejected_and_shortcuts_are_not_stored(tree):
    line = tree.line("a")
    a = tree.node(line, "a", 0)
    b = tree.node(line, "b", 1, parent=a)
    c = tree.node(line, "c", 2, parent=b)
    with pytest.raises(ValueError, match="cycle"):
        tree.store.add_edge(line, c, a, knowledge_at=at(3))
    assert tree.store.add_edge(line, a, c, knowledge_at=at(3)) is False
    assert tree.store.parents_at(c, at(4)) == (b,)


def test_edges_cannot_cross_lines(tree):
    one, two = tree.line("a"), tree.line("x")
    a = tree.node(one, "a", 0)
    x = tree.node(two, "x", 0)
    with pytest.raises(ValueError, match="cross"):
        tree.store.add_edge(one, a, x, knowledge_at=at(1))


def test_one_block_may_belong_to_two_lines(tree):
    trip, work = tree.line("trip"), tree.line("work")
    tree.node(trip, "call the client from the airport", 1)
    tree.node(work, "call the client from the airport", 1)
    assert set(tree.store.lines_for_block("call the client from the airport")) == {trip, work}


def test_traversal_bounds_fail_closed(tree):
    line = tree.line("n0")
    parent = tree.node(line, "n0", 0)
    for i in range(1, 6):
        parent = tree.node(line, f"n{i}", i, parent=parent)
    with pytest.raises(LineTraversalLimitExceeded):
        tree.view.paths_to_frontier(line, cutoff=at(10), max_nodes=3)
    with pytest.raises(LineTraversalLimitExceeded):
        tree.view.raw_closure(parent, cutoff=at(10), max_nodes=2)


def test_knowledge_time_is_required():
    """Audit 2: no wall-clock defaults; replay must be deterministic."""
    import inspect

    signature = inspect.signature(LineGraphStore.add_edge)
    assert signature.parameters["knowledge_at"].default is inspect.Parameter.empty
