"""Closure units are blocks, with stable local identity and no old text hydration."""

from copy import deepcopy

import pytest

from mr_mem.memory.semantic_closure import compile_semantic_delta
from tests.test_semantic_validator import delta, edge, point


@pytest.mark.parametrize(
    "edges,expected",
    [
        ([], (("p1",), ("p2",), ("p3",))),
        ([("p1", "p2", "cohabit")], (("p1", "p2"), ("p3",))),
        ([("p1", "p2", "cohabit"), ("p2", "p3", "cohabit")], (("p1", "p2", "p3"),)),
        ([("p1", "p2", "context")], (("p1",), ("p2",), ("p3",))),
    ],
)
def test_components_follow_only_cohabit(edges, expected):
    payload = delta()
    payload["points"] = [point(f"p{i}", f"完整意义{i}。") for i in (1, 2, 3)]
    payload["dependencies"] = [
        edge(
            from_point_id=a,
            target_kind="point",
            target_id=b,
            boundary_policy=policy,
            lifecycle_effect="none",
        )
        for a, b, policy in edges
    ]
    blocks = compile_semantic_delta(payload, activated_memory_ids=())
    assert tuple(b.member_point_ids for b in blocks) == expected
    assert all(
        b.analysis_text == "\n".join(f"完整意义{i[1:]}。" for i in b.member_point_ids)
        for b in blocks
    )


def test_order_and_duplicate_edge_invariance_and_defer_exclusion():
    payload = delta()
    payload["points"] = [point("p10", "第三。"), point("p2", "第二。"), point("p1", "第一。")]
    payload["points"].append(point("p4", "不明确。"))
    payload["points"][-1].update(unresolved_refs=["指代"])
    payload["dependencies"] = [
        edge(
            target_kind="point", target_id="p2", boundary_policy="cohabit", lifecycle_effect="none"
        ),
        edge(),
    ]
    baseline = compile_semantic_delta(payload, activated_memory_ids=("memory-old",))
    reordered = deepcopy(payload)
    reordered["points"].reverse()
    reordered["dependencies"].reverse()
    reordered["dependencies"] *= 2
    assert compile_semantic_delta(reordered, activated_memory_ids=("memory-old",)) == baseline
    assert baseline[0].member_point_ids == ("p1", "p2")
    assert baseline[0].analysis_text == "第一。\n第二。"
    assert baseline[0].context_memory_ids == ("memory-old",)
    assert len(baseline) == 2
    assert all("不明确" not in b.analysis_text for b in baseline)


def test_context_memory_is_only_metadata_and_defer_is_not_a_block():
    payload = delta()
    payload["dependencies"] = [edge()]
    (block,) = compile_semantic_delta(payload, activated_memory_ids=("memory-old",))
    assert block.analysis_text == payload["points"][0]["meaning"]
    assert block.context_memory_ids == ("memory-old",)
    payload["points"][0].update(status="defer")
    assert compile_semantic_delta(payload, activated_memory_ids=("memory-old",)) == ()
