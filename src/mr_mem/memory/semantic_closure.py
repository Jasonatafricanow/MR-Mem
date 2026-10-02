"""Pure connected-component closure; input meaning is never reinterpreted."""

import re
from dataclasses import dataclass

from mr_mem.memory.semantic_contracts import (
    BoundaryPolicy,
    DeltaSemanticDependency,
    DependencyTargetKind,
    SemanticDeltaV1,
)
from mr_mem.memory.semantic_validator import validate_semantic_delta


def local_order(point_id: str):
    """Point IDs carry stable local order; array/edge ordering grants no authority."""
    match = re.fullmatch(r"(.*?)([0-9]+)", point_id)
    return (match[1], int(match[2]), point_id) if match else (point_id, -1, point_id)


@dataclass(frozen=True, slots=True)
class CompiledSemanticBlock:
    member_point_ids: tuple[str, ...]
    analysis_text: str
    context_memory_ids: tuple[str, ...]
    dependencies: tuple[DeltaSemanticDependency, ...]


def compile_semantic_delta(
    payload: SemanticDeltaV1 | dict | str,
    *,
    activated_memory_ids: tuple[str, ...],
) -> tuple[CompiledSemanticBlock, ...]:
    delta = validate_semantic_delta(payload, activated_memory_ids=activated_memory_ids)
    points = {p.point_id: p for p in delta.points if p.eligible}
    parent = {pid: pid for pid in points}

    def find(pid):
        while parent[pid] != pid:
            parent[pid] = parent[parent[pid]]
            pid = parent[pid]
        return pid

    for edge in delta.dependencies:
        if (
            edge.from_point_id in points
            and edge.target_kind is DependencyTargetKind.POINT
            and edge.target_id in points
            and edge.boundary_policy is BoundaryPolicy.COHABIT
        ):
            left, right = find(edge.from_point_id), find(edge.target_id)
            parent[max((left, right), key=local_order)] = min((left, right), key=local_order)
    groups: dict[str, list[str]] = {}
    for pid in sorted(points, key=local_order):
        groups.setdefault(find(pid), []).append(pid)
    blocks = []
    for members in groups.values():
        dependencies = tuple(
            sorted(
                {edge for edge in delta.dependencies if edge.from_point_id in members},
                key=lambda e: (
                    local_order(e.from_point_id),
                    e.target_kind,
                    e.target_id,
                    e.relation,
                    e.boundary_policy,
                    e.lifecycle_effect,
                ),
            )
        )
        blocks.append(
            CompiledSemanticBlock(
                tuple(members),
                "\n".join(points[pid].meaning for pid in members),
                tuple(
                    sorted(
                        {
                            e.target_id
                            for e in dependencies
                            if e.target_kind is DependencyTargetKind.MEMORY
                        }
                    )
                ),
                dependencies,
            )
        )
    return tuple(blocks)
