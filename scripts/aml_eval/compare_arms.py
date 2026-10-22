"""Paired comparison of judged evaluation arms.

Usage:
    python scripts/aml_eval/compare_arms.py --arm A=results/A.judged.jsonl \
        --arm A+=results/A+.judged.jsonl --arm C=results/C.judged.jsonl \
        [--search A=results/A.search.jsonl ...] [--pair C:A+ --pair A+:A] [--out report.json]

Judged files: one object per line with "id" and "is_correct" (bool), or a numeric
"score" in [0, 1]; an optional "category" (e.g. question type) adds a breakdown.
Only ids present in every arm are compared. Pairs report the mean difference,
a paired bootstrap 95% interval, win/loss counts and, for binary labels, an
exact two-sided McNemar p-value. Search files add context-size statistics.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from math import comb
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    rows = {}
    for line in Path(path).read_text("utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def value(row: dict) -> float:
    if "is_correct" in row:
        return 1.0 if row["is_correct"] else 0.0
    if "score" in row:
        return float(row["score"])
    raise SystemExit(f"row {row.get('id')!r} has neither is_correct nor score (judged file?)")


def mcnemar(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(wins, losses) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired(ids, left, right, rounds=10_000, seed=7):
    diffs = [value(left[i]) - value(right[i]) for i in ids]
    mean = sum(diffs) / len(diffs)
    rng = random.Random(seed)
    boots = sorted(
        sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(rounds)
    )
    wins = sum(d > 0 for d in diffs)
    losses = sum(d < 0 for d in diffs)
    binary = all(d in (-1.0, 0.0, 1.0) for d in diffs)
    return {
        "mean_diff": round(mean, 4),
        "ci95": [round(boots[int(0.025 * rounds)], 4), round(boots[int(0.975 * rounds)], 4)],
        "wins": wins, "losses": losses, "ties": len(diffs) - wins - losses,
        "mcnemar_p": round(mcnemar(wins, losses), 4) if binary else None,
    }


def parse_pairs(items):
    return [tuple(item.split(":", 1)) for item in items]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", action="append", required=True, help="NAME=judged.jsonl")
    parser.add_argument("--search", action="append", default=[], help="NAME=search.jsonl")
    parser.add_argument("--pair", action="append", default=[], help="LEFT:RIGHT")
    parser.add_argument("--out")
    args = parser.parse_args()
    arms = {name: load(path) for name, path in (a.split("=", 1) for a in args.arm)}
    ids = sorted(set.intersection(*(set(rows) for rows in arms.values())))
    if not ids:
        raise SystemExit("no question ids shared by all arms")
    report: dict = {"questions": len(ids), "accuracy": {}, "by_category": {}, "pairs": {}}
    for name, rows in arms.items():
        report["accuracy"][name] = round(sum(value(rows[i]) for i in ids) / len(ids), 4)
        groups = defaultdict(list)
        for i in ids:
            groups[str(rows[i].get("category", "all"))].append(value(rows[i]))
        report["by_category"][name] = {
            k: {"n": len(v), "acc": round(sum(v) / len(v), 4)} for k, v in sorted(groups.items())
        }
    pairs = parse_pairs(args.pair) or [
        (left, right) for left, right in (("C", "A+"), ("A+", "A"), ("C", "A"))
        if left in arms and right in arms
    ]
    for left, right in pairs:
        report["pairs"][f"{left} vs {right}"] = paired(ids, arms[left], arms[right])
    if args.search:
        report["context"] = {}
        for name, path in (s.split("=", 1) for s in args.search):
            rows = [r for i, r in load(path).items() if i in set(ids)]
            n = max(len(rows), 1)
            report["context"][name] = {
                "mean_items": round(sum(r["items"] for r in rows) / n, 1),
                "mean_approx_tokens": round(sum(r["approx_tokens"] for r in rows) / n),
                "questions_with_lce": sum(r["lce_items"] > 0 for r in rows),
                "questions_with_thread": sum(r["thread_items"] > 0 for r in rows),
                "may_truncate": sum(r["may_truncate"] for r in rows),
            }
    text = json.dumps(report, ensure_ascii=False, indent=1)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
