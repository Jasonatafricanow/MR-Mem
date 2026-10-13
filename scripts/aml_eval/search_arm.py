"""Run every evaluation question against one read arm and save raw Search results.

Usage:
    python scripts/aml_eval/search_arm.py --arm C --mode path_a \
        --questions questions.jsonl --out results/C.search.jsonl \
        [--base-url http://127.0.0.1:8765] [--top-k 100] [--token-env MRMEM_TOKEN]

questions.jsonl: one object per line with "id", "query", "user_id" and optional
"options" (multiple choice). The output is resumable: ids already written are
skipped. Each output line keeps the Search response exactly as returned
("data", in rank order) plus a few counters used for the context-size report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ANSWER_INPUT_TOKENS = 117_760  # AML answer window minus output and safety tokens


def post(base: str, payload: dict, mode: str, token: str | None) -> dict:
    headers = {"Content-Type": "application/json", "X-MRMem-Search-Mode": mode}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        base.rstrip("/") + "/search", data=json.dumps(payload).encode("utf-8"),
        headers=headers, method="POST",
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 425, 429, 500, 502, 503, 504) or attempt == 5:
                raise
        except urllib.error.URLError:
            if attempt == 5:
                raise
        time.sleep(min(60, 2 ** attempt * 5))
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--mode", required=True, choices=("baseline", "path_a"))
    parser.add_argument("--questions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--token-env")
    args = parser.parse_args()
    token = os.environ.get(args.token_env) if args.token_env else None
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text("utf-8").splitlines() if line}
    questions = [
        json.loads(line) for line in Path(args.questions).read_text("utf-8").splitlines() if line
    ]
    with out.open("a", encoding="utf-8") as handle:
        for q in questions:
            if q["id"] in done:
                continue
            payload = {"query": q["query"], "user_id": q["user_id"], "top_k": args.top_k}
            if q.get("options"):
                payload["options"] = q["options"]
            data = post(args.base_url, payload, args.mode, token)["data"]
            kinds = [item["id"].split(":", 1)[0] for item in data]
            chars = sum(len(item["content"]) for item in data)
            handle.write(json.dumps({
                "id": q["id"], "arm": args.arm, "mode": args.mode, "data": data,
                "items": len(data), "lce_items": kinds.count("lce"),
                "thread_items": kinds.count("thread"), "source_items": kinds.count("source"),
                "chars": chars, "approx_tokens": chars // 4,
                "may_truncate": chars // 4 > ANSWER_INPUT_TOKENS,
            }, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
