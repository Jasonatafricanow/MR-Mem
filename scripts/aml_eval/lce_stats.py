"""LCE / Thread statistics for one MR-Mem AML data_dir.

Usage: python scripts/aml_eval/lce_stats.py <runtime.json>   (stop the server first)
"""
import json
import sys
from collections import Counter

from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime
from mr_mem_aml.source_store import user_scope

rt = MemoryRuntime(RuntimeConfig.load(sys.argv[1]))
try:
    memories = rt.core.load_all()
    users = sorted({m.scope.user_id for m in memories})
    origins, revisions, lost, lines, threads = Counter(), Counter(), 0, 0, 0
    for uid in users:
        scope = user_scope(uid)
        threads += len(rt.core.products.list_threads(scope))
        lce = rt.lce_for(scope)
        for head in () if lce is None else lce.accepted():
            lines += 1
            origins[head.origin] += 1
            revisions[head.revision_number] += 1
            lost += not head.evidence_complete
    print(json.dumps({
        "users": len(users),
        "canonical_blocks": len(memories),
        "lce_lines": lines,
        "lce_lines_per_user": round(lines / max(len(users), 1), 2),
        "head_origin": dict(origins),
        "head_revision_number": dict(sorted(revisions.items())),
        "lines_with_lost_evidence": lost,
        "open_threads": threads,
        "thread_failures_this_process": rt.thread_failures,
    }, indent=1))
finally:
    rt.close()
