"""MR-Mem's search and write as MCP tools, on 127.0.0.1 (11-09). One long-running process: opening the
34 compiled sessions takes 20 s, too long for every agent start. Runs on the thread that opened them:
each runtime answers only there. The tools are async on purpose: the SDK runs a plain function on a
worker thread, an async one on the event loop's thread, which is the one that opened them.

Every call is logged to usage.jsonl beside the written SBs (what was asked, how big the answer was, errors), to see how
agents actually use the tools.

usage: mcp_search.py [port]     (default 8771, path /mcp)
Environment: MRMEM_WORK_DIR, MRMEM_LIVE_DIR, MRMEM_USER and the embedding key of the compiled runtime.json."""

import json
import sys
import time
from datetime import UTC, datetime

import search as history
import write_sb
from mcp.server.mcpserver import MCPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8771
USAGE = write_sb.STORE.parent / "usage.jsonl"
h = history.History()
mcp = MCPServer("mr-mem")


def logged(tool: str, args: dict, call):
    """Run the call, append one usage line, and pass on its result or its error."""
    started, row = time.time(), {"at": datetime.now(UTC).isoformat(), "tool": tool, **args}
    try:
        result, facts = call()
        row.update(facts)
        return result
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        raise
    finally:
        row["ms"] = round((time.time() - started) * 1000)
        with USAGE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


@mcp.tool()
async def search(query: str, tokens: int = history.TOKENS) -> str:
    """Search the person's past work sessions with their agents (decisions, corrections, what was
    tried and why). Returns the matters found, each with the original words under it. A place reads
    "S7 #8686": line 8686 of session S7's file; the files are listed at the end. One call gives both,
    no follow-up needed. `tokens` limits the size of the answer: the least relevant parts are left
    out first. The default suits most questions."""

    def call():
        got = h.search(query, tokens)
        files = "\n".join(f"{k} = {v}" for k, v in got["files"].items())
        return f"{got['content']}\n\n【原文件】\n{files}", {
            "chars": len(got["content"]),
            "sessions": len(got["files"]),
        }

    return logged("search", {"query": query, "tokens": tokens}, call)


@mcp.tool()
async def write(sbs: list[dict]) -> str:
    """Record what this conversation has been about, from what you already understand; no need to
    re-read anything. One item per matter being talked about, in order: {"from": the opening words of
    the message where the matter starts, copied exactly, enough to be unique; "text": one or two
    sentences, every point the person made first, in their tone, then what you answered or did, only
    what actually happened; details stay in the original words; "when": "before" if it has already
    happened or been settled, "after" if still ahead; "status": "confirmed", "likely", "negated" or
    "unknown"; "standing": true if it is a lasting instruction or preference}. Only write what has
    not been written yet. It is searchable at once. If a quote is not found nothing is written."""

    def call():
        records = write_sb.write(sbs)
        h.add_written(records)
        first, last = records[0]["source"], records[-1]["source"]
        return (
            f"wrote {len(records)} SBs from {first['file']}, lines {first['first_line']}-{last['last_line']}",
            {
                "file": first["file"],
                "first_line": first["first_line"],
                "last_line": last["last_line"],
            },
        )

    return logged("write", {"sbs": len(sbs)}, call)


if __name__ == "__main__":
    try:
        mcp.run("streamable-http", host="127.0.0.1", port=PORT)
    finally:
        h.close()
