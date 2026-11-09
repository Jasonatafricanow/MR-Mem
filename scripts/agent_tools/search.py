"""One search over the person's compiled history (11-09). Replaces query.py, which picked sessions.

Every session was compiled on its own; nothing is imported or merged on disk. On start the raw units
and the SBs of all of them go into two in-memory indexes, so that a question ranks everything
together (design 5.3: sleeping and awake Lines compete equally; 5.4: no per-question threshold). The
readout is composed once by line_read, as for one history: each id it asks about is answered by the
session that holds it. SB and its raw come back together (design 5.5, frozen 5). Standing
instructions are not put in front of every answer here: over the whole history they are 128 SBs,
8.5k tokens per call; they compete like any other SB.

A place reads "S<n> #<line>": line <line> of session n's own file; the answer lists the files. The
words shown are read from that file with the compile's cleaning (secrets redacted). A session whose
file is gone or no longer lines up with what was compiled answers from its compile copy.

SBs the working agent wrote (write_sb.py, live/sb.jsonl) compete with the compiled ones from the
moment they are written. They are not on a Line yet: each is found alone, with its raw.

usage: search.py "问题" [tokens]
MRMEM_WORK_DIR: the compiled sessions. MRMEM_USER: their user id. Keys come from the environment."""

import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha1
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE))
import origin  # noqa: E402
import write_sb  # noqa: E402

from mr_mem import MemoryLifecycle  # noqa: E402
from mr_mem_aml import line_read, moves  # noqa: E402
from mr_mem_aml.config import RuntimeConfig  # noqa: E402
from mr_mem_aml.raw_arm import RawArm, RawUnit, units_for  # noqa: E402
from mr_mem_aml.runtime import MemoryRuntime  # noqa: E402
from mr_mem_aml.source_store import user_scope  # noqa: E402

WORK = Path(os.environ.get("MRMEM_WORK_DIR", "work"))
USER = os.environ.get("MRMEM_USER", "default")
TOKENS = 12000  # default limit on the size of an answer; not tuned yet
MARK, FOLD = "[用户转贴]", "[重复模板，省略]"
PLACE = re.compile(r"\bS(\d+) #")


def unit_rows(vectors) -> np.ndarray:
    m = np.asarray(vectors, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)


def fit(items: list[dict], tokens: int) -> list[dict]:
    """The items within ``tokens``. The stretch summaries go first, the furthest from the question
    first, then the lowest-ranked pieces from the end (as runtime._within, which counts items). A
    first piece that alone is too long is cut at a line."""
    size = lambda x: line_read.estimate_tokens(x["content"])  # noqa: E731
    items = [i for i in items if i["content"].strip()]
    while len(items) > 1 and sum(map(size, items)) > tokens:
        far = [n for n, i in enumerate(items) if str(i["id"]).startswith("thread:")]
        items.pop(far[-1] if far else len(items) - 1)
    if not items or size(items[0]) <= tokens:
        return items
    kept, used = [], 0
    for line in items[0]["content"].split("\n"):
        used += line_read.estimate_tokens(line)
        if used > tokens:
            break
        kept.append(line)
    text = items[0]["content"]
    cut = "\n".join(kept) if kept else text[: int(len(text) * tokens / size(items[0]))]
    return [{**items[0], "content": cut + "\n（已截断，其余见位置号对应的原文件）"}]


class Session:
    """One compiled session, and where its messages are in its own file."""

    def __init__(self, d: Path):
        self.stem = d.name
        self.dir = d
        self.rt = MemoryRuntime(RuntimeConfig.load(d / "runtime.json"))
        src = sqlite3.connect(d / "aml_source.sqlite")
        self.rows = src.execute(
            "select record_id, role, seq, occurred_at from messages order by seq"
        ).fetchall()
        src.close()
        self.first = self.rows[0][3]
        self.fns = None
        got = origin.messages(d.name)
        # the file only ever grows: what was compiled must be its head, message for message
        fits = (
            got is not None
            and len(got[1]) >= len(self.rows)
            and all(r[1] == m["role"] for r, m in zip(self.rows, got[1]))
        )
        self.file = got[0] if fits else None
        self.line = {r[2]: m["line"] for r, m in zip(self.rows, got[1])} if fits else {}
        self.words = {r[0]: m["content"] for r, m in zip(self.rows, got[1])} if fits else {}


class Live:
    """The SBs the working agent wrote from one session file, answering what compose asks by id."""

    rt = None
    words: dict = {}
    line: dict = {}

    def __init__(self, path: Path):
        self.file, self.stem, self.sb, self.msgs, self.dir = path, path.stem, {}, {}, None
        self.fns = {
            "step": self.step,
            "sources": self.sources,
            "message": self.message,
            "parents": lambda b: (),
            "children": lambda b: (),
            "span": lambda b, r: None,
            "pointers": lambda b: None,
        }

    def rid(self, line: int) -> str:
        return f"live:{self.stem}:{line}"

    def reload(self):
        self.file = origin.current(self.file) or self.file
        self.msgs = {m["line"]: m for m in origin.read(self.file)}

    def at(self, line: int) -> datetime | None:
        m = self.msgs.get(line)
        return None if m is None else datetime.fromtimestamp(m["timestamp"] / 1000, UTC)

    def lines(self, b: str) -> list[int]:
        src = self.sb[b]["source"]
        return [n for n in sorted(self.msgs) if src["first_line"] <= n <= src["last_line"]]

    def step(self, b: str) -> line_read.Step:
        first = self.sb[b]["source"]["first_line"]
        return line_read.Step(self.sb[b]["content"], self.at(first), False, (1, first))

    def sources(self, b: str) -> list[str]:
        return [self.rid(n) for n in self.lines(b)]

    def message(self, r: str) -> line_read.RawMessage | None:
        n = int(r.rsplit(":", 1)[1])
        m = self.msgs.get(n)
        return (
            None if m is None else line_read.RawMessage(m["content"], m["role"], self.at(n), (1, n))
        )


class History:
    def __init__(self):
        started = time.time()
        dirs = [d for d in sorted(WORK.iterdir()) if (d / "runtime.json").exists()]
        self.sessions = sorted((Session(d) for d in dirs), key=lambda s: s.first)
        self.policy = line_read.LineReadPolicy.of(self.sessions[0].rt.config.line_read)
        self.embedding = self.sessions[0].rt.raw.embedding
        self.raw, self.sbs = RawArm(self.embedding), RawArm(self.embedding)
        self.owner, self.blocks, self.first_line, self.threads = {}, {}, {}, {}
        for n, s in enumerate(self.sessions):
            index = s.rt.raw._users[USER]
            self.raw.add_vectors([index.units[i] for i in index.ids], index.vectors)
            for rid, *_ in s.rows:
                self.owner[rid] = n
                self.blocks[rid] = s.rt._blocks_by_record.get(rid, ())
            active = [
                m for m in s.rt.core.canonical.load_all() if m.lifecycle is MemoryLifecycle.ACTIVE
            ]
            if active:
                self.sbs.add_vectors(
                    [
                        RawUnit(f"sb:{m.memory_id}", USER, m.content, m.memory_id, None)
                        for m in active
                    ],
                    unit_rows(s.rt._embed_texts([m.content for m in active])),
                )
                firsts = unit_rows(s.rt._embed_texts([m.content.split("\n")[0] for m in active]))
                for m, v in zip(active, firsts):
                    self.owner[m.memory_id] = n
                    self.first_line[m.memory_id] = v
            for t, x in s.rt.source.threads(USER).items():  # thread ids restart in every session
                self.threads[f"{n}:{t}"] = x
        written = write_sb.load()
        for f in dict.fromkeys(r["source"]["file"] for r in written):
            self.add_written([r for r in written if r["source"]["file"] == f])
        print(
            f"opened {len(self.sessions)} sessions, {len(self.first_line)} SBs in {time.time() - started:.0f}s",
            file=sys.stderr,
        )

    def add_written(self, records: list[dict]) -> None:
        """Make SBs from one session file searchable, with the messages they cover."""
        path = Path(records[0]["source"]["file"])
        n = next(
            (k for k, s in enumerate(self.sessions) if isinstance(s, Live) and s.stem == path.stem),
            None,
        )
        if n is None:
            self.sessions.append(Live(path))
            n = len(self.sessions) - 1
        live = self.sessions[n]
        live.reload()
        for r in records:
            b = (
                "live-"
                + sha1(f"{path}:{r['source']['first_line']}:{r['text']}".encode()).hexdigest()[:24]
            )
            content = f"{r['text']} {moves._stamp(r)}"
            live.sb[b] = {**r, "content": content}
            self.owner[b] = n
            self.sbs.add([RawUnit(f"sb:{b}", USER, content, b, None)])
            self.first_line[b] = unit_rows([self.embedding.embed(content.split(chr(10))[0])])[0]
            fresh = [
                SimpleNamespace(
                    role=live.msgs[k]["role"],
                    content=live.msgs[k]["content"],
                    user_id=USER,
                    record_id=live.rid(k),
                    occurred_at=live.at(k),
                )
                for k in live.lines(b)
            ]
            if fresh:
                self.raw.add(units_for(b, fresh))
            for m in fresh:
                self.owner[m.record_id] = n
                self.blocks[m.record_id] = (b,)

    def fn(self, n: int) -> dict:
        s = self.sessions[n]
        if s.fns is None:
            s.fns = s.rt._read_fns(user_scope(USER), USER, "")
        return s.fns

    def place(self, n: int, at):
        """A session's (1, seq) as (the session's number in the whole history, its line in its file)."""
        return None if at is None else (n + 1, self.sessions[n].line.get(at[1], at[1]))

    def search(self, query: str, tokens: int = TOKENS) -> dict:
        if tokens < 1:
            raise ValueError("tokens must be positive")
        policy = replace(self.policy, readout_tokens=tokens, tz_minutes=480)
        q = unit_rows([self.embedding.embed(query[:4096])])[0]
        records = list(
            dict.fromkeys(u.record_id for u, _ in self.raw.search(USER, query, policy.raw_k))
        )
        sb_hits = [u.record_id for u, _ in self.sbs.search(USER, query, policy.sb_k)]

        def by(name):
            return lambda b: self.fn(self.owner[b])[name](b)

        def relevance(b):
            return float(q @ self.first_line[b]) if b in self.first_line else 0.0

        def step(b):
            n, got = self.owner[b], self.fn(self.owner[b])["step"](b)
            return replace(
                got,
                place=self.place(n, got.place),
                later=tuple(self.place(n, a) for a in got.later),
            )

        def message(r):
            n, got = self.owner[r], self.fn(self.owner[r])["message"](r)
            if got is None:
                return None
            fresh = self.sessions[n].words.get(r, got.text)
            # a paste mark or a folded repeat is what the compile decided about these words
            text = got.text if got.text.startswith(MARK) or FOLD in got.text else fresh
            return replace(got, text=text, place=self.place(n, got.place))

        def blocks_of(r):
            return self.blocks.get(r, ())

        compiled = lambda b: b in self.first_line  # noqa: E731
        closest = lambda blocks: max(blocks, key=relevance)  # noqa: E731
        ranked = [
            b
            for b in line_read.rank_blocks(
                records, blocks_of, sb_hits, policy.rrf_k, closest=closest
            )
            if compiled(b)
        ]
        hit_blocks = [
            closest(bs)
            for r in records
            if policy.hit_anchors
            if (bs := [b for b in blocks_of(r) if compiled(b)])
        ]
        items = line_read.compose_within(
            query,
            ranked,
            records,
            fill_records=records,
            hit_parts={},
            uncompiled_records=[r for r in records if not any(compiled(b) for b in blocks_of(r))],
            step=step,
            sources=by("sources"),
            parents=lambda b: self.fn(self.owner[b])["parents"](b),
            children=lambda b: self.fn(self.owner[b])["children"](b),
            message=message,
            lang="zh",
            policy=policy,
            span=lambda b, r: self.fn(self.owner[b])["span"](b, r),
            relevance=relevance,
            pointers=lambda b: self.fn(self.owner[b])["pointers"](b) if b in self.owner else None,
            threads=lambda: self.threads,
            hit_blocks=hit_blocks,
            blocks_of=blocks_of,
        )
        text = "\n\n".join(i["content"] for i in fit(items, tokens))
        files = {}
        for number in sorted({int(x) for x in PLACE.findall(text)}):
            files[f"S{number}"] = self.where(self.sessions[number - 1])
        return {"content": text, "files": files}

    @staticmethod
    def where(s) -> str:
        """The session's file as it is now (it may have been archived or compressed since opening)."""
        if s.file is None:
            return f"{s.dir / 'aml_source.sqlite'}（原文件对不上，#后是消息序号）"
        now = origin.current(s.file)
        if now is None:
            return f"{s.file}（原文件已不在，#后是当时的行号）"
        return f"{now}（zst 压缩，行号按解压后计）" if now.suffix == ".zst" else str(now)

    def close(self):
        for s in self.sessions:
            if s.rt is not None:
                s.rt.close()


if __name__ == "__main__":
    h = History()
    try:
        t = time.time()
        got = h.search(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else TOKENS)
        print(
            f"search {time.time() - t:.1f}s, ~{line_read.estimate_tokens(got['content'])} tokens",
            file=sys.stderr,
        )
        print(got["content"])
        print("\n" + json.dumps(got["files"], ensure_ascii=False, indent=1))
    finally:
        h.close()
