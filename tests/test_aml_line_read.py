"""Readout by the upward index (mr_mem_aml/line_read.py): raw hits -> SBs -> the Line around them,
plus the raw the chosen SBs point to."""

from datetime import datetime

import pytest

from mr_mem_aml import line_read as lr
from mr_mem_aml.config import RuntimeConfig


def test_two_routes_agreeing_rank_first_and_unknown_records_add_nothing():
    blocks = {"r1": ["a"], "r2": ["b", "a"], "r3": []}
    ranked = lr.rank_blocks(["r1", "r2", "r3", "r9"], lambda r: blocks.get(r, []), ["c", "b"])
    assert ranked[0] == "a"  # two raw hits
    assert ranked[1] == "b"  # one raw hit + the SB search: both routes
    assert set(ranked) == {"a", "b", "c"}


def test_a_hit_message_cut_into_many_sbs_lifts_only_the_closest_one():
    blocks = {"r1": ["a1", "a2", "a3", "a4"], "r2": ["b"]}
    ranked = lr.rank_blocks(["r1", "r2"], lambda r: blocks[r], [], closest=lambda bs: bs[2])
    assert ranked == ["a3", "b"]  # the second hit is second, not fifth


def test_walk_goes_up_and_down_to_the_latest_within_limits():
    parents = {"b": ["a"], "c": ["b"], "d": ["c"], "e": ["d"]}
    children = {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["e"]}
    got = lr.walk("c", lambda x: parents.get(x, []), lambda x: children.get(x, []), up=1, down=1)
    assert got == {"b", "c", "d"}
    assert lr.walk("a", lambda x: [], lambda x: children.get(x, []), up=2, down=6) == {
        "a",
        "b",
        "c",
        "d",
        "e",
    }


def _compose():
    t = datetime(2026, 10, 13, 10, 0)
    steps = {
        "a": lr.Step("定了去重键：内容加时间戳", t, True),
        "b": lr.Step("开始清洗", None, False),
    }
    msgs = {
        "r1": lr.RawMessage("过程话\n去重键改为内容加时间戳", "assistant", t),
        "r2": lr.RawMessage("别的消息", "user", None),
    }
    return lr.compose(
        "去重键",
        ["a"],
        ["r2", "r1"],
        step=steps.__getitem__,
        sources=lambda b: ["r1"] if b == "a" else [],
        parents=lambda b: [],
        children=lambda b: ["b"] if b == "a" else [],
        message=msgs.get,
        lang="zh",
        policy=lr.LineReadPolicy(),
    )


def test_lines_first_then_the_raw_the_chosen_sbs_cite_whole():
    items = _compose()
    assert [i["kind"] for i in items] == ["line", "source"]  # r2: a raw hit no chosen SB points to
    assert "[2026-10-13 10:00 已发生] 定了去重键" in items[0]["content"]
    assert "[未确认] 开始清洗" in items[0]["content"]  # no message time: order only
    assert items[1]["id"] == "raw:r1"  # cited by the chosen SB, before r2
    assert (
        items[1]["content"]
        == "【原文片段】\n[2026-10-13 10:00 助手] 过程话\n去重键改为内容加时间戳"
    )


def test_cited_raw_comes_by_the_index_even_when_the_raw_search_missed_it():
    t = datetime(2026, 10, 13, 10, 0)
    msgs = {"r1": lr.RawMessage("week 1: backend\nweek 2: frontend", "assistant", t)}
    items = lr.compose(
        "q",
        ["a"],
        [],
        step=lambda b: lr.Step("s", t, True),
        sources=lambda b: ["r1"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(),
    )
    assert [i["id"] for i in items] == ["line:a", "raw:r1"]
    assert items[1]["content"].endswith("week 1: backend\nweek 2: frontend")


def test_only_the_paragraphs_an_sb_points_to_are_given():
    t = datetime(2026, 10, 13, 10, 0)
    msgs = {
        "r1": lr.RawMessage(
            "Sure!\n\nWeek 1: backend\n\nWeek 2: frontend\n\nGood luck.", "assistant", t
        )
    }

    def read(span):
        return lr.compose(
            "q",
            ["a", "b"],
            [],
            step=lambda x: lr.Step(x, t, True),
            sources=lambda x: ["r1"],
            parents=lambda x: [],
            children=lambda x: [],
            message=msgs.get,
            lang="en",
            policy=lr.LineReadPolicy(),
            span=span,
        )

    assert read(lambda x, r: {"a": [2], "b": [3]}[x])[-1]["content"].endswith(
        "] Week 1: backend\nWeek 2: frontend"
    )
    assert read(lambda x, r: [2] if x == "a" else None)[-1]["content"].endswith(
        "Good luck."
    )  # b points to all
    assert lr.cut("a\n\nb\n\nc\n\nd", {1, 4}) == "a\n…\nd"


def test_line_read_needs_raw_hits_and_a_line_graph(tmp_path):
    with pytest.raises(ValueError):
        RuntimeConfig(data_dir=tmp_path, compiler={"type": "command", "argv": ["x"]}, line_read={})
    with pytest.raises(ValueError):
        lr.LineReadPolicy.of({"sbs": 0})


def test_times_are_shown_at_the_persons_offset():
    t = datetime(2026, 10, 12, 16, 0)
    items = lr.compose(
        "q",
        ["a"],
        [],
        step=lambda b: lr.Step("s", t, True),
        sources=lambda b: [],
        parents=lambda b: [],
        children=lambda b: [],
        message=lambda r: None,
        lang="zh",
        policy=lr.LineReadPolicy(tz_minutes=480),
    )
    assert "[2026-10-13 00:00 已发生] s" in items[0]["content"]


def test_steps_without_times_follow_the_message_order_and_show_where_they_sit():
    steps = {
        "a": lr.Step("later", None, True, (2, 61)),
        "b": lr.Step("earlier", None, False, (1, 29)),
    }
    items = lr.compose(
        "q",
        ["a"],
        [],
        step=steps.__getitem__,
        sources=lambda b: [],
        parents=lambda b: ["b"] if b == "a" else [],
        children=lambda b: [],
        message=lambda r: None,
        lang="en",
        policy=lr.LineReadPolicy(),
    )
    assert items[0]["content"].split("\n")[1:] == [
        "[S1 #29 not confirmed] earlier",
        "[S2 #61 happened] later",
    ]


def test_raw_full_puts_the_raw_hits_whole_after_the_lines_in_search_order():
    t = datetime(2026, 10, 13, 10, 0)
    msgs = {
        "r1": lr.RawMessage("long " * 200, "assistant", t),
        "r2": lr.RawMessage("second", "user", t),
    }
    items = lr.compose(
        "q",
        ["a"],
        ["r2", "r1"],
        step=lambda b: lr.Step("s", t, True),
        sources=lambda b: ["r1"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(raw_full=2),
    )
    assert [i["id"] for i in items] == ["line:a", "raw:r2", "raw:r1"]  # P1 order, not cited-first
    assert items[2]["content"].endswith("long ")  # whole, no snippet cut


def test_standing_instructions_come_first_whatever_the_question():
    steps = {
        "s": lr.Step("Person: always format code with syntax highlighting", None, True, (1, 55)),
        "a": lr.Step("asks how to add login", None, False, (2, 90)),
    }
    items = lr.compose(
        "show me login",
        ["a"],
        [],
        step=steps.__getitem__,
        sources=lambda b: [],
        parents=lambda b: [],
        children=lambda b: [],
        message=lambda r: None,
        lang="en",
        policy=lr.LineReadPolicy(),
        standing=["s"],
    )
    assert [i["id"] for i in items] == ["standing", "line:a"]
    assert items[0]["content"] == (
        "[Standing instructions and preferences]\n"
        "[S1 #55] Person: always format code with syntax highlighting"
    )


def test_a_part_can_point_to_lines_of_a_code_paragraph():
    text = "Sure!\n\nUse this:\n\n```python\na = 1\nb = 2\nc = 3\n```"
    assert lr.cut(text, {2, (3, 1, 2)}) == "Use this:\na = 1\nb = 2"
    assert lr.cut(text, {2, (3, 2, 3)}) == "Use this:\n…\nb = 2\nc = 3"  # line 1 left out
    assert lr.cut(text, {1, (3, 3, 3)}) == "Sure!\n…\nc = 3"


def test_hits_projection_gives_the_hit_steps_and_only_the_line_between_them():
    parents = {"b": ["a"], "c": ["b"], "d": ["c"], "e": ["d"], "f": ["a"]}
    steps = {
        x: lr.Step(f"{x} said\n- part of {x}", None, True, (1, i)) for i, x in enumerate("abcdef")
    }
    items = lr.compose(
        "q",
        ["e", "b"],
        [],
        step=steps.__getitem__,
        sources=lambda x: [],
        parents=lambda x: parents.get(x, []),
        children=lambda x: [],
        message=lambda r: None,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits", gap=3),
    )
    assert [i["kind"] for i in items] == ["line"]
    assert items[0]["content"].split("\n")[1:] == [
        "[S1 #1 happened] b said",
        "- part of b",
        "[S1 #2 happened] c said",
        "[S1 #3 happened] d said",
        "[S1 #4 happened] e said",
        "- part of e",
    ]  # not a (above), not f (aside)
    far = lr.compose(
        "q",
        ["e", "b"],
        [],
        step=steps.__getitem__,
        sources=lambda x: [],
        parents=lambda x: parents.get(x, []),
        children=lambda x: [],
        message=lambda r: None,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits", gap=1),
    )
    assert len(far) == 2  # too far apart: two pieces, nothing between


def test_relay_adds_the_most_relevant_steps_near_the_anchors():
    parents = {"b": ["a"], "c": ["b"], "d": ["a"], "e": ["d"], "z": []}
    children = {"a": ["b", "d"], "b": ["c"], "d": ["e"]}
    steps = {x: lr.Step(f"{x} said", None, True, (1, i)) for i, x in enumerate("abcdez")}
    score = {"c": 0.9, "e": 0.2, "d": 0.1, "a": 0.0, "z": 1.0}
    items = lr.compose(
        "q",
        ["b"],
        [],
        step=steps.__getitem__,
        sources=lambda x: [],
        parents=lambda x: parents.get(x, []),
        children=lambda x: children.get(x, []),
        message=lambda r: None,
        lang="en",
        relevance=score.get,
        policy=lr.LineReadPolicy(sbs=1, projection="relay", reach=2, extend=2),
    )
    shown = [line.split("] ")[1] for i in items for line in i["content"].split("\n")[1:]]
    assert sorted(shown) == [
        "b said",
        "c said",
        "d said",
    ]  # c, d the most relevant within 2; z not near, e too far


def test_code_seen_before_is_given_as_its_changes():
    t = datetime(2026, 10, 13, 10, 0)
    v1 = "```python\nimport flask\napp = Flask()\nrun(port=5000)\n```"
    v2 = "```python\nimport flask\napp = Flask()\nrun(port=8000)\n```"
    msgs = {"r1": lr.RawMessage(v1, "user", t, (1, 1)), "r2": lr.RawMessage(v2, "user", t, (1, 3))}
    items = lr.compose(
        "q",
        ["a", "b"],
        [],
        step=lambda b: lr.Step(b, t, True, (1, 1 if b == "a" else 3)),
        sources=lambda b: ["r1"] if b == "a" else ["r2"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits", code_diff=True),
    )
    assert items[-2]["content"].endswith(v1)
    assert items[-1]["content"].endswith(
        "```python\n"
        "# changes from S1 #1 (- removed, + added)\n"
        "-run(port=5000)\n"
        "+run(port=8000)\n"
        "```"
    )


def test_the_readout_gives_the_fragments_and_nothing_around_them():
    t = datetime(2026, 10, 13, 10, 0)
    said = "First thing. Second thing that matters. Third thing."
    msgs = {"u": lr.RawMessage(said, "user", t, (1, 1))}
    items = lr.compose(
        "q",
        ["s"],
        [],
        step=lambda b: lr.Step("second", t, True, (1, 1)),
        sources=lambda b: ["u"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits"),
        pointers=lambda b: {"chars": True, "own": {"u": [[13, 39]]}, "parts": []},
    )
    assert items[-1]["content"].endswith("Person] Second thing that matters.")
    assert lr.cut_chars(said, [[0, 12], [40, 52]]) == "First thing.\n…\nThird thing."


def test_a_step_with_its_own_status_is_not_labelled_twice():
    steps = {
        "a": lr.Step(
            "The assistant suggests Redis [unknown; after]", None, True, (1, 3), ((1, 9),)
        ),
        "b": lr.Step("The person moved the deadline [confirmed; before]", None, True, (1, 5)),
        "c": lr.Step("The person plans to deploy [likely; after]", None, False, (1, 7)),
    }
    items = lr.compose(
        "q",
        ["a", "b", "c"],
        [],
        step=steps.__getitem__,
        sources=lambda x: [],
        parents=lambda x: [],
        children=lambda x: [],
        message=lambda r: None,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits"),
    )
    shown = [line for i in items for line in i["content"].split("\n")[1:]]
    assert (
        "[S1 #3] The assistant suggests Redis [unknown; after] (later settled, see S1 #9)" in shown
    )  # by a later step
    assert "[S1 #5] The person moved the deadline [confirmed; before]" in shown
    assert "[S1 #7] The person plans to deploy [likely; after]" in shown


def test_event_projection_gives_the_anchors_matter_with_its_parts_and_the_persons_steps():
    # "ask" has parts "list" (the person's) and "reply" (the assistant's); "next" carries "ask" on,
    # and is a matter of its own with part "detail"; "other" is unrelated
    part_of = {"list": ["ask"], "reply": ["ask"], "detail": ["next"]}
    parts = {"ask": ["list", "reply"], "next": ["detail"]}
    carries = {"ask": ["next"], "next": ["ask"]}
    steps = {
        x: lr.Step(f"{x} said", None, True, (1, i))
        for i, x in enumerate(["ask", "list", "reply", "next", "detail", "other"])
    }
    who = {x: "user" for x in steps} | {"reply": "assistant"}

    def shown(**policy):
        items = lr.compose(
            "q",
            ["list"],
            [],
            step=steps.__getitem__,
            sources=lambda x: [x],
            parents=lambda x: [],
            children=lambda x: [],
            message=lambda r: lr.RawMessage(r, who[r], None),
            lang="en",
            policy=lr.LineReadPolicy(sbs=1, **policy),
            part_of=lambda x: part_of.get(x, []),
            parts=lambda x: parts.get(x, []),
            carries=lambda x: carries.get(x, []),
            everything=lambda: list(steps),
        )
        return [
            line.split("] ")[1]
            for i in items
            if i["kind"] == "line"
            for line in i["content"].split("\n")[1:]
        ]

    assert shown(projection="event") == ["ask said", "list said", "reply said"]
    assert shown(projection="event", extend_from="person") == ["ask said", "list said"]
    assert shown(projection="event", extend_from="person", near_events=1) == [
        "ask said",
        "list said",
        "next said",
        "detail said",
    ]
    assert (
        len(shown(projection="all", extend_from="person")) == 5
    )  # the baseline: every SB of the person


def test_what_the_assistant_said_is_shown_as_told_under_the_persons_step_it_answers():
    # "ask" (the person) -> "advice" (the assistant, two lines, a status of its own) -> "chose" (the
    # person)
    parents = {"advice": ["ask"], "chose": ["advice"]}
    steps = {
        "ask": lr.Step("The person asks which library", None, True, (1, 1)),
        "advice": lr.Step(
            "The assistant recommends Flask-Login [unknown; after]\nwith a config example",
            None,
            False,
            (1, 2),
        ),
        "chose": lr.Step("The person uses Flask-Login [confirmed; before]", None, True, (1, 3)),
    }
    msgs = {
        "ask": lr.RawMessage("which library?", "user", None, (1, 1)),
        "advice": lr.RawMessage("Use Flask-Login: ...", "assistant", None, (1, 2)),
        "chose": lr.RawMessage("I went with Flask-Login", "user", None, (1, 3)),
    }

    def items(assistant):
        return lr.compose(
            "q",
            ["ask", "advice", "chose"],
            [],
            step=steps.__getitem__,
            sources=lambda b: [b],
            parents=lambda b: parents.get(b, []),
            children=lambda b: [],
            message=msgs.get,
            lang="en",
            policy=lr.LineReadPolicy(sbs=3, projection="hits", assistant=assistant),
        )

    told = items("told")
    assert told[0]["content"].split("\n")[1:] == [
        "[S1 #1 happened] The person asks which library",
        # first line, no status
        "  - [S1 #2 Agent told the person] The assistant recommends Flask-Login",
        "[S1 #3] The person uses Flask-Login [confirmed; before]",
    ]
    assert [i["id"] for i in told if i["kind"] == "source"] == [
        "raw:ask",
        "raw:chose",
    ]  # the person's words only
    assert "raw:advice" in [i["id"] for i in items("told_raw")]
    assert (
        "[S1 #2] The assistant recommends Flask-Login [unknown; after]"
        in items("step")[0]["content"]
    )


def test_a_question_on_a_threads_summary_gets_that_stretch_with_its_nearest_steps():
    steps = {
        b: lr.Step(f"The person {b}", None, True, (1, n))
        for n, b in enumerate(["tracker", "queue", "cache", "deploy"], 1)
    }
    known = {
        "t:limits": {
            "summary": "Keeping the API calls under the limits: a tracker, then a cache.",
            "members": ["tracker", "queue", "cache"],
            "last_seq": 3,
        }
    }
    items = lr.compose(
        "which cache?",
        ["deploy"],
        [],
        step=steps.__getitem__,
        sources=lambda b: [],
        parents=lambda b: [],
        children=lambda b: [],
        message=lambda r: None,
        lang="en",
        policy=lr.LineReadPolicy(sbs=1, projection="hits", thread_k=1, thread_sbs=2),
        relevance=lambda b: {"cache": 0.9, "tracker": 0.5, "queue": 0.1}.get(b, 0.0),
        threads=lambda: known,
        thread_hits=["t:limits"],
    )
    stretch = next(i for i in items if i["id"] == "thread:t:limits")["content"].split("\n")
    assert stretch[1] == "Keeping the API calls under the limits: a tracker, then a cache."
    assert [line.split("] ")[1] for line in stretch[2:]] == [
        "The person tracker",
        "The person cache",
    ]  # nearest 2, in order


def test_fusion_fills_in_only_the_raw_hits_no_given_sb_comes_from():
    steps = {"plan": lr.Step("The person plans the trip", None, True, (1, 1))}
    msgs = {
        r: lr.RawMessage(f"text of {r}", "user", None, (1, n))
        for n, r in enumerate(["m1", "m2", "m3"], 1)
    }
    items = lr.compose(
        "q",
        ["plan"],
        ["m1", "m2", "m3"],
        step=steps.__getitem__,
        sources=lambda b: ["m1"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(sbs=1, projection="hits", raw_fill=1),
    )
    filled = [i["id"] for i in items if i["kind"] == "source" and "Other raw" in i["content"]]
    assert filled == ["raw:m2"]  # m1 is covered by the SB; one fill only


def test_p1s_whole_list_backs_the_lines_up_not_only_the_anchor_hits():
    steps = {"plan": lr.Step("The person plans the trip", None, True, (1, 1))}
    names = ["m1", "m2", "m3", "m4"]
    msgs = {r: lr.RawMessage(f"text of {r}", "user", None, (1, n)) for n, r in enumerate(names, 1)}
    items = lr.compose(
        "q",
        ["plan"],
        ["m1"],
        fill_records=names,
        step=steps.__getitem__,
        sources=lambda b: ["m1"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(sbs=1, projection="hits", raw_fill=100),
    )
    filled = [i["id"] for i in items if i["kind"] == "source" and i["id"] != "raw:m1"]
    assert filled == ["raw:m2", "raw:m3", "raw:m4"]  # beyond the one anchor hit, in P1's order


def test_uncompiled_raw_hit_survives_without_fill():
    msgs = {
        r: lr.RawMessage(r, "user", None, (1, n))
        for n, r in enumerate(["compiled", "uncompiled"], 1)
    }
    items = lr.compose(
        "q",
        ["sb"],
        list(msgs),
        uncompiled_records=["uncompiled"],
        step=lambda b: lr.Step("A compiled topic", None, True, (1, 1)),
        sources=lambda b: ["compiled"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits", raw_fill=0),
    )
    assert [i["id"] for i in items if i["kind"] == "source"] == ["raw:compiled", "raw:uncompiled"]
    empty = lr.compose(
        "q",
        [],
        ["uncompiled"],
        uncompiled_records=["uncompiled"],
        step=lambda b: None,
        sources=lambda b: [],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(raw_fill=0),
    )
    assert [i["id"] for i in empty] == ["raw:uncompiled"]


def test_past_top_k_the_furthest_stretch_summaries_go_not_the_raw_hits():
    from mr_mem_aml.runtime import _within

    items = [
        {"id": "standing"},
        *({"id": f"thread:t{n}"} for n in range(60)),
        *({"id": f"line:{n}"} for n in range(20)),
        *({"id": f"raw:r{n}"} for n in range(30)),
    ]
    kept = _within(items, 99)
    ids = [i["id"] for i in kept]
    assert len(kept) == 99 and all(f"raw:r{n}" in ids for n in range(30))
    assert "thread:t0" in ids and "thread:t59" not in ids
    assert _within(items[:50], 99) == items[:50]


def test_raw_units_get_their_vector_when_a_failed_add_is_retried():
    import numpy as np

    from mr_mem_aml.raw_arm import RawArm, RawUnit

    class Flaky:
        fail = True

        def embed_many(self, texts):
            if self.fail:
                raise RuntimeError("embedding service down")
            return [[1.0, 0.0]] * len(texts)

    arm, at = RawArm(Flaky()), __import__("datetime").datetime(2026, 1, 1)
    units = [RawUnit("u1", "a", "hello there", "r1", at), RawUnit("u2", "a", "bye now", "r2", at)]
    try:
        arm.add(units)
    except RuntimeError:
        pass
    arm.embedding.fail = False
    arm.add(units)  # the retried Add
    index = arm._users["a"]
    assert index.ids == ["u1", "u2"] and np.asarray(index.vectors).shape == (2, 2)


def test_embeddings_are_batched_in_order_and_cached(tmp_path):
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from mr_mem_aml.embedding import OpenAICompatibleEmbedding

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            texts = body["input"] if isinstance(body["input"], list) else [body["input"]]
            seen.append(len(texts))
            rows = [{"index": i, "embedding": [float(len(t)), 1.0]} for i, t in enumerate(texts)]
            data = json.dumps({"data": list(reversed(rows))}).encode()  # out of order on purpose
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        emb = OpenAICompatibleEmbedding(
            base_url=f"http://127.0.0.1:{server.server_port}",
            model="m",
            dimension=2,
            cache_path=tmp_path / "c.sqlite",
            batch=10,
            workers=3,
        )
        texts = [("x" * n) for n in range(1, 26)]
        assert [v[0] for v in emb.embed_many(texts)] == [float(n) for n in range(1, 26)]
        assert sorted(seen) == [5, 10, 10]  # 25 texts in three requests
        emb.embed_many(texts)
        assert len(seen) == 3  # all from the cache
    finally:
        server.shutdown()


def test_nearest_paragraphs_keep_the_ones_with_the_questions_rare_words_in_order():
    from mr_mem_aml.line_read import _nearest_paragraphs

    paras = [f"common filler paragraph number {n}" for n in range(6)]
    paras[1] = "common filler about the deadline"
    paras[4] = "common filler with the budget of forty units"
    got = _nearest_paragraphs("\n\n".join(paras), "What budget and deadline did we set?", 2)
    assert got.split("\n\n") == ["…", paras[1], "…", paras[4], "…"]
    # nothing of the question in it: the message's opening, not an arbitrary paragraph
    assert _nearest_paragraphs("\n\n".join(paras), "zebra?", 2).split("\n\n")[0] == paras[0]


def test_a_raw_hit_sits_under_its_sb_with_its_time_and_the_words_do_not_change():
    t = datetime(2026, 10, 13, 10, 0)
    msgs = {
        "said": lr.RawMessage("I moved the launch to May 3", "user", t, (1, 1)),
        "answer": lr.RawMessage("Noted, May 3 it is", "assistant", t, (1, 2)),
        "other": lr.RawMessage("Unrelated words", "user", t, (1, 3)),
        "bare": lr.RawMessage("Not compiled yet", "user", t, (1, 4)),
    }
    steps = {
        "person": lr.Step("The person moves the launch", t, True, (1, 1)),
        "told": lr.Step("The assistant acknowledges", t, True, (1, 2)),
        "far": lr.Step("The person talks of something else", t, True, (1, 3)),
    }
    src = {"person": ["said", "answer"], "told": ["answer"], "far": ["other"]}

    def read(**policy):
        return lr.compose(
            "launch",
            ["person", "far", "told"],
            ["said", "answer", "other", "bare"],
            uncompiled_records=["bare"],
            step=steps.__getitem__,
            sources=lambda b: src[b],
            parents=lambda b: ["person"] if b == "told" else [],
            children=lambda b: ["told"] if b == "person" else [],
            message=msgs.get,
            lang="en",
            policy=lr.LineReadPolicy(
                projection="hits", assistant="told", raw_of_sb="hits", raw_fill=0, **policy
            ),
            relevance=lambda b: {"person": 2.0, "told": 3.0, "far": 1.0}[b],
            blocks_of=lambda r: [b for b, s in src.items() if r in s],
        )

    apart, under = read(), read(raw_under_sb=True)
    assert [i["id"] for i in apart if i["kind"] == "source"] == [
        "raw:said",
        "raw:answer",
        "raw:other",
        "raw:bare",
    ]
    assert [i["id"] for i in under if i["kind"] == "source"] == ["raw:bare"]
    text = next(i["content"] for i in under if i["id"] == "line:person")
    lines = text.split("\n")
    at = next(n for n, x in enumerate(lines) if "moves the launch" in x)
    # the person's step holds them, the closer "told" line does not, and the told line comes after
    assert "10-13 10:00" in lines[at + 1] and lines[at + 1].endswith("I moved the launch to May 3")
    assert lines[at + 2].endswith("Noted, May 3 it is")
    assert "acknowledges" in lines[at + 3]
    assert "Unrelated words" in next(i["content"] for i in under if i["id"] == "line:far")
    for words in ("May 3 it is", "launch to May 3", "Unrelated words", "Not compiled yet"):
        assert sum(words in i["content"] for i in apart) == 1
        assert sum(words in i["content"] for i in under) == 1


def test_raw_under_sb_needs_a_projection_that_shows_steps():
    with pytest.raises(ValueError):
        lr.LineReadPolicy.of({"raw_under_sb": True, "projection": "event"})


def test_carried_words_sit_under_their_step_and_the_whole_readout_keeps_its_budget():
    t = datetime(2026, 10, 13, 10, 0)
    msgs = {
        "hit": lr.RawMessage("The hit", "user", t, (1, 2)),
        "near": lr.RawMessage("Words of the near step", "user", t, (1, 1)),
        "far": lr.RawMessage("Words of the far step " * 40, "user", t, (1, 3)),
    }
    steps = {
        "near": lr.Step("The person does the near thing", t, True, (1, 1)),
        "far": lr.Step("The person does the far thing", t, True, (1, 3)),
    }
    src = {"near": ["near", "hit"], "far": ["far"]}

    def read(**policy):
        return lr.compose_within(
            "q",
            ["near", "far"],
            ["hit"],
            step=steps.__getitem__,
            sources=lambda b: src[b],
            parents=lambda b: [],
            children=lambda b: [],
            message=msgs.get,
            lang="en",
            policy=lr.LineReadPolicy(
                projection="hits",
                assistant="told",
                raw_of_sb="hits",
                raw_fill=0,
                raw_under_sb=True,
                sb_raw=True,
                **policy,
            ),
            relevance=lambda b: 1.0,
            blocks_of=lambda r: [b for b, s in src.items() if r in s],
        )

    text = "\n".join(i["content"] for i in read())
    assert lr.CARRIED["en"] not in text
    lines = text.split("\n")
    at = next(n for n, x in enumerate(lines) if "near thing" in x)
    assert lines[at + 1].endswith("Words of the near step")  # message order under the step
    assert lines[at + 2].endswith("The hit")
    joined = "\n".join(i["content"] for i in read(readout_tokens=60))
    assert "Words of the near step" in joined and "The hit" in joined
    assert "Words of the far step" not in joined  # the rest of the readout is measured first


def test_a_session_date_is_written_once_in_a_readout():
    day = datetime(2024, 4, 5)
    msgs = {r: lr.RawMessage(r, "user", None, (3, n), day) for n, r in enumerate(["r1", "r2"], 1)}
    steps = {"a": lr.Step("The person does a", None, True, (3, 1), session=day)}

    def read(**policy):
        return lr.compose(
            "q",
            ["a"],
            ["r1", "r2"],
            step=steps.__getitem__,
            sources=lambda b: ["r1", "r2"],
            parents=lambda b: [],
            children=lambda b: [],
            message=msgs.get,
            lang="en",
            policy=lr.LineReadPolicy(projection="hits", raw_of_sb="hits", raw_fill=0, **policy),
        )

    count = lambda items: sum(i["content"].count("session began") for i in items)  # noqa: E731
    assert count(read()) == 3
    assert count(read(session_once=True)) == 1


def test_a_fill_gives_every_message_the_lines_do_not_hold_when_the_pool_is_the_whole_history():
    msgs = {f"r{n}": lr.RawMessage(f"message {n}", "assistant", None, (1, n)) for n in range(40)}
    items = lr.compose(
        "q",
        ["sb"],
        list(msgs)[:30],
        fill_records=list(msgs),
        step=lambda b: lr.Step("The person pastes a document", None, True, (1, 0)),
        sources=lambda b: ["r0"],
        parents=lambda b: [],
        children=lambda b: [],
        message=msgs.get,
        lang="en",
        policy=lr.LineReadPolicy(projection="hits", assistant="told", raw_fill=2000),
    )
    assert sum(any(f"message {n}" in i["content"] for i in items) for n in range(40)) == 40
