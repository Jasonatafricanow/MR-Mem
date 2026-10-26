"""Readout by the upward index (mr_mem_aml/line_read.py): raw hits -> SBs -> the Line around them,
plus raw snippets, within one budget."""

from datetime import datetime

import pytest

from mr_mem_aml import line_read as lr
from mr_mem_aml.config import RuntimeConfig


def test_two_routes_agreeing_rank_first_and_unknown_records_add_nothing():
    blocks = {"r1": ["a"], "r2": ["b", "a"], "r3": []}
    ranked = lr.rank_blocks(["r1", "r2", "r3", "r9"], lambda r: blocks.get(r, []), ["c", "b"])
    assert ranked[0] == "a"            # two raw hits
    assert ranked[1] == "b"            # one raw hit + the SB search: both routes
    assert set(ranked) == {"a", "b", "c"}


def test_walk_goes_up_and_down_to_the_latest_within_limits():
    parents = {"b": ["a"], "c": ["b"], "d": ["c"], "e": ["d"]}
    children = {"a": ["b"], "b": ["c"], "c": ["d"], "d": ["e"]}
    got = lr.walk("c", lambda x: parents.get(x, []), lambda x: children.get(x, []), up=1, down=1)
    assert got == {"b", "c", "d"}
    assert lr.walk("a", lambda x: [], lambda x: children.get(x, []), up=2, down=6) == {"a", "b", "c", "d", "e"}


def test_snippet_takes_the_paragraph_sharing_most_words_with_the_question():
    text = "先跑一下。\n去重前 173,530 条，去重后 24,716 条。\n其余是过程话。"
    assert lr.snippet(text, "去重前后各多少条", 250).startswith("去重前 173,530")
    assert len(lr.snippet("长" * 400, "长", 250)) == 250


def _compose(budget=2000, line_chars=1200):
    t = datetime(2026, 10, 13, 10, 0)
    steps = {"a": lr.Step("定了去重键：内容加时间戳", t, True), "b": lr.Step("开始清洗", None, False)}
    msgs = {"r1": lr.RawMessage("过程话\n去重键改为内容加时间戳", "assistant", t),
            "r2": lr.RawMessage("别的消息", "user", None)}
    return lr.compose("去重键", ["a"], ["r2", "r1"], step=steps.__getitem__,
                      sources=lambda b: ["r1"] if b == "a" else [],
                      parents=lambda b: [], children=lambda b: ["b"] if b == "a" else [],
                      message=msgs.get, lang="zh",
                      policy=lr.LineReadPolicy(budget=budget, line_chars=line_chars))


def test_lines_first_then_snippets_cited_by_the_chosen_sbs_first():
    items = _compose()
    assert [i["kind"] for i in items] == ["line", "source", "source"]
    assert "[10-13 10:00 已发生] 定了去重键" in items[0]["content"]
    assert "[进行中] 开始清洗" in items[0]["content"]          # no message time: order only
    assert items[1]["id"] == "raw:r1"                        # cited by the chosen SB, before r2
    assert items[1]["content"].startswith("【原文片段】\n[10-13 10:00 助手] 去重键改为")
    assert not items[2]["content"].startswith("【原文片段】")


def test_everything_stays_within_the_budget():
    items = _compose(budget=60, line_chars=40)
    assert sum(len(i["content"]) for i in items) <= 60
    assert items[0]["kind"] == "line" and len(items[0]["content"]) == 40


def test_line_read_needs_raw_hits_and_a_line_graph(tmp_path):
    with pytest.raises(ValueError):
        RuntimeConfig(data_dir=tmp_path, compiler={"type": "command", "argv": ["x"]}, line_read={})
    with pytest.raises(ValueError):
        lr.LineReadPolicy.of({"line_chars": 3000, "budget": 2000})


def test_times_are_shown_at_the_persons_offset():
    t = datetime(2026, 10, 12, 16, 0)
    items = lr.compose("q", ["a"], [], step=lambda b: lr.Step("s", t, True), sources=lambda b: [],
                       parents=lambda b: [], children=lambda b: [], message=lambda r: None, lang="zh",
                       policy=lr.LineReadPolicy(tz_minutes=480))
    assert "[10-13 00:00 已发生] s" in items[0]["content"]
