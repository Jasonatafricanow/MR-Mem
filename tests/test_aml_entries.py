"""Entries placement: the model writes per-message entries, code assembles the
SemanticBlocks and links (mr_mem_aml/entries.py)."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mr_mem_aml import entries as em
from mr_mem_aml.compiler import SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime
from mr_mem_aml.source_store import RawMessage


def mat(*msgs, steps=()):
    """msgs: (id, role, [piece texts]); steps: (ref, text)."""
    return {
        "STEPS": [{"ref": r, "text": t, "state": "happened"} for r, t in steps],
        "rejected": [], "CONTEXT": [],
        "NEW": [{"id": mid, "role": role,
                 "pieces": [{"id": f"{mid}.{j}", "text": t} for j, t in enumerate(texts, 1)]}
                for mid, role, texts in msgs],
    }


def entry(msg, pieces, says, *uses, did=None, happened=True, settles=()):
    e = {"msg": msg, "pieces": list(pieces), "says": says, "happened": happened,
         "uses": [{"to": to, "quote": quote} for to, quote in uses], "settles": list(settles)}
    if did is not None:
        e["did"] = did
    return e


def shape(result):
    return [(b["pieces"], b["after"]) for b in result["blocks"]]


REPORT = ("m1", "assistant", ["**1. 缓存**", "缓存已经加上，导入从 9 分钟降到 2 分钟。",
                              "**2. 导出**", "导出做完了，CSV 格式没有改。"])
IMPORT = [("N1", "用户要求加快导入，并保持 CSV 格式。")]


# --- assembly, work scene ---------------------------------------------------

def test_several_points_each_pair_with_the_section_they_answer():
    material = mat(REPORT, ("m2", "user", ["缓存很好，就这样。", "导出的格式还是不对。"]),
                   steps=IMPORT)
    result = em.assemble(material, [
        entry("m1", ["m1.1-m1.2"], "助手汇报缓存已加上。", ("N1", "加快导入"), did=True),
        entry("m1", ["m1.3", "m1.4"], "助手汇报导出已做完。", ("N1", "保持 CSV 格式"), did=True),
        entry("m2", ["m2.1"], "用户认可缓存。", ("m1.2", "导入从 9 分钟降到 2 分钟")),
        entry("m2", ["m2.2"], "用户指出导出格式不对。", ("m1.4", "CSV 格式没有改")),
    ], "work")
    assert shape(result) == [(["m1.1", "m1.2", "m2.1"], ["N1"]),
                             (["m1.3", "m1.4", "m2.2"], ["N1"])]  # the pairing is no link
    assert result["blocks"][0]["content"] == "助手汇报缓存已加上。\n用户认可缓存。"
    assert result["blocks"][1]["sources"] == ["m1", "m2"]


THREE = ("m1", "assistant", ["**1. 缓存**", "缓存已经加上。", "**2. 导出**", "导出做完了。",
                             "**3. 日志**", "日志改成按天切分了。"])
DECISIONS = [("N1", "用户决定加快导入、整理导出和日志。"), ("N2", "用户决定日志按天切分。")]
THREE_FEEDBACK = ("m2", "user", ["缓存可以。", "导出格式不对。", "日志可以。"])


def three_feedback():
    return [entry("m2", ["m2.1"], "用户认可缓存。", ("m1.2", "缓存已经加上")),
            entry("m2", ["m2.2"], "用户指出导出格式不对。", ("m1.4", "导出做完了")),
            entry("m2", ["m2.3"], "用户认可日志。", ("m1.6", "按天切分"))]


def test_a_report_cut_in_three_shares_the_source_only_its_first_part_names():
    material = mat(THREE, THREE_FEEDBACK, steps=DECISIONS)
    raw = [entry("m1", ["m1.1-m1.2"], "助手汇报缓存已加上。", ("N1", "加快导入"), did=True),
           entry("m1", ["m1.3-m1.4"], "助手汇报导出已做完。", did=True),
           entry("m1", ["m1.5-m1.6"], "助手汇报日志按天切分。", did=True), *three_feedback()]
    shared = em.assemble(material, raw, "work")
    # the decision the report carries out, never "the block before"
    assert [b["after"] for b in shared["blocks"]] == [["N1"], ["N1"], ["N1"]]
    assert shared["audit"]["shared_report_links"] == [
        {"block": "b2", "msg": "m1", "to": ["N1"]}, {"block": "b3", "msg": "m1", "to": ["N1"]}]
    alone = em.assemble(material, raw, "work", share_report_links=False)
    assert [b["after"] for b in alone["blocks"]] == [["N1"], [], []]


def test_a_part_naming_its_own_source_takes_no_shared_one():
    material = mat(THREE, THREE_FEEDBACK, steps=DECISIONS)
    raw = [entry("m1", ["m1.1-m1.2"], "助手汇报缓存已加上。", ("N1", "加快导入"), did=True),
           entry("m1", ["m1.3-m1.4"], "助手汇报导出已做完。", did=True),
           entry("m1", ["m1.5-m1.6"], "助手汇报日志按天切分。", ("N2", "日志按天切分"), did=True),
           *three_feedback()]
    result = em.assemble(material, raw, "work")
    assert [b["after"] for b in result["blocks"]] == [["N1"], ["N1", "N2"], ["N2"]]
    assert [x["block"] for x in result["audit"]["shared_report_links"]] == ["b2"]


def test_sources_are_never_borrowed_from_another_message():
    material = mat(("m1", "assistant", ["**1. 缓存**", "缓存已经加上。"]),
                   ("m2", "assistant", ["**2. 导出**", "导出做完了。"]),
                   ("m3", "user", ["缓存可以。", "导出格式不对。"]), steps=DECISIONS)
    result = em.assemble(material, [
        entry("m1", ["m1.1-m1.2"], "助手汇报缓存已加上。", ("N1", "加快导入"), did=True),
        entry("m2", ["m2.1-m2.2"], "助手汇报导出已做完。", did=True),  # another message
        entry("m3", ["m3.1"], "用户认可缓存。", ("m1.2", "缓存已经加上")),
        entry("m3", ["m3.2"], "用户指出导出格式不对。", ("m2.2", "导出做完了")),
    ], "work")
    assert shape(result) == [(["m1.1", "m1.2", "m3.1"], ["N1"]),
                             (["m2.1", "m2.2", "m3.2"], [])]
    assert result["audit"]["shared_report_links"] == []


def test_one_feedback_takes_the_whole_report():
    material = mat(REPORT, ("m2", "user", ["两点都可以，继续。"]))
    result = em.assemble(material, [
        entry("m1", ["m1.1", "m1.2"], "助手汇报缓存已加上。", did=True),
        entry("m1", ["m1.3", "m1.4"], "助手汇报导出已做完。", did=True),
        entry("m2", ["m2.1"], "用户认可。", ("m1.2", "导入从 9 分钟")),
    ], "work")
    assert shape(result) == [(["m1.1", "m1.2", "m1.3", "m1.4", "m2.1"], [])]


def test_a_late_report_of_other_work_stands_on_its_own():
    material = mat(("m1", "assistant", ["P5 的测试跑完了，全部通过。"]),
                   ("m2", "assistant", ["另外，M5 的后台任务也跑完了，生成了 3 个文件。"]),
                   ("m3", "user", ["P5 没问题，合并吧。"]),
                   steps=[("N1", "用户要求跑 P5 测试。"),
                          ("N2", "用户决定把 M5 后台任务也跑一遍。")])
    result = em.assemble(material, [
        entry("m1", ["m1.1"], "助手汇报 P5 测试全部通过。", ("N1", "跑 P5 测试"), did=True),
        entry("m2", ["m2.1"], "助手汇报 M5 后台任务跑完。", ("N2", "M5 后台任务"), did=True),
        entry("m3", ["m3.1"], "用户决定合并 P5。", ("m1.1", "P5 的测试跑完了")),
    ], "work")
    assert shape(result) == [(["m1.1", "m3.1"], ["N1"]), (["m2.1"], ["N2"])]


def test_an_unanswered_proposal_is_dropped_and_its_links_pass_to_its_message():
    material = mat(("m1", "assistant", ["基于上一轮的结论，我做了两件事。", "**1. 切分**",
                                        "切分改好了，测试通过。", "**2. 承接**",
                                        "承接加了引文校验。", "**3. 下一步**",
                                        "建议接下来做读出。"]),
                   ("m2", "user", ["切分可以。", "承接的阈值太松。"]),
                   steps=[("N1", "用户决定先修切分和承接。")])
    result = em.assemble(material, [
        entry("m1", ["m1.1"], "助手说明基于上一轮结论。", ("N1", "先修切分"), did=False),
        entry("m1", ["m1.2-m1.3"], "助手汇报切分改好。", did=True),
        entry("m1", ["m1.4-m1.5"], "助手汇报承接加了引文校验。", did=True),
        entry("m1", ["m1.6-m1.7"], "助手建议接下来做读出。", did=False, happened=False),
        entry("m2", ["m2.1"], "用户认可切分。", ("m1.3", "切分改好了")),
        entry("m2", ["m2.2"], "用户认为阈值太松。", ("m1.5", "引文校验")),
    ], "work")
    assert shape(result) == [(["m1.2", "m1.3", "m2.1"], ["N1"]),
                             (["m1.4", "m1.5", "m2.2"], ["N1"])]
    audit = result["audit"]
    assert [d["pieces"] for d in audit["dropped"]] == [["m1.1"], ["m1.6", "m1.7"]]
    assert audit["inherited"] == [{"msg": "m1", "pieces": ["m1.1"], "to": ["b1", "b2"],
                                   "via": "message"}]


def test_an_unanswered_report_is_a_block_and_an_unanswered_proposal_is_not():
    material = mat(("m1", "user", ["把导入加快。"]),
                   ("m2", "assistant", ["导入降到 2 分钟了。", "要不要顺便改导出？"]),
                   ("m3", "user", ["另外，周五前要交报告。"]))
    result = em.assemble(material, [
        entry("m1", ["m1.1"], "用户要求加快导入。", happened=False),
        entry("m2", ["m2.1"], "助手汇报导入降到 2 分钟。", ("m1.1", "把导入加快")),  # no "did"
        entry("m2", ["m2.2"], "助手提议改导出。", did=False, happened=False),
        entry("m3", ["m3.1"], "用户说周五前要交报告。", happened=False),
    ], "work")
    assert shape(result) == [(["m1.1"], []), (["m2.1"], ["b1"]), (["m3.1"], [])]
    assert result["audit"]["dropped"][0]["pieces"] == ["m2.2"]


def test_quotes_are_checked_and_may_redirect_a_link():
    material = mat(("m1", "user", ["软上限改成 60 吧。"]), ("m2", "user", ["思考模式还是要关。"]),
                   ("m3", "user", ["这个数也不对。"]), ("m4", "user", ["40 条太少。"]),
                   steps=[("N1", "用户要求把窗口软上限设为 40 条。"),
                          ("N2", "用户决定编译时关掉思考模式。")])
    result = em.assemble(material, [
        entry("m1", ["m1.1"], "a", ("N1", "软上限设为 40 条")),
        entry("m2", ["m2.1"], "b", ("N1", "编译时关掉思考模式")),  # cites N1, quotes N2
        entry("m3", ["m3.1"], "c", ("N2", "一句完全不相干的话")),  # found nowhere
        entry("m4", ["m4.1"], "d", ("N1", "窗口软上限设为 40 个")),  # a small slip
    ], "work")
    assert [b["after"] for b in result["blocks"]] == [["N1"], ["N2"], [], ["N1"]]
    audit = result["audit"]
    assert audit["redirected"] == [{"from": "b2", "cited": "N1", "to": ["N2"],
                                    "quote": "编译时关掉思考模式"}]
    assert [(u["block"], u["to"]) for u in audit["unresolved"]] == [("b3", "N2")]


def test_nothing_is_linked_that_the_model_did_not_write():
    material = mat(("m1", "user", ["把导入加快。"]), ("m2", "assistant", ["导入降到 2 分钟了。"]),
                   ("m3", "user", ["很好。"]), ("m4", "assistant", ["导出也改了。"]),
                   ("m5", "user", ["下一步做读出。"]))
    result = em.assemble(material, [
        entry("m1", ["m1.1"], "用户要求加快导入。", ("m5.1", "下一步做读出")),  # a later piece
        entry("m2", ["m2.1"], "助手汇报导入降到 2 分钟。", did=True),
        entry("m3", ["m3.1"], "用户认可。"),
        entry("m4", ["m4.1"], "助手汇报导出改了。", did=True),
        entry("m5", ["m5.1"], "用户要求做读出。", happened=False),
    ], "work")
    assert [b["after"] for b in result["blocks"]] == [[]] * len(result["blocks"])
    assert result["audit"]["not_earlier"] == [{"from": "b1", "to": "b5", "via": "m5.1"}]


def test_a_trailing_unanswered_run_is_carried_only_when_asked():
    material = mat(("m1", "user", ["把导入加快。"]), ("m2", "assistant", ["导入降到 2 分钟了。"]))
    raw = [entry("m1", ["m1.1"], "用户要求加快导入。"),
           entry("m2", ["m2.1"], "助手汇报导入降到 2 分钟。", ("m1.1", "把导入加快"), did=True)]
    carried = em.assemble(material, raw, "work", carry=True)
    assert shape(carried) == [(["m1.1"], [])] and carried["carry_from"] == "m2"
    kept = em.assemble(material, raw, "work", carry=False)
    assert shape(kept) == [(["m1.1"], []), (["m2.1"], ["b1"])] and kept["carry_from"] is None


def test_bad_entries_are_audited_and_ranges_expand():
    material = mat(("m1", "user", ["一。", "二。", "三。"]))
    result = em.assemble(material, [
        entry("m1", ["m1.1-m1.3"], "用户说了三点。"),
        entry("m9", ["m9.1"], "不存在的消息。"),
        "not an entry",
    ], "work")
    assert shape(result) == [(["m1.1", "m1.2", "m1.3"], [])]
    assert len(result["audit"]["bad_entries"]) == 2
    assert result["audit"]["user_coverage"] == 1.0


# --- assembly, chat scene ---------------------------------------------------

def test_chat_answers_go_with_the_question_and_nothing_is_dropped():
    material = mat(("m1", "user", ["推荐一部电影？"]),
                   ("m2", "assistant", ["推荐《海上钢琴师》。", "另外，周末会下雨。"]))
    raw = [entry("m1", ["m1.1"], "用户请助手推荐电影。"),
           entry("m2", ["m2.1"], "助手推荐《海上钢琴师》。", ("m1.1", "推荐一部电影")),
           entry("m2", ["m2.2"], "助手说周末会下雨。")]
    result = em.assemble(material, raw, "chat")
    # an answer is not a report: the aside does not take the answer's source
    assert shape(result) == [(["m1.1", "m2.1"], []), (["m2.2"], [])]
    assert result["audit"]["dropped"] == [] and result["audit"]["shared_report_links"] == []


# --- input --------------------------------------------------------------------

def test_input_drops_notices_keeps_real_context_and_cuts_pieces():
    notice = "<task-notification>\n<task-id>b1</task-id>\n</task-notification>"
    context = [{"role": "user", "text": f"c{k}"} for k in range(8)] + [
        {"role": "system", "text": notice}, {"role": "assistant", "text": "No response requested."}]
    new = [{"id": "m1", "role": "system", "text": notice},
           {"id": "m2", "role": "assistant", "text": "No response requested."},
           {"id": "m3", "role": "user", "text": "第一句。第二句？\n第三行"},
           {"id": "m4", "role": "assistant",
            "text": "做完了。\n\n## 结果\n- 一\n- 二\n\n| a | b |\n|---|---|"}]
    steps = [{"ref": "N1", "text": "同一步", "state": "happened"},
             {"ref": "N2", "text": "同一步 ", "state": "happened"},
             {"ref": "N3", "text": "同一步", "state": "in progress"}]
    material = em.material(new, context, steps, [])
    assert [m["text"] for m in material["CONTEXT"]] == [f"c{k}" for k in range(2, 8)]
    assert [s["ref"] for s in material["STEPS"]] == ["N1", "N3"]
    assert [(m["id"], [p["id"] for p in m["pieces"]]) for m in material["NEW"]] == [
        ("m3", ["m3.1", "m3.2", "m3.3"]), ("m4", ["m4.1", "m4.2", "m4.3", "m4.4", "m4.5"])]
    assert [p["text"] for p in material["NEW"][1]["pieces"]][1:] == [
        "## 结果", "- 一", "- 二", "| a | b |\n|---|---|"]


def test_the_language_of_says_is_decided_by_code():
    assert em.language_of(["把 Thread 和 Path A 的 SB 先编一遍，看看 v7 的结果。"]) == "Chinese"
    assert em.language_of(["I moved to Lisbon in May and my knee hurts."]) == "English"
    assert em.language_of(["1234 !!"]) is None
    material = em.material([{"id": "m1", "role": "user", "text": "先编一遍再说。"}], [], [], [])
    assert "in Chinese, the language of the conversation." in em.prompt(material)


# --- compiler and runtime ---------------------------------------------------

def message(i, role, text, minute=0):
    return RawMessage(f"rec{i}", "u", "s", "r", i, role, text,
                      datetime(2026, 10, 14, 12, minute, tzinfo=UTC), f"f{i}")


class EntryBackend:
    """One entry per NEW message; a message uses the first piece of the NEW
    message just before it (what a model would write, not what code assumes)."""

    def __init__(self):
        self.calls: list[dict] = []

    def complete(self, prompt: str) -> str:
        material = json.loads(prompt.split("\nINPUT:\n", 1)[1])
        self.calls.append(material)
        out, prev = [], None
        for m in material["NEW"]:
            first = m["pieces"][0]
            e = {"msg": m["id"], "pieces": [f"{first['id']}-{m['pieces'][-1]['id']}"],
                 "says": f"{m['role']} said {first['text']}", "happened": True,
                 "uses": [] if prev is None else [{"to": prev["id"], "quote": prev["text"]}]}
            if m["role"] == "assistant":
                e["did"] = True
            out.append(e)
            prev = first
        return json.dumps({"entries": out}, ensure_ascii=False)


def test_the_compiler_returns_points_blocks_and_carries_an_unanswered_tail():
    backend = EntryBackend()
    compiler = SemanticCompiler(backend, schema="entries", scene="work")
    new = (message(1, "user", "请把导入加快。"), message(2, "system", "<task-notification>x"),
           message(3, "assistant", "导入降到 2 分钟了。"), message(4, "user", "很好。"),
           message(5, "assistant", "接着改导出。"))
    trace: dict = {}
    blocks, kept = compiler.compile(new, (), {"steps": [], "rejected": []}, window=True,
                                    trace=trace)
    assert kept == 4  # m5 waits for the person's answer in the next window
    assert [b.source_record_ids for b in blocks] == [("rec1",), ("rec3", "rec4")]
    assert [(x.after, x.committed) for x in blocks[1].lines] == [("b1", True)]
    assert [m["id"] for m in backend.calls[0]["NEW"]] == ["m1", "m3", "m4", "m5"]
    assert trace["messages"] == {"m1": "rec1", "m3": "rec3", "m4": "rec4", "m5": "rec5"}
    assert trace["audit"]["carried"] == ["m5"] and trace["language"] == "Chinese"
    assert len(trace["entries"]) == 4 and "m3.1" in trace["pieces"]
    assert compiler.version == "mr-mem-aml-entry-compiler-v1"


def test_a_window_of_notices_only_calls_no_model():
    backend = EntryBackend()
    compiler = SemanticCompiler(backend, schema="entries", scene="work")
    assert compiler.compile((message(1, "system", "<task-notification>x"),), (), {},
                            window=True) == ((), 1)
    assert backend.calls == []


def test_entries_need_attributed_assistant_parts():
    with pytest.raises(ValueError):
        SemanticCompiler(None, schema="entries", assistant_policy="context_only")


def runtime(tmp_path: Path, backend) -> MemoryRuntime:
    pytest.importorskip("lce")
    config = RuntimeConfig(
        data_dir=tmp_path / "data", compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree", placement="entries", scene="work",
        compile_window={"soft": 3, "hard": 5},
    )
    return MemoryRuntime(config)


def add(rt, rid, *messages, start_minute=0):
    return rt.add({"request_id": rid, "user_id": "u1", "session_id": "s1", "messages": [
        {"role": role, "content": text,
         "timestamp": 1_790_000_000_000 + (start_minute + i) * 60_000}
        for i, (role, text) in enumerate(messages)]})


def test_runtime_journals_entries_and_reaches_back_past_notices(tmp_path, monkeypatch):
    backend = EntryBackend()
    monkeypatch.setattr("mr_mem_aml.runtime.build_backend", lambda spec: backend)
    rt = runtime(tmp_path, backend)
    notices = [("system", f"<task-notification>{k}</task-notification>") for k in range(25)]
    try:
        add(rt, "r1", ("user", "请把导入加快。"), ("assistant", "导入降到 2 分钟了。"),
            *notices, ("user", "很好，再看导出。"), ("assistant", "导出也改好了。"),
            ("user", "可以。"))
        rt.flush()
        first, second = backend.calls
        # notices do not count toward the caps: one window holds 28 raw messages
        assert [m["id"] for m in first["NEW"]] == ["m1", "m2", "m28"]
        # CONTEXT reaches back past the 25 notices to the real messages
        assert [m["text"] for m in second["CONTEXT"]] == [
            "请把导入加快。", "导入降到 2 分钟了。", "很好，再看导出。"]
        rows = sqlite3.connect(tmp_path / "data" / "aml_source.sqlite").execute(
            "SELECT compiled FROM windows ORDER BY first_seq").fetchall()
        journal = json.loads(rows[0][0])
        assert journal["prompt_version"] == "mr-mem-aml-entry-compiler-v1"
        assert len(journal["assembly"]["entries"]) == 3
        assert [[x["after"] for x in b["lines"]] for b in journal["blocks"]] == [[None], ["b1"]]
        assert len(rt.core.load_all()) == 3
    finally:
        rt.close()
