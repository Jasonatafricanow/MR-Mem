"""Work scene, two-step compile (mr_mem_aml/moves.py): step 1 writes the SBs ("what happened ->
what the person did"), step 2 places them among candidates; the prompt follows the person's language."""

import json
import re
import sqlite3
from pathlib import Path

import pytest

from mr_mem_aml import moves as mv
from mr_mem_aml.compiler import SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime

pytest.importorskip("lce")


# --- language -------------------------------------------------------------------

def test_the_language_follows_the_person_not_the_agent():
    assert mv.person_language(["先暂停，修好再跑", "好了吗？"]) == "zh"
    assert mv.person_language(["Stop first, fix it, then rerun", "done?"]) == "en"
    # a Chinese speaker with English code in the message is still Chinese
    assert mv.person_language(["把 compile_window 的 soft 改成 40，hard 60"]) == "zh"


def test_content_round_trip():
    c = mv.content_of("跑完第 2 页，一个节点装了 12 条", "先暂停，修好再跑", "zh")
    assert mv.split_content(c) == ("跑完第 2 页，一个节点装了 12 条", "先暂停，修好再跑")
    assert mv.split_content("Reran pages 1-3") == ("Reran pages 1-3", "")
    assert mv.split_content(mv.content_of("", "要求先清洗数据", "zh")) == ("", "要求先清洗数据")


# --- step 1 ---------------------------------------------------------------------

def test_step1_one_sb_per_point_and_no_sb_for_an_unanswered_report():
    roles = ["assistant", "user", "assistant", "assistant"]
    data = {"moves": [
        {"msg": "M2", "about": ["M1"], "situation": "缓存加上了，导入 9 分钟降到 2 分钟", "move": "认可", "happened": True},
        {"msg": "M2", "about": ["M1"], "situation": "导出没改 CSV 格式", "move": "要求改格式"},
        {"msg": "M1", "about": [], "situation": "x", "move": "not the person's"},       # dropped
        {"msg": "M9", "about": [], "situation": "x", "move": "unknown id"}],            # dropped
        "reports": [{"msgs": ["M3", "M4"], "situation": "格式改好了", "happened": False}]}  # no SB
    sbs = mv.parse_step1(data, roles)
    assert [(b["sources"], b["move"], b["happened"]) for b in sbs] == [([0, 1], "认可", True), ([0, 1], "要求改格式", True)]


def test_step1_about_only_names_earlier_agent_messages():
    sbs = mv.parse_step1({"moves": [{"msg": "M2", "about": ["M1", "M2", "M3"], "situation": "s",
                                     "move": "m"}]}, ["assistant", "user", "assistant"])
    assert sbs[0]["sources"] == [0, 1]


def test_the_tail_after_the_person_goes_to_the_next_window():
    assert mv.tail_start(["assistant", "user", "assistant", "assistant"]) == 2
    assert mv.tail_start(["assistant", "user"]) is None


def test_step1_slim4_spans_become_sources_with_the_agent_messages():
    roles = ["user", "assistant", "assistant", "user", "assistant"]
    data = [{"id": "b1", "turns": "M1-M3", "summary": "建了 worktree，用户同意", "unresolved": None},
            {"id": "b2", "turns": "M4-M9", "summary": "推送完成", "unresolved": "M4"},   # clipped to M5
            {"id": "b3", "turns": "10-09#46", "summary": "x"},                        # no M span: dropped
            {"id": "b4", "turns": "M2", "summary": ""}]                               # no summary: dropped
    sbs = mv.parse_step1_slim4(data, roles)
    assert [(b["sources"], b["situation"], b["move"], b["unresolved"]) for b in sbs] == [
        ([0, 1, 2], "建了 worktree，用户同意", "", None), ([3, 4], "推送完成", "", 3)]
    assert mv.content_of(sbs[0]["situation"], sbs[0]["move"], "zh") == "建了 worktree，用户同意"


def test_step1_slim4_leaving_most_of_the_person_out_is_asked_again():
    roles = ["user", "assistant", "user", "assistant", "user"]
    with pytest.raises(ValueError):
        mv.check_step1_slim4([{"turns": "M1-M2", "summary": "s"}], roles)
    mv.check_step1_slim4([{"turns": "M1-M3", "summary": "s"}], roles)
    assert mv.extract_json_array('```json\n[{"turns":"M1","summary":"s"}]\n```') == [{"turns": "M1", "summary": "s"}]


# --- step 2 ---------------------------------------------------------------------

def test_step2_links_only_to_candidates_or_earlier_sbs():
    data = {"links": [{"sb": "s2", "to": "N1", "why": "carries out"}, {"sb": "s2", "to": "s1"},
                      {"sb": "s1", "to": "s2"},            # later SB: dropped
                      {"sb": "s1", "to": "N7"},            # no such candidate: dropped
                      {"sb": "s2", "to": "N1"}],           # duplicate: dropped
            "settles": [{"sb": "s2", "step": "N2"}, {"sb": "s3", "step": "N1"}]}
    out = mv.parse_step2(data, n_sbs=2, n_cands=2)
    assert out["after"] == {1: [("N", 0), ("s", 0)]}
    assert out["why"][(1, ("N", 0))] == "carries out"
    assert out["settles"] == {1: [1]}


def test_prompts_follow_the_language():
    zh = mv.step1_prompt("zh", [{"role": "user", "text": "开始吧"}], [])
    en = mv.step1_prompt("en", [{"role": "user", "text": "go ahead"}], [])
    assert "用中文写" in zh and "[M1 人] 开始吧" in zh
    assert "Write in English" in en and "[M1 Person] go ahead" in en
    p2 = mv.step2_prompt("en", [{"text": "a", "time": "t"}], [{"text": "b", "time": "t0", "state": "happened"}], [[0]])
    assert "s1 [t] a\n   candidates: N1" in p2 and "N1 [t0, happened] b" in p2


# --- runtime: two steps per window, journal, replay -------------------------------

class TwoStepBackend:
    """Step 1: one SB per message of the person, about the agent message before it.
    Step 2: every SB after the first builds on the SB just before it (s or the latest N)."""

    def __init__(self):
        self.calls = {"step1": 0, "step2": 0}

    def complete(self, prompt: str) -> str:
        if "SBs of this window" in prompt or "本窗口的 SB" in prompt:
            self.calls["step2"] += 1
            sbs = re.findall(r"^s(\d+) ", prompt, flags=re.M)
            cands = re.findall(r"^N(\d+) ", prompt, flags=re.M)
            links = [{"sb": f"s{k}", "to": f"s{int(k) - 1}", "why": "next"} for k in sbs if int(k) > 1]
            if cands:
                links.append({"sb": "s1", "to": f"N{cands[-1]}", "why": "carries on"})
            return json.dumps({"links": links, "settles": []})
        self.calls["step1"] += 1
        new = re.split(r"NEW[:：]\n", prompt, maxsplit=1)[1]
        rows = re.findall(r"^\[M(\d+)[^\]]* (Person|人|Agent)\] (.*)$", new, flags=re.M)
        moves = [{"msg": f"M{i}", "about": [f"M{int(i) - 1}"] if int(i) > 1 else [],
                  "situation": f"sit {text}", "move": f"move {text}", "happened": True}
                 for i, who, text in rows if who in ("Person", "人")]
        return json.dumps({"moves": moves, "reports": []})


def runtime(tmp_path: Path, backend) -> MemoryRuntime:
    config = RuntimeConfig(
        data_dir=tmp_path / "data", compiler={"type": "command", "argv": ["x"]},
        path_a_engine="tree", compile_window={"soft": 3, "hard": 5}, scene="work", placement="moves",
    )
    return MemoryRuntime(config, compiler=SemanticCompiler(backend, schema="moves", scene="work"))


def add(rt, rid, *texts, start_minute=0):
    messages = [{"role": "assistant" if i % 2 == 0 else "user", "content": t,
                 "timestamp": 1_690_000_000_000 + (start_minute + i) * 60_000}
                for i, t in enumerate(texts)]
    return rt.add({"request_id": rid, "user_id": "u1", "session_id": "s1", "messages": messages})


def edges(tmp_path: Path) -> int:
    path = next((tmp_path / "data" / "lce_tree").rglob("lines.sqlite"))
    return sqlite3.connect(path).execute("SELECT COUNT(*) FROM line_edges").fetchone()[0]


def test_two_steps_per_window_links_on_the_graph_and_journal(tmp_path):
    backend = TwoStepBackend()
    rt = runtime(tmp_path, backend)
    try:
        add(rt, "r1", "报告一", "好，继续", "报告二", "改一下格式", "报告三", "可以")
        rt.flush()
        assert backend.calls["step1"] == backend.calls["step2"] >= 1
        assert len(rt.core.load_all()) == 3            # one SB per point of the person
        assert all("→ 人：move" in m.content for m in rt.core.load_all())  # Chinese prompt
        assert edges(tmp_path) >= 2
        rows = sqlite3.connect(tmp_path / "data" / "aml_source.sqlite").execute(
            "SELECT compiled FROM windows").fetchall()
        placed = [json.loads(r[0])["placement"] for r in rows]
        assert all(p["language"] == "zh" and "refs" in p and "after" in p for p in placed)
    finally:
        rt.close()


def test_a_journaled_window_replays_both_steps_without_a_model_call(tmp_path):
    first = TwoStepBackend()
    rt = runtime(tmp_path / "a", first)
    try:
        add(rt, "r1", "report one", "ok go on", "report two", "fix the format", "report three", "fine")
        rt.flush()
        rows = sqlite3.connect(tmp_path / "a" / "data" / "aml_source.sqlite").execute(
            "SELECT window_id,user_id,first_seq,last_seq,compiled FROM windows ORDER BY first_seq").fetchall()
        assert all(json.loads(r[4])["placement"]["language"] == "en" for r in rows)
    finally:
        rt.close()
    replay = TwoStepBackend()
    rt = runtime(tmp_path / "b", replay)
    try:
        for r in rows:
            rt.source.put_window(r[0], r[1], r[2], r[3], json.loads(r[4]))
        add(rt, "r1", "report one", "ok go on", "report two", "fix the format", "report three", "fine")
        rt.flush()
        assert replay.calls == {"step1": 0, "step2": 0}
        assert edges(tmp_path / "b") == edges(tmp_path / "a")
    finally:
        rt.close()


def test_moves_is_the_work_scene_only():
    with pytest.raises(ValueError):
        SemanticCompiler(TwoStepBackend(), schema="moves", scene="chat")
    with pytest.raises(ValueError):
        RuntimeConfig(data_dir=Path("x"), compiler={"type": "command", "argv": ["x"]},
                      path_a_engine="tree", compile_window={"soft": 3, "hard": 5},
                      scene="chat", placement="moves")


def test_a_step1_answer_missing_most_of_the_person_is_asked_again():
    roles = ["assistant", "user", "assistant", "user", "assistant", "user"]
    one = {"moves": [{"msg": "M6", "about": ["M5"], "situation": "s", "move": "m"}]}
    with pytest.raises(ValueError):
        mv.check_step1(one, roles)
    two = {"moves": [{"msg": "M2", "situation": "s", "move": "m"}, {"msg": "M4", "situation": "s", "move": "m"}]}
    mv.check_step1(two, roles)  # two of three: kept


def test_interleaved_sessions_are_marked_and_about_stays_in_its_session():
    new = [{"role": "assistant", "text": "清洗查完了", "session": "a"},
           {"role": "assistant", "text": "回滚脚本提交了", "session": "b"},
           {"role": "user", "text": "这两个可以同步进行", "session": "a"}]
    prompt = mv.step1_prompt("zh", new, [])
    assert "[M1 会话A Agent]" in prompt and "[M2 会话B Agent]" in prompt
    data = {"moves": [{"msg": "M3", "about": ["M1", "M2"], "situation": "s", "move": "m"}]}
    assert mv.parse_step1(data, ["assistant", "assistant", "user"], ["a", "b", "a"])[0]["sources"] == [0, 2]
