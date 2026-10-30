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


def test_chat_scene_with_slim4_carries_the_unanswered_question_only():
    assert mv.tail_start(["user", "assistant", "user"], "chat") == 2
    assert mv.tail_start(["user", "assistant", "user", "user"], "chat") == 2
    assert mv.tail_start(["user", "assistant"], "chat") is None
    assert mv.tail_start(["user", "user", "user"], "chat") is None        # between people: nothing waits
    assert "Return only a valid JSON array" in mv.step1_prompt_slim4("en", [{"role": "user", "text": "hi"}], [])
    for lang in ("zh", "en"):  # the chat scene has its own template (SB freeze 9-25)
        chat = mv.step1_prompt_slim4(lang, [{"role": "user", "text": "hi"}], [], "chat")
        assert '"status":"unknown"' in chat and mv.SUMMARY[lang] not in chat
        assert mv.SUMMARY[lang] in mv.step1_prompt_slim4(lang, [{"role": "user", "text": "hi"}], [])
    sbs = mv.parse_step1_slim4([{"turns": "M1", "text": "has finished A", "status": "happened"},
                                {"turns": "M1-M2", "text": "asks how to do B; assistant recommends C", "status": "unknown"},
                                {"turns": "M2", "text": "plans D", "status": "not yet"}], ["user", "assistant"])
    assert [(b["situation"][:5], b["happened"]) for b in sbs] == [("has f", True), ("asks ", False), ("plans", False)]
    assert "same matter going on" in mv.step2_prompt("en", [{"text": "s"}], [], [[]], "chat")
    SemanticCompiler(object(), schema="moves", scene="chat", step1="slim4")
    with pytest.raises(ValueError):
        SemanticCompiler(object(), schema="moves", scene="chat")


def test_chat_step2_shows_earlier_steps_under_the_persons_matter():
    cands = [{"text": "wants a sprint plan", "time": "t0", "matter": 0, "who": "person"},
             {"text": "gives a two-week plan", "time": "t1", "matter": 0, "who": "assistant"},
             {"text": "asks about Docker", "time": "t2", "matter": 2, "who": "person"}]
    p2 = mv.step2_prompt("en", [{"text": "sprint 1 is done", "time": "t3"}], cands, [[1]], "chat")
    assert "s1 [t3] sprint 1 is done\n   matters: N1\n" in p2
    assert "N1 (the person) [t0] wants a sprint plan\n   N2 (the assistant) [t1] gives a two-week plan\nN3" in p2


def test_chat_step1_carries_standing_instructions_into_the_journal():
    from mr_mem_aml.compiler import blocks_from_json, blocks_to_json, CompiledBlock

    sb = mv.parse_step1_slim4([{"turns": "M1", "text": "always use version numbers", "status": "happened",
                                "standing": True}], ["user"])[0]
    assert sb["standing"] is True
    block = CompiledBlock("c", ("r",), None, None, standing=True)
    assert blocks_from_json(blocks_to_json((block,)))[0].standing is True
    assert '"standing":false' in mv.step1_prompt_slim4("en", [{"role": "user", "text": "hi"}], [], "chat")


def test_long_messages_are_numbered_by_paragraph_and_turns_may_point_into_them():
    from mr_mem_aml import moves as mv
    long = "Sure!\n\nWeek 1: backend\n\n```python\nx = 1\n\ny = 2\n```\n\nGood luck."
    assert len(mv.paragraphs(long)) == 4 and mv.paragraphs(long)[2].startswith("```python")
    assert mv.paragraphs("one line") == ["one line"]
    assert mv.paragraphs("my code:\n```py\nx = 1\n```\nwhy?") == ["my code:", "```py\nx = 1\n```", "why?"]  # code on its own
    prompt = mv.step1_prompt_slim4("en", [{"role": "user", "text": "plan?"},
                                          {"role": "assistant", "text": long}], [], scene="chat")
    assert "(M2.2) Week 1: backend" in prompt and "(M1.1)" not in prompt
    data = [{"turns": "M1, M2.2-M2.3", "text": "asked for a plan; got week 1"},
            {"turns": "M1-M2", "text": "whole"}, {"turns": "M2.2-3", "text": "same paragraphs"},
            {"turns": "M2.1-M2.4", "text": "all of it"}]
    sbs = mv.parse_step1_slim4(data, ["user", "assistant"], [1, 4])
    got = {b["situation"]: (b["sources"], b["spans"]) for b in sbs}
    assert got == {"asked for a plan; got week 1": ([0, 1], {1: [2, 3]}), "whole": ([0, 1], {}),
                   "same paragraphs": ([1], {1: [2, 3]}), "all of it": ([1], {})}


def test_chat_step1_sbs_and_what_builds_on_them_come_from_one_answer():
    from mr_mem_aml import moves as mv
    answer = "Sure!\n\nWeek 1: backend\n\n```python\na = 1\nb = 2\nc = 3\n```"
    prompt = mv.step1_prompt_slim4("en", [{"role": "user", "text": "plan?"},
                                          {"role": "assistant", "text": answer}], [], scene="chat")
    assert "(M2.3) ```python" in prompt and "  2| b = 2" in prompt   # code lines are numbered
    raw = """ANALYSIS: the person moved the deadline and asked for a plan.
JSON:
[{"turns":"M1","text":"The deadline moved to March 31","when":"before","status":"confirmed","standing":false},
 {"turns":"M1","text":"The person asks for a plan","when":"before","status":"confirmed","next":[
   {"turns":"M2.2","text":"The assistant plans week 1: backend","when":"after","status":"unknown","next":[
     {"turns":"M2.3:2-2","text":"The assistant sets b to 2","when":"after","status":"unknown"}]}]}]"""
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user", "assistant"], [1, 3])
    by = {b["situation"].split(" [")[0]: b for b in sbs}
    assert by["The deadline moved to March 31"]["situation"].endswith("[confirmed; before]")
    assert by["The deadline moved to March 31"]["happened"] is True and by["The deadline moved to March 31"]["under"] is None
    ask, week, b = by["The person asks for a plan"], by["The assistant plans week 1: backend"], by["The assistant sets b to 2"]
    assert sbs[week["under"]] is ask and sbs[b["under"]] is week       # the steps building on the ask
    assert week["sources"] == [1] and week["pointers"]["own"] == {1: [2]} and week["happened"] is False


def test_each_sb_keeps_the_fragment_of_the_words_it_comes_from():
    from mr_mem_aml import moves as mv
    said = ("I'm trying to optimize the dashboard, and I've never actually integrated Flask-Login, "
            "so I'm starting from scratch, and I'd like to use Marshmallow to validate data")
    reply = "Sure.\n\nUse LoginManager.\n\nUse Schema classes."
    raw = """JSON:
[{"turns":"M1","from":"I've never actually integrated","to":"starting from scratch","text":"Never integrated Flask-Login","when":"before","status":"confirmed"},
 {"turns":"M1","from":"I'd like to use Marshmallow","to":"validate data","text":"Wants Marshmallow","when":"after","status":"likely","next":[
   {"turns":"M2.3","from":"Use Schema","to":"classes.","text":"Use Schema classes","when":"after","status":"unknown"}]},
 {"turns":"M1","from":"words that are not there","to":"","text":"Copied wrong","when":"before","status":"confirmed"}]"""
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user", "assistant"], [1, 3], [said, reply])
    by = {b["situation"].split(" [")[0]: b["pointers"] for b in sbs}
    (s, e), = by["Never integrated Flask-Login"]["own"][0]
    assert said[s:e] == "I've never actually integrated Flask-Login, so I'm starting from scratch"
    assert said[slice(*by["Wants Marshmallow"]["own"][0][0])] == "I'd like to use Marshmallow to validate data"
    assert reply[slice(*by["Use Schema classes"]["own"][1][0])] == "Use Schema classes."
    assert by["Copied wrong"]["own"] == {0: None} and by["Copied wrong"]["missed"] == 1   # whole, and counted


def test_the_persons_words_no_sb_covers_are_found():
    from mr_mem_aml import moves as mv
    said = "Thanks. I never used Docker before this. Also please show how to deploy it with Gunicorn on Render tomorrow."
    sbs = [{"pointers": {"own": {0: [[8, 40]]}}}]          # only "I never used Docker before this."
    left = mv.uncovered(sbs, ["user"], [said])
    assert left == [(0, "Also please show how to deploy it with Gunicorn on Render tomorrow.")]   # "Thanks." too short


def test_code_the_person_pastes_goes_with_the_sb_that_brings_it_in():
    from mr_mem_aml import moves as mv
    said = "My login fails. Here's my code:\n\n```python\nlogin()\n```\n\nAlso, should I use bcrypt?"
    raw = ('JSON:\n[{"turns":"M1","from":"My login fails","to":"my code:","text":"The person reports login fails",'
           '"when":"before","status":"confirmed","standing":false},'
           '{"turns":"M1","from":"Also, should","to":"bcrypt?","text":"The person asks about bcrypt",'
           '"when":"before","status":"confirmed","standing":false}]')
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user"], [3], [said])
    code = said.index("```python")
    first, second = (b["pointers"]["own"][0] for b in sbs)
    assert any(s == code for s, _ in first) and not any(s == code for s, _ in second)


def test_an_answer_that_stops_keeps_only_what_it_compiled_and_says_where():
    from mr_mem_aml import moves as mv
    assert mv.stop_at('[...]\nSTOP: M12', 14) == 12
    assert mv.stop_at('[...]', 14) == 14 and mv.stop_at('STOP: M40', 14) == 14   # none, or past the end: all
    sbs = [{"sources": [0], "under": None}, {"sources": [1], "under": 0},
           {"sources": [2], "under": 1}, {"sources": [1, 2], "under": 0}]
    kept = mv.compiled_before(sbs, 2)
    assert [b["sources"] for b in kept] == [[0], [1]] and kept[1]["under"] == 0


def test_what_the_assistant_says_is_not_confirmed_by_saying_it():
    from mr_mem_aml import moves as mv
    sbs = mv.parse_step1_chat([{"turns": "M1", "text": "The person measured 150ms", "when": "before", "status": "confirmed",
                                "next": [{"turns": "M2", "text": "The assistant says the cache made it faster",
                                          "when": "before", "status": "confirmed"}]}], ["user", "assistant"])
    assert [(b["situation"].rsplit(" [", 1)[1], b["happened"]) for b in sbs] == [("confirmed; before]", True), ("unknown; before]", False)]


def test_words_that_come_more_than_once_give_the_shortest_span():
    from mr_mem_aml import moves as mv
    said = "please show an example of A, and please show an example of B, and please show an example of C"
    s, e = mv.anchor(said, "please show an example", "of B")
    assert said[s:e] == "please show an example of B"


def test_a_need_keeps_its_own_words_and_its_things_keep_theirs():
    from mr_mem_aml import moves as mv
    said = "Help me add sessions with Flask-Login, and use Marshmallow to validate, and use Flask-Cors for CORS"
    raw = """JSON:
[{"turns":"M1","from":"Help me add sessions","to":"for CORS","text":"Wants sessions","when":"before","status":"confirmed","next":[
   {"turns":"M1","from":"use Marshmallow","to":"to validate","text":"Wants Marshmallow","when":"after","status":"likely"},
   {"turns":"M1","from":"use Flask-Cors","to":"for CORS","text":"Wants Flask-Cors","when":"after","status":"likely"}]}]"""
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user"], [1], [said])
    need = next(b for b in sbs if b["situation"].startswith("Wants sessions"))
    words = " ".join(said[s:e] for s, e in need["pointers"]["own"][0])
    assert words.startswith("Help me add sessions with Flask-Login") and "Marshmallow" not in words and "Cors" not in words


def test_words_copied_from_code_with_bare_quotes_still_read():
    raw = '[{\n "turns": "M2",\n "from": "def admin():",\n "to": "Unauthorized"}), 403",\n "text": "gives code"\n}]'
    assert mv.extract_json_array(raw)[0]["to"] == 'Unauthorized"}), 403'
