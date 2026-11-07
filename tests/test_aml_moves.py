"""Work scene, two-step compile (mr_mem_aml/moves.py): step 1 writes the SBs ("what happened ->
what the person did"), step 2 places them among candidates; the prompt follows the person's
language."""

import json
import re
import sqlite3
from pathlib import Path

import pytest

from mr_mem_aml import moves as mv
from mr_mem_aml.compiler import SemanticCompiler
from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.runtime import MemoryRuntime
from tests.aml_live_backend import LiveBackend

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
    data = {
        "moves": [
            {
                "msg": "M2",
                "about": ["M1"],
                "situation": "缓存加上了，导入 9 分钟降到 2 分钟",
                "move": "认可",
                "happened": True,
            },
            {"msg": "M2", "about": ["M1"], "situation": "导出没改 CSV 格式", "move": "要求改格式"},
            {"msg": "M1", "about": [], "situation": "x", "move": "not the person's"},  # dropped
            {"msg": "M9", "about": [], "situation": "x", "move": "unknown id"},
        ],  # dropped
        "reports": [{"msgs": ["M3", "M4"], "situation": "格式改好了", "happened": False}],
    }  # no SB
    sbs = mv.parse_step1(data, roles)
    assert [(b["sources"], b["move"], b["happened"]) for b in sbs] == [
        ([0, 1], "认可", True),
        ([0, 1], "要求改格式", True),
    ]


def test_step1_about_only_names_earlier_agent_messages():
    sbs = mv.parse_step1(
        {"moves": [{"msg": "M2", "about": ["M1", "M2", "M3"], "situation": "s", "move": "m"}]},
        ["assistant", "user", "assistant"],
    )
    assert sbs[0]["sources"] == [0, 1]


def test_the_tail_after_the_person_goes_to_the_next_window():
    assert mv.tail_start(["assistant", "user", "assistant", "assistant"]) == 2
    assert mv.tail_start(["assistant", "user"]) is None



# --- step 2 ---------------------------------------------------------------------



def test_prompts_follow_the_language():
    zh = mv.step1_prompt("zh", [{"role": "user", "text": "开始吧"}], [])
    en = mv.step1_prompt("en", [{"role": "user", "text": "go ahead"}], [])
    assert "用中文写" in zh and "[M1 人] 开始吧" in zh
    assert "Write in English" in en and "[M1 Person] go ahead" in en
    st = mv.stretch_prompt("zh", [{"text": "a", "time": "t"}], [], [], [])
    assert "s1 [t] a" in st and "前因" in st


# --- runtime: two steps per window, journal, replay -------------------------------


class TwoStepBackend:
    """Step 1: one SB per message of the person, about the agent message before it.
    The stretch layer: one closed stretch over the window's SBs, so they go on one after another
    on the Line."""

    def __init__(self):
        self.calls = {"step1": 0, "stretches": 0}

    def complete(self, prompt: str) -> str:
        if re.search(r"\nSBS[:：]\n", prompt):
            self.calls["stretches"] += 1
            return LiveBackend()._stretches(prompt)
        self.calls["step1"] += 1
        new = re.split(r"NEW[:：]\n", prompt, maxsplit=1)[1]
        rows = re.findall(r"^\[M(\d+)[^\]]* (Person|人|Agent)\] (.*)$", new, flags=re.M)
        moves = [
            {
                "msg": f"M{i}",
                "about": [f"M{int(i) - 1}"] if int(i) > 1 else [],
                "situation": f"sit {text}",
                "move": f"move {text}",
                "happened": True,
            }
            for i, who, text in rows
            if who in ("Person", "人")
        ]
        return json.dumps({"moves": moves, "reports": []})


def runtime(tmp_path: Path, backend) -> MemoryRuntime:
    config = RuntimeConfig(
        data_dir=tmp_path / "data",
        compiler={"type": "command", "argv": ["x"]},
        compile_window={"soft": 3, "hard": 5},
        scene="work",
    )
    return MemoryRuntime(config, compiler=SemanticCompiler(backend, scene="work"))


def add(rt, rid, *texts, start_minute=0):
    messages = [
        {
            "role": "assistant" if i % 2 == 0 else "user",
            "content": t,
            "timestamp": 1_690_000_000_000 + (start_minute + i) * 60_000,
        }
        for i, t in enumerate(texts)
    ]
    return rt.add({"request_id": rid, "user_id": "u1", "session_id": "s1", "messages": messages})


def edges(tmp_path: Path) -> int:
    path = next((tmp_path / "data" / "lce_tree").rglob("lines.sqlite"))
    return sqlite3.connect(path).execute("SELECT COUNT(*) FROM line_edges").fetchone()[0]


def test_step1_and_the_stretch_layer_link_on_the_graph_and_journal(tmp_path):
    backend = TwoStepBackend()
    rt = runtime(tmp_path, backend)
    try:
        add(rt, "r1", "报告一", "好，继续", "报告二", "改一下格式", "报告三", "可以")
        rt.flush()
        assert backend.calls["step1"] == backend.calls["stretches"] >= 1
        assert len(rt.core.load_all()) == 3  # one SB per point of the person
        assert all("→ 人：move" in m.content for m in rt.core.load_all())  # Chinese prompt
        assert edges(tmp_path) >= 2
        rows = (
            sqlite3.connect(tmp_path / "data" / "aml_source.sqlite")
            .execute("SELECT compiled FROM windows")
            .fetchall()
        )
        placed = [json.loads(r[0])["placement"] for r in rows]
        assert all(p["language"] == "zh" and "refs" in p and "after" in p for p in placed)
    finally:
        rt.close()


def test_a_journaled_window_replays_both_steps_without_a_model_call(tmp_path):
    first = TwoStepBackend()
    rt = runtime(tmp_path / "a", first)
    try:
        add(
            rt,
            "r1",
            "report one",
            "ok go on",
            "report two",
            "fix the format",
            "report three",
            "fine",
        )
        rt.flush()
        rows = (
            sqlite3.connect(tmp_path / "a" / "data" / "aml_source.sqlite")
            .execute(
                "SELECT window_id,user_id,first_seq,last_seq,compiled FROM windows ORDER BY firs"
                "t_seq"
            )
            .fetchall()
        )
        assert all(json.loads(r[4])["placement"]["language"] == "en" for r in rows)
    finally:
        rt.close()
    replay = TwoStepBackend()
    rt = runtime(tmp_path / "b", replay)
    try:
        for r in rows:
            rt.source.put_window(r[0], r[1], r[2], r[3], json.loads(r[4]))
        add(
            rt,
            "r1",
            "report one",
            "ok go on",
            "report two",
            "fix the format",
            "report three",
            "fine",
        )
        rt.flush()
        assert replay.calls == {"step1": 0, "stretches": 0}
        assert edges(tmp_path / "b") == edges(tmp_path / "a")
    finally:
        rt.close()



def test_a_step1_answer_missing_most_of_the_person_is_asked_again():
    roles = ["assistant", "user", "assistant", "user", "assistant", "user"]
    one = {"moves": [{"msg": "M6", "about": ["M5"], "situation": "s", "move": "m"}]}
    with pytest.raises(ValueError):
        mv.check_step1(one, roles)
    two = {
        "moves": [
            {"msg": "M2", "situation": "s", "move": "m"},
            {"msg": "M4", "situation": "s", "move": "m"},
        ]
    }
    mv.check_step1(two, roles)  # two of three: kept


def test_interleaved_sessions_are_marked_and_about_stays_in_its_session():
    new = [
        {"role": "assistant", "text": "清洗查完了", "session": "a"},
        {"role": "assistant", "text": "回滚脚本提交了", "session": "b"},
        {"role": "user", "text": "这两个可以同步进行", "session": "a"},
    ]
    prompt = mv.step1_prompt("zh", new, [])
    assert "[M1 会话A Agent]" in prompt and "[M2 会话B Agent]" in prompt
    data = {"moves": [{"msg": "M3", "about": ["M1", "M2"], "situation": "s", "move": "m"}]}
    assert mv.parse_step1(data, ["assistant", "assistant", "user"], ["a", "b", "a"])[0][
        "sources"
    ] == [0, 2]



def test_chat_step1_carries_standing_instructions_into_the_journal():
    from mr_mem_aml.compiler import CompiledBlock, blocks_from_json, blocks_to_json

    sb = mv.parse_step1_chat(
        [
            {
                "turns": "M1",
                "text": "always use version numbers",
                "when": "before",
                "status": "confirmed",
                "standing": True,
            }
        ],
        ["user"],
    )[0]
    assert sb["standing"] is True
    block = CompiledBlock("c", ("r",), standing=True)
    assert blocks_from_json(blocks_to_json((block,)))[0].standing is True
    assert '"standing":false' in mv.step1_prompt_topic("en", [{"role": "user", "text": "hi"}], [])


def test_long_messages_are_numbered_by_paragraph_and_turns_may_point_into_them():
    from mr_mem_aml import moves as mv

    long = "Sure!\n\nWeek 1: backend\n\n```python\nx = 1\n\ny = 2\n```\n\nGood luck."
    assert len(mv.paragraphs(long)) == 4 and mv.paragraphs(long)[2].startswith("```python")
    assert mv.paragraphs("one line") == ["one line"]
    assert mv.paragraphs("my code:\n```py\nx = 1\n```\nwhy?") == [
        "my code:",
        "```py\nx = 1\n```",
        "why?",
    ]  # code on its own
    prompt = mv.step1_prompt_topic(
        "en", [{"role": "user", "text": "plan?"}, {"role": "assistant", "text": long}], []
    )
    assert "(M2.2) Week 1: backend" in prompt and "(M1.1)" not in prompt
    data = [
        {"turns": "M1, M2.2-M2.3", "text": "asked for a plan; got week 1"},
        {"turns": "M1-M2", "text": "whole"},
        {"turns": "M2.2-3", "text": "same paragraphs"},
        {"turns": "M2.1-M2.4", "text": "all of it"},
    ]
    sbs = mv.parse_step1_chat(data, ["user", "assistant"], [1, 4])
    got = {b["situation"].split(" [")[0]: (b["sources"], b["spans"]) for b in sbs}
    assert got == {
        "asked for a plan; got week 1": ([0, 1], {1: [2, 3]}),
        "whole": ([0, 1], {}),
        "same paragraphs": ([1], {1: [2, 3]}),
        "all of it": ([1], {}),
    }


def test_chat_step1_sbs_and_what_builds_on_them_come_from_one_answer():
    from mr_mem_aml import moves as mv

    answer = "Sure!\n\nWeek 1: backend\n\n```python\na = 1\nb = 2\nc = 3\n```"
    prompt = mv.step1_prompt_topic(
        "en", [{"role": "user", "text": "plan?"}, {"role": "assistant", "text": answer}], []
    )
    assert "(M2.3) ```python" in prompt and "  2| b = 2" in prompt  # code lines are numbered
    raw = (
        "ANALYSIS: the person moved the deadline and asked for a plan.\n"
        "JSON:\n"
        '[{"turns":"M1","text":"The deadline moved to March 31","when":"before","'
        'status":"confirmed","standing":false},\n'
        ' {"turns":"M1","text":"The person asks for a plan","when":"before","stat'
        'us":"confirmed","next":[\n'
        '   {"turns":"M2.2","text":"The assistant plans week 1: backend","when":"af'
        'ter","status":"unknown","next":[\n'
        '     {"turns":"M2.3:2-2","text":"The assistant sets b to 2","when":"after'
        '","status":"unknown"}]}]}]'
    )
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user", "assistant"], [1, 3])
    by = {b["situation"].split(" [")[0]: b for b in sbs}
    assert by["The deadline moved to March 31"]["situation"].endswith("[confirmed; before]")
    assert (
        by["The deadline moved to March 31"]["happened"] is True
        and by["The deadline moved to March 31"]["under"] is None
    )
    ask, week, b = (
        by["The person asks for a plan"],
        by["The assistant plans week 1: backend"],
        by["The assistant sets b to 2"],
    )
    assert sbs[week["under"]] is ask and sbs[b["under"]] is week  # the steps building on the ask
    assert (
        week["sources"] == [1] and week["pointers"]["own"] == {1: [2]} and week["happened"] is False
    )


def test_each_sb_keeps_the_fragment_of_the_words_it_comes_from():
    from mr_mem_aml import moves as mv

    said = (
        "I'm trying to optimize the dashboard, and I've never actually integrated Flask-Login, "
        "so I'm starting from scratch, and I'd like to use Marshmallow to validate data"
    )
    reply = "Sure.\n\nUse LoginManager.\n\nUse Schema classes."
    raw = (
        "JSON:\n"
        '[{"turns":"M1","from":"I\'ve never actually integrated","to":"starting from'
        ' scratch","text":"Never integrated Flask-Login","when":"before","status":'
        '"confirmed"},\n'
        ' {"turns":"M1","from":"I\'d like to use Marshmallow","to":"validate data",'
        '"text":"Wants Marshmallow","when":"after","status":"likely","next":[\n'
        '   {"turns":"M2.3","from":"Use Schema","to":"classes.","text":"Use Sch'
        'ema classes","when":"after","status":"unknown"}]},\n'
        ' {"turns":"M1","from":"words that are not there","to":"","text":"Copie'
        'd wrong","when":"before","status":"confirmed"}]'
    )
    sbs = mv.parse_step1_chat(
        mv.extract_after_json(raw), ["user", "assistant"], [1, 3], [said, reply]
    )
    by = {b["situation"].split(" [")[0]: b["pointers"] for b in sbs}
    ((s, e),) = by["Never integrated Flask-Login"]["own"][0]
    assert said[s:e] == "I've never actually integrated Flask-Login, so I'm starting from scratch"
    assert (
        said[slice(*by["Wants Marshmallow"]["own"][0][0])]
        == "I'd like to use Marshmallow to validate data"
    )
    assert reply[slice(*by["Use Schema classes"]["own"][1][0])] == "Use Schema classes."
    assert (
        by["Copied wrong"]["own"] == {0: None} and by["Copied wrong"]["missed"] == 1
    )  # whole, and counted


def test_the_persons_words_no_sb_covers_are_found():
    from mr_mem_aml import moves as mv

    said = (
        "Thanks. I never used Docker before this. Also please show how to deploy it with Guni"
        "corn on Render tomorrow."
    )
    sbs = [{"pointers": {"own": {0: [[8, 40]]}}}]  # only "I never used Docker before this."
    left = mv.uncovered(sbs, ["user"], [said])
    assert left == [
        (0, "Also please show how to deploy it with Gunicorn on Render tomorrow.")
    ]  # "Thanks." too short


def test_code_the_person_pastes_goes_with_the_sb_that_brings_it_in():
    from mr_mem_aml import moves as mv

    said = (
        "My login fails. Here's my code:\n\n```python\nlogin()\n```\n\nAlso, should I use bcrypt?"
    )
    raw = (
        "JSON:\n"
        '[{"turns":"M1","from":"My login fails","to":"my code:","text":"The perso'
        'n reports login fails",'
        '"when":"before","status":"confirmed","standing":false},'
        '{"turns":"M1","from":"Also, should","to":"bcrypt?","text":"The person asks about bcrypt",'
        '"when":"before","status":"confirmed","standing":false}]'
    )
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user"], [3], [said])
    code = said.index("```python")
    first, second = (b["pointers"]["own"][0] for b in sbs)
    assert any(s == code for s, _ in first) and not any(s == code for s, _ in second)


def test_an_answer_that_stops_keeps_only_what_it_compiled_and_says_where():
    from mr_mem_aml import moves as mv

    assert mv.stop_at("[...]\nSTOP: M12", 14) == 12
    assert (
        mv.stop_at("[...]", 14) == 14 and mv.stop_at("STOP: M40", 14) == 14
    )  # none, or past the end: all
    sbs = [
        {"sources": [0], "under": None},
        {"sources": [1], "under": 0},
        {"sources": [2], "under": 1},
        {"sources": [1, 2], "under": 0},
    ]
    kept = mv.compiled_before(sbs, 2)
    assert [b["sources"] for b in kept] == [[0], [1]] and kept[1]["under"] == 0


def test_what_the_assistant_says_is_not_confirmed_by_saying_it():
    from mr_mem_aml import moves as mv

    sbs = mv.parse_step1_chat(
        [
            {
                "turns": "M1",
                "text": "The person measured 150ms",
                "when": "before",
                "status": "confirmed",
                "next": [
                    {
                        "turns": "M2",
                        "text": "The assistant says the cache made it faster",
                        "when": "before",
                        "status": "confirmed",
                    }
                ],
            }
        ],
        ["user", "assistant"],
    )
    assert [(b["situation"].rsplit(" [", 1)[1], b["happened"]) for b in sbs] == [
        ("confirmed; before]", True),
        ("unknown; before]", False),
    ]


def test_words_that_come_more_than_once_give_the_shortest_span():
    from mr_mem_aml import moves as mv

    said = (
        "please show an example of A, and please show an example of B, and please show an exa"
        "mple of C"
    )
    s, e = mv.anchor(said, "please show an example", "of B")
    assert said[s:e] == "please show an example of B"


def test_a_need_keeps_its_own_words_and_its_things_keep_theirs():
    from mr_mem_aml import moves as mv

    said = (
        "Help me add sessions with Flask-Login, and use Marshmallow to validate, and use Flas"
        "k-Cors for CORS"
    )
    raw = (
        "JSON:\n"
        '[{"turns":"M1","from":"Help me add sessions","to":"for CORS","text":"W'
        'ants sessions","when":"before","status":"confirmed","next":[\n'
        '   {"turns":"M1","from":"use Marshmallow","to":"to validate","text":"W'
        'ants Marshmallow","when":"after","status":"likely"},\n'
        '   {"turns":"M1","from":"use Flask-Cors","to":"for CORS","text":"Wants'
        ' Flask-Cors","when":"after","status":"likely"}]}]'
    )
    sbs = mv.parse_step1_chat(mv.extract_after_json(raw), ["user"], [1], [said])
    need = next(b for b in sbs if b["situation"].startswith("Wants sessions"))
    words = " ".join(said[s:e] for s, e in need["pointers"]["own"][0])
    assert (
        words.startswith("Help me add sessions with Flask-Login")
        and "Marshmallow" not in words
        and "Cors" not in words
    )


def test_words_copied_from_code_with_bare_quotes_still_read():
    raw = (
        "[{\n"
        ' "turns": "M2",\n'
        ' "from": "def admin():",\n'
        ' "to": "Unauthorized"}), 403",\n'
        ' "text": "gives code"\n'
        "}]"
    )
    assert mv.extract_json_array(raw)[0]["to"] == 'Unauthorized"}), 403'



# --- chat step 1: broken answers, pasted material, people (2026-10-31, mini AML run) ------------


def _sb(turns):
    return {
        "turns": turns,
        "text": "The person ...",
        "when": "before",
        "status": "confirmed",
        "standing": False,
    }


def test_chat_check_rejects_one_item_and_done():
    with pytest.raises(ValueError):
        mv.check_step1_chat([_sb("M1")], ["user", "assistant"] * 5, done=10)


def test_chat_check_takes_pasted_material_answered_and_stops():
    # a rulebook pasted over M1-M6, the question M7, the answer M8: no SB for the rulebook
    roles = ["user"] * 7 + ["assistant"]
    mv.check_step1_chat([_sb("M7"), {**_sb("M8"), "text": "The assistant ..."}], roles, done=8)
    # the answer stopped after M1 (it held enough SBs): only M1 counts
    mv.check_step1_chat([_sb("M1")], ["user"] * 9, done=1)
    # nothing to compile (a machine log): no retry
    mv.check_step1_chat([], ["user"] * 9, done=9)



def test_the_open_stretch_is_cut_with_the_window():
    from mr_mem_aml.moves import parse_stretches, stretch_prompt

    prompt = stretch_prompt(
        "en", [{"text": "new", "time": ""}], [{"text": "old", "time": ""}], [], []
    )
    assert "o1 old\ns1 new" in prompt
    carried = parse_stretches({"stretches": [{"sbs": "o1-s1", "closed": False}]}, 1, 0, 0, n_open=1)
    assert carried["stretches"][0]["joins_open"] and carried["stretches"][0]["sbs"] == [0]
    assert carried["open_summary"] == ""
    turned = {
        "stretches": [
            {"sbs": "o1", "closed": True, "summary": "Cause none, ..."},
            {"sbs": "s1", "closed": False},
        ]
    }
    got = parse_stretches(turned, 1, 0, 0, n_open=1)
    assert got["open_summary"] == "Cause none, ..." and not got["stretches"][0]["joins_open"]
    for bad in (
        {"stretches": [{"sbs": "o1", "closed": True}, {"sbs": "s1"}]},  # no summary
        {"stretches": [{"sbs": "s1"}]},
    ):  # the o's left out
        try:
            parse_stretches(bad, 1, 0, 0, n_open=1)
            raise AssertionError(f"must be asked again: {bad}")
        except ValueError:
            pass


def test_block_numbers_cut_the_raw_text_with_compression_on():
    # an SB's blocks are paragraph numbers of the raw text, also when step 1 reads an assistant
    # message cut down: spans taken from the shortened text cut the raw text in the wrong place
    import json as _json
    from types import SimpleNamespace

    from mr_mem_aml.compiler import SemanticCompiler

    first = ("Opening words " * 40).strip()
    second = ("The second paragraph about tokens " * 30).strip()
    third = ("Closing words " * 40).strip()
    raw = "\n\n".join([first, second, third])
    seen = {}

    class Backend:
        def complete(self, prompt):
            seen["prompt"] = prompt
            return (
                'JSON:\n[{"turns":"M1","text":"The person asks","when":"before",'
                '"status":"confirmed","standing":false},'
                '{"turns":"M2.2","text":"The assistant explains tokens","when":"before",'
                '"status":"unknown","standing":false}]\nSTOP: M2'
            )

    def message(role, text, n):
        return SimpleNamespace(
            role=role,
            content=text,
            session_id="s",
            record_id=f"r{n}",
            occurred_at=None,
            time_given=False,
        )

    compiler = SemanticCompiler(Backend(), assistant_head=600, blocks=True)
    blocks = compiler._compile_moves(
        [message("user", "How do tokens work?", 1), message("assistant", raw, 2)],
        (),
        {"language": "en"},
        False,
        None,
    )
    assert "(M2.3)" in seen["prompt"] and "only the paragraphs on this thing" in seen["prompt"]
    assert second not in seen["prompt"]  # compression still holds
    [(start, end)] = _json.loads(blocks[1].pointers)["own"]["r2"]
    assert raw[start:end] == second
