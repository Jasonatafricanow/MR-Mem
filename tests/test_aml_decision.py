"""Contract of the decision bypass (decision.py): when a note is written, and that a failure
leaves the readout alone."""

from mr_mem_aml.decision import DecisionBypass, Message


def bypass(answers):
    d = DecisionBypass({"base_url": "http://unused"})
    d._ask = lambda state, questions: answers(state, questions)
    return d


HITS = [
    Message("user", "I set up my API key in .env"),
    Message("assistant", "Great, here is how to use the key"),
    Message("user", "I've never obtained an API key"),
]


def said(state, questions):
    if "kind" in questions:
        return {"kind": {"choice": "yesno"}}
    did = "set up" in state["message"]
    return {"affirm": {"noul": 0.9 if did else 0.1}, "deny": {"noul": 0.1 if did else 0.95}}


def test_contradiction_quotes_both_messages_of_the_person():
    note = bypass(said).note("Have I obtained an API key?", HITS)
    assert "set up my API key" in note and "never obtained" in note


def test_one_side_only_or_failure_gives_no_note():
    def one_side(state, questions):
        return (
            said(state, questions)
            if "kind" in questions
            else {"affirm": {"noul": 0.9}, "deny": {"noul": 0.1}}
        )

    def down(state, questions):
        raise RuntimeError("endpoint down")

    assert bypass(one_side).note("Have I obtained an API key?", HITS) is None
    assert bypass(down).note("Have I obtained an API key?", HITS) is None


def test_absence_note_when_only_the_assistant_gave_it():
    def where(state, questions):
        if "kind" in questions:
            return {"kind": {"choice": "specific"}}
        return {"src": {"choice": "assistant_only"}}

    assert "assistant's suggestions" in bypass(where).note(
        "Which ESLint rules did I enforce?", HITS
    )


def test_absence_checks_the_tail_and_skips_over_budget_messages():
    seen = []
    def answer(state, questions):
        if "kind" in questions:
            return {"kind": {"choice": "specific"}}
        seen.append(state["messages"])
        assert "I used ESLint" in state["messages"]
        return {"src": {"choice": "person"}}
    d = bypass(answer)
    assert d.note("Which rules?", [Message("user", "x" * 700 + " I used ESLint")]) is None
    assert len(seen) == 1
    assert d.note("Which rules?", [Message("user", "x" * 16000)]) is None
    assert len(seen) == 1  # no incomplete-message absence judgment


def test_contradiction_quotes_exact_checked_text_with_source_and_time():
    hits = [Message("user", "x" * 3500 + " set up my API key", "r1", "2026-01-01"),
            Message("user", "I've never obtained an API key", "r2", "2025-01-01")]
    note = bypass(said).note("Have I obtained an API key?", hits)
    assert "set up my API key" in note and "source=r1" in note
    assert "2026-01-01" in note and "compare their times" in note
    assert bypass(said).note("Have I obtained an API key?", [
        Message("user", "x" * 6500 + " set up"), hits[1]]) is None
