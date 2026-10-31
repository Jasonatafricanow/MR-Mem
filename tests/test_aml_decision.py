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


def test_long_hits_are_still_judged():
    # 30 BEAM hits run past 15000 characters; skipping them judged no question at all
    seen = []

    def answer(state, questions):
        if "kind" in questions:
            return {"kind": {"choice": "specific"}}
        seen.append(state["messages"])
        return {"src": {"choice": "absent"}}

    hits = [Message("user", "I set up the linter. " + "x" * 2000) for _ in range(30)]
    assert bypass(answer).note("Which rules?", hits) is not None
    assert len(seen) == 1 and len(seen[0]) <= 15000

    long_did = [Message("user", "I set up my API key " + "x" * 7000), HITS[2]]
    assert bypass(said).note("Have I obtained an API key?", long_did) is not None


def test_deadline_returns_without_waiting_for_slow_backend():
    import threading
    import time
    done = threading.Event()
    release = threading.Event()
    d = DecisionBypass({"base_url": "http://unused", "total_timeout": 0.03})
    def slow(state, questions):
        release.wait(1)
        done.set()
        return {"kind": {"choice": "other"}}
    d._ask = slow
    start = time.monotonic()
    try:
        assert d.note("q", HITS) is None
        assert time.monotonic() - start < 0.5
        assert d.health()["timeouts"] == 1
        assert d.health()["status"] == "degraded"
    finally:
        release.set()
        assert done.wait(1)


def test_health_distinguishes_unused_failure_and_recovery():
    d = bypass(lambda state, questions: {"kind": {"choice": "other"}})
    assert d.health()["status"] == "not_checked"
    assert d.note("q", HITS) is None
    assert d.health()["status"] == "available"
    def fail(state, questions):
        raise RuntimeError("private error text must not reach health")
    d._ask = fail
    assert d.note("q", HITS) is None
    assert d.health()["failures"] == 1
    assert "private" not in str(d.health())


def test_retry_after_cannot_extend_the_total_deadline(monkeypatch):
    import urllib.error
    import urllib.request
    from email.message import Message as Headers
    calls = []
    headers = Headers()
    headers["Retry-After"] = "60"
    def limited(*args, **kwargs):
        calls.append(kwargs["timeout"])
        raise urllib.error.HTTPError("http://unused", 429, "rate limited", headers, None)
    monkeypatch.setattr(urllib.request, "urlopen", limited)
    d = DecisionBypass({"base_url": "http://unused", "total_timeout": 0.1})
    assert d.note("q", HITS) is None
    assert len(calls) == 1 and calls[0] <= 0.1
    assert d.health()["timeouts"] == 1
