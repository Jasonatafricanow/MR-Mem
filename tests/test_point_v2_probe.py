import json
from pathlib import Path

import pytest

from examples.point_v2_probe import measure_body_output_tokens, run_probe
from mr_mem.point_sidecar_v2 import SIDECAR_TOOL, BodyResponseError

FIXTURES = Path(__file__).parent / "fixtures/point_v2_dialogues.json"


def reply(meaning="", raw=None):
    args = {
        "points": [
            {
                "slot": 0,
                "meaning": meaning,
                "context_links": [],
                "unresolved_refs": [],
            }
        ]
        if meaning
        else []
    }
    return {
        "content": "继续。",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": SIDECAR_TOOL,
                    "arguments": json.dumps({"response": "继续。", **args}) if raw is None else raw,
                },
            }
        ],
    }


def framed_reply(meaning="", raw=None):
    arguments = reply(meaning, raw)["tool_calls"][0]["function"]["arguments"]
    if raw is None:
        arguments = json.dumps({"points": json.loads(arguments)["points"]})
    return "继续。\n<point_sidecar>" + arguments + "\n</point_sidecar>"


def test_probe_uses_one_call_keeps_prefix_and_never_exposes_host_identities(tmp_path):
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    calls = []

    def body(messages, tool):
        calls.append(messages)
        assert tool["function"]["strict"]
        return reply(messages[-2]["content"]), {}

    records = run_probe(cases, body, tmp_path / "probe.json", transport="tool")
    assert len(calls) == len(records) == 22
    assert all(record["model_call_count"] == 1 for record in records)
    assert calls[1][1:3] == [
        {"role": "user", "content": cases[0]["turns"][0]},
        {"role": "assistant", "content": "继续。"},
    ]
    assert records[1]["points"][0]["turn_id"] == "turn-1"
    assert '"alias": "p0"' in calls[1][-1]["content"]
    assert "point-v2:" not in json.dumps(calls)


def test_rejected_sidecar_saves_response_and_continues_without_repair(tmp_path):
    calls = []

    def body(messages, _tool):
        calls.append(messages)
        return framed_reply(raw="{"), {"finish_reason": "stop"}

    records = run_probe([{"id": "bad", "turns": ["继续", "后续"]}], body, tmp_path / "probe.json")
    assert len(calls) == len(records) == 2
    assert all(r["response"] == "继续。" and r["sidecar_status"] == "REJECTED" for r in records)
    assert calls[1][2] == {"role": "assistant", "content": "继续。"}
    assert "Previous Point proposals: []" in calls[1][-1]["content"]


@pytest.mark.parametrize("failure", [RuntimeError, BodyResponseError])
def test_unavailable_normal_response_stops_and_saves_evidence(tmp_path, failure):
    calls = []

    def body(messages, _tool):
        calls.append(messages)
        if failure is RuntimeError:
            raise RuntimeError("Body unavailable")
        return {"content": None}, {}

    output = tmp_path / "probe.json"
    with pytest.raises(failure):
        run_probe([{"id": "bad", "turns": ["继续", "后续"]}], body, output)
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == len(records) == 1 and records[0]["error"]


def test_truncation_preserves_content_but_rejects_even_parseable_sidecar(tmp_path):
    records = run_probe(
        [{"id": "truncated", "turns": ["继续"]}],
        lambda _messages, _tool: (framed_reply("明确承诺"), {"finish_reason": "length"}),
        tmp_path / "truncated.json",
    )
    assert records[0]["response"] == "继续。" and records[0]["points"] == []
    assert records[0]["sidecar_error"] == "incomplete Body sidecar"


@pytest.mark.parametrize("reply_field", ["native_assistant_after_turn", "assistant_after_turn"])
def test_source_replay_keeps_original_assistant_turns(tmp_path, reply_field):
    calls = []
    native_reply = {"role": "assistant", "content": "native reply"}

    def body(messages, _tool):
        calls.append(messages)
        return framed_reply(), {}

    records = run_probe(
        [
            {
                "id": "native",
                "history": [{"role": "user", "content": "original prefix"}],
                "turns": ["first", "correction"],
                reply_field: [[native_reply], []],
            }
        ],
        body,
        tmp_path / "native.json",
    )
    assert calls[1][1:4] == [
        {"role": "user", "content": "original prefix"},
        {"role": "user", "content": "first"},
        native_reply,
    ]
    assert records[0]["response"] == "继续。"


@pytest.mark.parametrize("trace", [[], [{"bytes": [-1]}], [{"bytes": [65]}]])
def test_unreliable_trace_never_becomes_an_estimate(trace):
    assert (
        measure_body_output_tokens(reply(), {"output_token_trace": trace})[
            "visible_overhead_tokens_range"
        ]
        is None
    )


def test_tool_overhead_requires_byte_alignment_and_provider_accounting():
    raw = reply()
    trace = [{"bytes": [byte]} for byte in raw["content"].encode()]
    usage = {
        "output_token_trace": trace,
        "completion_tokens": len(trace) + 30,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    assert measure_body_output_tokens(raw, usage)["visible_overhead_tokens_range"] == [20, 20]
    usage["completion_tokens"] = 1
    assert measure_body_output_tokens(raw, usage)["visible_overhead_tokens_range"] is None


def test_frame_tokens_are_measured_only_from_exact_trace_bytes():
    raw = framed_reply("明确承诺")
    trace = [{"bytes": [byte]} for byte in raw.encode()]
    usage = {
        "output_token_trace": trace,
        "completion_tokens": len(trace) + 10,
        "completion_tokens_details": {"reasoning_tokens": 10},
    }
    overhead = len(raw.encode()) - len("继续。".encode())
    assert measure_body_output_tokens(raw, usage, transport="frame")[
        "visible_overhead_tokens_range"
    ] == [overhead, overhead]
    usage["completion_tokens"] += 1
    measured = measure_body_output_tokens(raw, usage, transport="frame")
    assert measured["visible_overhead_tokens_range"] == [overhead, overhead + 1]
    assert measured["untraced_visible_tokens"] == 1
    usage["output_token_trace"][0]["bytes"] = [65]
    assert (
        measure_body_output_tokens(raw, usage, transport="frame")["visible_overhead_tokens_range"]
        is None
    )


def test_native_missing_generated_reply_is_failure_but_source_replay_continues(tmp_path):
    calls = []
    original = {"role": "assistant", "content": "original source reply"}

    def body(messages, _tool):
        calls.append(messages)
        return ("" if len(calls) == 1 else framed_reply("明确更正")), {"finish_reason": "stop"}

    records = run_probe(
        [
            {
                "id": "native",
                "turns": ["first", "correction"],
                "native_assistant_after_turn": [[original], []],
            }
        ],
        body,
        tmp_path / "native.json",
    )
    assert len(calls) == len(records) == 2 and records[0]["error"]
    assert "response" not in records[0] and records[1]["sidecar_status"] == "ACCEPTED"
    assert calls[1][2] == original
