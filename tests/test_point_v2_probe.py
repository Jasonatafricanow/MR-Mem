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
                "status": "RESOLVED",
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
                    "arguments": json.dumps(args) if raw is None else raw,
                },
            }
        ],
    }


def test_probe_uses_one_call_keeps_prefix_and_never_exposes_host_identities(tmp_path):
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    calls = []

    def body(messages, tool):
        calls.append(messages)
        assert tool["function"]["strict"]
        return reply(messages[-2]["content"]), {}

    records = run_probe(cases, body, tmp_path / "probe.json")
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
        return reply(raw="{"), {"finish_reason": "stop"}

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
        lambda _messages, _tool: (reply("明确承诺"), {"finish_reason": "length"}),
        tmp_path / "truncated.json",
    )
    assert records[0]["response"] == "继续。" and records[0]["points"] == []
    assert records[0]["sidecar_error"] == "incomplete Body sidecar"


def test_native_replay_keeps_original_assistant_turns(tmp_path):
    calls = []
    native_reply = {"role": "assistant", "content": "native reply"}

    def body(messages, _tool):
        calls.append(messages)
        return reply(), {}

    records = run_probe(
        [
            {
                "id": "native",
                "history": [{"role": "user", "content": "original prefix"}],
                "turns": ["first", "correction"],
                "native_assistant_after_turn": [[native_reply], []],
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
