import json
from pathlib import Path

import pytest

from examples.point_v2_probe import measure_body_output_tokens, run_probe
from mr_mem.point_sidecar_v2 import PointSidecarError

FIXTURES = Path(__file__).parent / "fixtures/point_v2_dialogues.json"


def test_dialogue_probe_uses_one_body_call_per_turn_and_preserves_the_prefix(tmp_path):
    cases = json.loads(FIXTURES.read_text(encoding="utf-8"))["cases"]
    calls = []

    def body(messages):
        calls.append(messages)
        slots = json.loads(
            messages[-1]["content"].split("Allocated current-turn Point IDs: ")[1].splitlines()[0]
        )
        return json.dumps(
            {
                "schema_version": "body_point_sidecar_v2",
                "response": "继续。",
                "points": [
                    {
                        "point_id": slots[0],
                        "meaning": messages[-2]["content"],
                        "status": "RESOLVED",
                        "context_links": [],
                        "unresolved_refs": [],
                    }
                ],
            }
        ), {"completion_tokens": 100}

    records = run_probe(cases, body, tmp_path / "probe.json")
    expected = sum(len(case["turns"]) for case in cases)
    assert expected == len(calls) == len(records) == 22
    assert all(record["model_call_count"] == 1 for record in records)
    assert calls[1][1:3] == [
        {"role": "user", "content": cases[0]["turns"][0]},
        {"role": "assistant", "content": "继续。"},
    ]
    assert records[1]["points"][0]["turn_id"] == "turn-1"


@pytest.mark.parametrize("failure", [PointSidecarError, RuntimeError])
def test_probe_stops_and_saves_evidence_without_a_repair_call(tmp_path, failure):
    calls = []

    def body(messages):
        calls.append(messages)
        if failure is RuntimeError:
            raise RuntimeError("Body unavailable")
        return "{}", {}

    output = tmp_path / "probe.json"
    with pytest.raises(failure):
        run_probe([{"id": "bad", "turns": ["继续", "后续"]}], body, output)
    records = json.loads(output.read_text(encoding="utf-8"))
    assert len(calls) == len(records) == 1 and records[0]["error"]


def test_incomplete_body_is_saved_without_admitting_a_parseable_prefix(tmp_path):
    output = tmp_path / "truncated.json"
    with pytest.raises(RuntimeError):
        run_probe(
            [{"id": "truncated", "turns": ["继续"]}],
            lambda _: ("{}", {"finish_reason": "length"}),
            output,
        )
    record = json.loads(output.read_text(encoding="utf-8"))[0]
    assert record["raw_body_output"] == "{}"
    assert record["provider_usage"]["finish_reason"] == "length"
    assert "points" not in record


@pytest.mark.parametrize("trace", [[], [{"bytes": [-1]}], [{"bytes": [65]}]])
def test_missing_or_unreliable_token_trace_never_becomes_an_estimate(trace):
    result = measure_body_output_tokens("{}", {"output_token_trace": trace})
    assert result["visible_overhead_tokens_range"] is None


def test_native_replay_keeps_original_assistant_turns_as_source(tmp_path):
    calls = []
    native_reply = {"role": "assistant", "content": "native reply"}

    def body(messages):
        calls.append(messages)
        return json.dumps(
            {"schema_version": "body_point_sidecar_v2", "response": "probe reply", "points": []}
        ), {}

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
    assert records[0]["response"] == "probe reply"


def test_duplicate_prefill_trace_requires_bytes_and_provider_accounting_to_match():
    raw = '{"response":"ok","points":[]}'
    chunks = [b'{"', *[bytes([byte]) for byte in raw.encode()[2:]]]
    trace = [{"bytes": list(chunk)} for chunk in [b'{"', *chunks]]
    usage = {
        "output_token_trace": trace,
        "completion_tokens": len(trace),
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    result = measure_body_output_tokens(raw, usage)
    assert result["visible_output_tokens"] == len(chunks)
    assert result["token_trace_prefill_adjustment"] == 1
    usage["completion_tokens"] += 1
    assert measure_body_output_tokens(raw, usage)["visible_overhead_tokens_range"] is None
