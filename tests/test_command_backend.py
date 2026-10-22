import sys
import time

from mr_mem_aml.compiler import CommandBackend

# A CLI that prints its final result event and then stays alive, as agy does.
LINGERING = r"""
import json, sys, time
sys.stdin.read()
print("log line, not json", flush=True)
print(json.dumps({"event": "step_update"}), flush=True)
print(json.dumps({"event": "result", "result": {"status": "SUCCESS",
                  "response": '{"blocks": []}'}}), flush=True)
time.sleep(60)
"""


def test_stream_json_returns_the_result_without_waiting_for_exit():
    backend = CommandBackend([sys.executable, "-c", LINGERING], timeout=30, attempts=1,
                             stream_json=True)
    started = time.monotonic()
    assert backend.complete("prompt") == '{"blocks": []}'
    assert time.monotonic() - started < 20
