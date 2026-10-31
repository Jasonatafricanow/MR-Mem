"""Validate deployment settings without sending any provider requests, then serve."""

import os
import signal
from dataclasses import replace

from mr_mem_aml.config import RuntimeConfig
from mr_mem_aml.server import serve


def main() -> None:
    config = RuntimeConfig.load(os.environ.get("MRMEM_CONFIG", "/app/deploy/runtime.server.json"))
    config = replace(config, compile_concurrency=int(os.environ.get(
        "MRMEM_COMPILE_CONCURRENCY", config.compile_concurrency)))
    required = {config.auth_token_env}
    for spec in (config.compiler, config.embedding, config.decision):
        if spec and spec.get("api_key_env"):
            required.add(spec["api_key_env"])
    missing = sorted(name for name in required if name and not os.environ.get(name, "").strip())
    if missing:
        raise SystemExit("Missing environment variables: " + ", ".join(missing))

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    serve(config)


if __name__ == "__main__":
    main()
