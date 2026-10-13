"""``mr-mem-aml serve|check --config runtime.json`` and ``mr-mem-aml fetch-model``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from mr_mem_aml.config import RuntimeConfig


def _check(config: RuntimeConfig) -> int:
    """Preflight every working condition before a long evaluation run."""
    report: dict[str, str] = {}
    ok = True
    try:
        import lce  # noqa: F401

        report["lce"] = "importable"
    except ImportError:
        report["lce"] = "MISSING (pip install -e .[lce])" if config.lce_enabled else "disabled"
        ok = ok and not config.lce_enabled
    from mr_mem_aml.runtime import build_backend, build_embedding

    try:
        started = time.monotonic()
        reply = build_backend(config.compiler).complete('Return exactly this JSON: {"ok": true}')
        elapsed = time.monotonic() - started
        report["compiler"] = (
            f"ok ({elapsed:.1f}s for a trivial call)" if "ok" in reply
            else f"unexpected reply: {reply[:200]}"
        )
        if config.compiler.get("type") == "command" and elapsed > 5:
            report["compiler"] += (
                " - a command-line agent adds fixed startup cost to every Add;"
                " prefer an OpenAI-compatible endpoint for dataset runs"
            )
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        report["compiler"] = f"FAILED: {exc}"
        ok = False
    try:
        embedding = build_embedding(config.embedding)
        if embedding is None:
            report["embedding"] = "not configured (BM25 only)"
        else:
            vector = embedding.embed("health check")
            report["embedding"] = f"ok ({len(vector)} dims)"
            if len(vector) != embedding.identity.dimension:
                report["embedding"] += " DIMENSION MISMATCH"
                ok = False
    except Exception as exc:  # noqa: BLE001 - diagnostic boundary
        report["embedding"] = f"FAILED: {exc}"
        ok = False
    report["data_dir"] = str(config.data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mr-mem-aml")
    parser.add_argument("command", choices=("serve", "check", "fetch-model"))
    parser.add_argument("--config", help="runtime JSON (serve, check)")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5", help="fetch-model")
    parser.add_argument("--dir", help="fetch-model: target directory for model files")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    if args.command == "fetch-model":
        if not args.dir:
            parser.error("fetch-model needs --dir")
        from mr_mem_aml.embedding import fetch_fastembed_model

        path = fetch_fastembed_model(args.model, args.dir)
        print(json.dumps({"model": args.model, "model_dir": str(path)}, indent=1))
        return 0
    if not args.config:
        parser.error(f"{args.command} needs --config")
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    config = RuntimeConfig.load(args.config)
    if args.command == "check":
        return _check(config)
    from mr_mem_aml.server import serve

    serve(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
