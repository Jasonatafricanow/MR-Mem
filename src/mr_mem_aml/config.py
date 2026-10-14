"""Runtime configuration (JSON file). Fields: docs/AML.md; full configs: examples/."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mr_mem.integrations.path_a_read import ReadPolicy


@dataclass(frozen=True)
class RuntimeConfig:
    data_dir: Path
    compiler: dict[str, Any]
    search_mode: str = "path_a"  # path_a | baseline
    lce_enabled: bool = True
    context_messages: int = 20
    assistant_policy: str = "attributed"  # attributed | context_only
    embedding: dict[str, Any] | None = None
    hyde: str = "off"  # off | fallback | augmented
    read_policy: ReadPolicy = field(default_factory=ReadPolicy)
    host: str = "127.0.0.1"
    port: int = 8765
    auth_token_env: str | None = None
    compile_concurrency: int = 4  # parallel model calls across different user_ids
    # Search reads on top of raw-message retrieval (the P1 baseline protocol,
    # raw_arm.py); off = the older Block-retrieval base list.
    raw_base: bool = False
    append_options: bool = True  # multiple-choice options appended to the query (P1)
    # A request the compiler endpoint permanently rejects (HTTP 4xx, e.g. content
    # moderation) keeps its raw evidence but yields no Blocks instead of failing Add.
    skip_rejected_compiles: bool = False
    # existing lines shown to the compiler per batch (path_a_lines.select_known_lines)
    line_policy: dict[str, int] = field(
        default_factory=lambda: {"recent": 10, "cued": 10, "due": 10})
    # LCE engine: "flat" = Thread handoff into per-Thread Baselines (older);
    # "tree" = Thread + Path A Line trees ported from LCE (design v6 4.4-4.5).
    path_a_engine: str = "flat"
    # committed steps shown to the tree compiler per batch
    step_policy: dict[str, int] = field(default_factory=lambda: {"recent": 10, "cued": 10})

    def __post_init__(self) -> None:
        if self.path_a_engine not in ("flat", "tree"):
            raise ValueError("path_a_engine must be 'flat' or 'tree'")
        if self.search_mode not in ("path_a", "baseline"):
            raise ValueError("search_mode must be 'path_a' or 'baseline'")
        if self.hyde not in ("off", "fallback", "augmented"):
            raise ValueError("hyde must be off, fallback or augmented")
        if self.compiler.get("type") not in ("command", "openai"):
            raise ValueError("compiler.type must be 'command' or 'openai'")
        if type(self.compile_concurrency) is not int or not 1 <= self.compile_concurrency <= 64:
            raise ValueError("compile_concurrency must be an integer in [1, 64]")
        if self.embedding and self.embedding.get("type") not in ("fastembed", "openai", "gemini", "ark"):
            raise ValueError("embedding.type must be fastembed, openai, gemini or ark")
        if type(self.context_messages) is not int or not 0 <= self.context_messages <= 200:
            raise ValueError("context_messages must be an integer in [0, 200]")

    @classmethod
    def load(cls, path: Path | str) -> RuntimeConfig:
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        data_dir = Path(data.pop("data_dir"))
        if not data_dir.is_absolute():
            data_dir = (source.parent / data_dir).resolve()
        policy = ReadPolicy(**data.pop("read_policy", {}))
        return cls(data_dir=data_dir, read_policy=policy, **data)
