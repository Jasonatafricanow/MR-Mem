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
    # the completion backend HyDE writes its hypothetical passage with; None: the compiler's
    hyde_backend: dict | None = None
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
    # Any compile that still fails after its retries keeps its raw evidence (searchable as P1 is)
    # and
    # yields no Blocks, so Add answers 200 as the AML contract requires; counted, never silent.
    keep_raw_on_compile_failure: bool = False
    # Search first compiles what is left of that user's history (the tail window an Add has not
    # closed);
    # AML never calls flush, and Search may run 30 minutes.
    compile_before_search: bool = False
    # existing lines shown to the compiler per batch (path_a_lines.select_known_lines)
    line_policy: dict[str, int] = field(
        default_factory=lambda: {"recent": 10, "cued": 10, "due": 10}
    )
    # LCE engine: "flat" = Thread handoff into per-Thread Baselines (older);
    # "tree" = Thread + Path A Line trees ported from LCE (design v6 4.4-4.5).
    path_a_engine: str = "flat"
    # committed steps shown to the tree compiler per batch
    step_policy: dict[str, int] = field(default_factory=lambda: {"recent": 10, "cued": 10})
    # Tree engine: compile in windows of our own instead of per Add (design 4.7):
    # {"soft": n, "hard": n, "measure": "messages" | "chars", "gap_minutes": a pause that
    # ends a stretch}; "chars" counts the text, as message lengths differ a hundredfold.
    # Add stores raw text at once; a window closes at the first natural breakpoint
    # after "soft", where a block is complete (session change, long pause, and the
    # turn boundary of the scene), at "hard" at the latest; flush compiles what is left.
    compile_window: dict[str, int | str] | None = None
    # Conversation scene (design 4.1, 4.7): "chat" = the interaction is the content,
    # a turn ends when the person speaks again after the assistant; "work" = a block
    # is the assistant's report and the person's feedback on it, so a turn ends when
    # the assistant speaks again after the person.
    scene: str = "chat"
    # Tree engine placement: "lines" = blocks placed on lines judged by their
    # summaries; "points" = each block linked to the earlier steps it carries on,
    # lines are what the links make, and no summary takes part in placement;
    # "entries" = placed like "points", but the model only writes per-message
    # entries and code assembles the blocks and links (trial, compiler entries.py);
    # "moves" = work scene, two steps per window: the model writes the SBs ("what
    # happened -> what the person did"), then places them among candidates retrieved
    # with each SB's own text (compiler moves.py, user 2026-10-24).
    placement: str = "lines"
    # placement "moves", step 2: per SB the top "k" earlier SBs by each of four
    # searches, plus the "latest" SBs before the window
    place_policy: dict[str, int] = field(default_factory=lambda: {"k": 10, "latest": 15})
    # placement "moves", step 1: "moves" = one SB per point of the person; "slim4" = one SB
    # per stretch with one result, as the bbz_diag slim4 prompt (trial, 2026-10-26, Chinese only)
    step1: str = "moves"
    # placement "moves", after step 1: "steps" = step 2's links, then the threads call and the
    # summaries call; "stretches" = one call cutting the SBs into stretches and summarizing the
    # closed ones (docs/LCE_FROZEN.md, 2026-10-31)
    layer2: str = "steps"
    # search in path_a mode reads by the upward index (line_read.py, design 5.3, 5.5): raw hits ->
    # their SBs -> the Line around them, plus raw snippets; None keeps the PathAReader read.
    # Keys of line_read.LineReadPolicy (sbs, up, down, raw_k, sb_k, rrf_k, tz_minutes, raw_full)
    line_read: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.path_a_engine not in ("flat", "tree"):
            raise ValueError("path_a_engine must be 'flat' or 'tree'")
        if self.placement not in ("lines", "points", "entries", "moves"):
            raise ValueError("placement must be 'lines', 'points', 'entries' or 'moves'")
        if self.placement == "moves" and (
            (self.scene != "work" and self.step1 not in ("slim4", "topic"))
            or self.compile_window is None
        ):
            raise ValueError(
                "placement 'moves' compiles in windows, in the work scene unless step1 is 'slim4'"
            )
        if self.scene not in ("chat", "work"):
            raise ValueError("scene must be 'chat' or 'work'")
        if self.step1 not in ("moves", "slim4", "topic"):
            raise ValueError("step1 must be 'moves', 'slim4' or 'topic'")
        if self.layer2 not in ("steps", "stretches"):
            raise ValueError("layer2 must be 'steps' or 'stretches'")
        if self.line_read is not None:
            from mr_mem_aml.line_read import LineReadPolicy

            LineReadPolicy.of(self.line_read)
            if not (
                self.raw_base
                and self.path_a_engine == "tree"
                and self.placement in ("points", "moves")
            ):
                raise ValueError(
                    "line_read needs raw_base and the tree engine's Line graph (points or moves)"
                )
        if self.compile_window is not None:
            w = self.compile_window
            if self.path_a_engine != "tree":
                raise ValueError("compile_window needs path_a_engine 'tree'")
            if not 1 <= w.get("soft", 0) <= w.get("hard", 0):
                raise ValueError("compile_window needs 1 <= soft <= hard")
            if w.get("measure", "messages") not in ("messages", "chars"):
                raise ValueError("compile_window measure must be 'messages' or 'chars'")
        if self.search_mode not in ("path_a", "baseline"):
            raise ValueError("search_mode must be 'path_a' or 'baseline'")
        if self.hyde not in ("off", "fallback", "augmented"):
            raise ValueError("hyde must be off, fallback or augmented")
        if self.compiler.get("type") not in ("command", "openai"):
            raise ValueError("compiler.type must be 'command' or 'openai'")
        if type(self.compile_concurrency) is not int or not 1 <= self.compile_concurrency <= 64:
            raise ValueError("compile_concurrency must be an integer in [1, 64]")
        if self.embedding and self.embedding.get("type") not in (
            "fastembed",
            "openai",
            "gemini",
            "ark",
        ):
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
