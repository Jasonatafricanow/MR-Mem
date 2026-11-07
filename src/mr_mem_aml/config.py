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
    context_messages: int = 20
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
    # Compile in windows of our own, not per Add (design 4.7): {"soft": n, "hard": n,
    # "measure": "messages" | "chars", "gap_minutes": a pause that ends a stretch,
    # "max_messages", "max_chars_per_user"}; "chars" counts the text, as message lengths
    # differ a hundredfold. Add stores raw text at once; a window closes at the first natural
    # breakpoint after "soft", where an SB is complete (session change, long pause, and the
    # turn boundary of the scene), at "hard" at the latest; flush compiles what is left.
    compile_window: dict[str, int | str] = field(
        default_factory=lambda: {"soft": 40000, "hard": 80000, "measure": "chars"}
    )
    # Conversation scene (design 4.1, 4.7): "chat" = the interaction is the content,
    # a turn ends when the person speaks again after the assistant; "work" = a block
    # is the assistant's report and the person's feedback on it, so a turn ends when
    # the assistant speaks again after the person.
    scene: str = "chat"
    # search in path_a mode reads by the upward index (line_read.py, design 5.3, 5.5): raw hits ->
    # their SBs -> the Line around them, plus raw snippets; None: the PathAReader read, which
    # gives raw hits and SBs without the Lines.
    # Keys of line_read.LineReadPolicy (sbs, up, down, raw_k, sb_k, rrf_k, tz_minutes, raw_full)
    line_read: dict[str, int] | None = None
    # windows compiled in the background, not inside Add: {"search_wait": seconds a Search waits
    # for the user's compiling to go idle before it answers with what is there (15), "idle_tail":
    # seconds without an Add before the tail window is compiled too (30), "search_closes_tail":
    # false: a Search only waits for the compiling under way, it does not compile the tail}. For a
    # gateway that
    # cuts requests short (InstaBox: about 20 seconds); None: Add compiles, Search flushes first
    background_compile: dict[str, float] | None = None
    # what step 1 reads ({"assistant_head": n}: each assistant message to n characters, a code
    # block in it to its first 200; search and readout keep the raw text whole). Evaluation
    # ingests whole histories at once, where nothing is cached: compile cost is every token read
    # {"blocks": true}: step 1 says which paragraphs of a message an SB comes from, as character
    # spans of the raw text (docs/LCE_DESIGN.md 5.5); assistant_head then cuts each paragraph, not
    # the message, so that the numbers hold
    compile_input: dict[str, int] | None = None
    # Search, path_a with line_read: a decision model judges the question over its raw hits and may
    # put one note in front (decision.py: the person's own contradicting statements, or a detail
    # the person never stated). {"base_url" | "base_url_env", "api_key_env", "model", "timeout",
    # "workers", "hits"}; None: off
    decision: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.scene not in ("chat", "work"):
            raise ValueError("scene must be 'chat' or 'work'")
        if self.line_read is not None:
            from mr_mem_aml.line_read import LineReadPolicy

            LineReadPolicy.of(self.line_read)
            if not self.raw_base:
                raise ValueError("line_read needs raw_base")
        if self.decision is not None and (
            self.line_read is None
            or not (self.decision.get("base_url") or self.decision.get("base_url_env"))
        ):
            raise ValueError("decision needs line_read and a base_url or base_url_env")
        if self.background_compile is not None:
            b = self.background_compile
            if not all(
                isinstance(b.get(k, 1), int | float) and b.get(k, 1) >= 0
                for k in ("search_wait", "idle_tail")
            ):
                raise ValueError("background_compile search_wait and idle_tail must be >= 0")
        w = self.compile_window
        if not 1 <= w.get("soft", 0) <= w.get("hard", 0):
            raise ValueError("compile_window needs 1 <= soft <= hard")
        if w.get("measure", "messages") not in ("messages", "chars"):
            raise ValueError("compile_window measure must be 'messages' or 'chars'")
        if "max_chars_per_user" in w and not (
            type(w["max_chars_per_user"]) is int and w["max_chars_per_user"] >= 1
        ):
            raise ValueError("compile_window max_chars_per_user must be an integer >= 1")
        if "max_messages" in w and not (type(w["max_messages"]) is int and w["max_messages"] >= 1):
            raise ValueError("compile_window max_messages must be an integer >= 1")
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
