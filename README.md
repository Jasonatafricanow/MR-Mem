# MR-Mem

[中文说明](README.zh-CN.md)

[![quality](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/quality.yml)
[![privacy](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/privacy.yml/badge.svg?branch=main)](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/privacy.yml)

**A memory system that keeps what was said as the authority and what it understood as a
traceable projection.** MR-Mem stores every message as raw evidence, compiles the
conversation into semantic blocks (**SB**s — one thing being talked about), links the blocks
into **Lines** (how a matter moved on) and cuts Lines into **stretches**, each summarized
once as cause → who → when → what → outcome. A question is answered from that structure:
raw hits lead to their SBs, the SBs to their stretches, and each SB brings back the exact
words it was compiled from. Nothing the compiler wrote replaces the source; every projection
can be followed back to it.

```text
raw messages ──BM25 + vectors (RRF)──▶ raw evidence      (P1 baseline)
   │ compile in windows, closing at natural breakpoints
   ▼
SB: one matter talked about ── carries pointers to its words
   │ links: "this step carries that one on"
   ▼
Line ──▶ stretches with cause → who → when → what → outcome   (LCE)
```

| | |
|---|---|
| **Where it was tested** | AML 2026 Round 2, Full track (official leaderboard 2026-11-19: overall **59.78**, rank 26; memory governance D1 and D2 both tied 4th; see [below](#aml-2026-results-and-cost)), then public BEAM, PersonaMem-v2 and CLBench sets locally |
| **What is on `main`** | The submitted version plus the post-competition changes measured as gains: SBs carried with their linked steps, message times written into the readout, and precise readout (only the SBs the search found carry their words; dev set 37.1 → 43.1, stable over two answer passes) |
| **How it got here** | [Research log](docs/DEVLOG.md) — what was built, what was dropped and why |
| **Design** | [Frozen definitions](docs/LCE_FROZEN.md) · [LCE design](docs/LCE_DESIGN.md) · [technical report](docs/TECHNICAL_REPORT.md) · [configuration guide](docs/CONFIG_GUIDE.md) |

| Branch | What it holds |
|---|---|
| `main` | The version above |
| `feat/aml-20261013` | The AML submission, frozen |
| `dev` | Ongoing work; experimental switches default off |

Scores are development measurements on public data and one official run, not independent
validation.

## AML 2026: results and cost

AML compares memory systems only: the platform writes conversations through Add, calls
Search per question, and answers and grades with a model and prompt it locks. Every entry
answers with the same model; what differs is the material the memory hands over, and what
it costs to produce it.

**Official leaderboard (Full track, published 2026-11-19)**

| | Score | Position |
|---|---|---|
| Overall | **59.78** | 26th; 5th place is 62.28, 2.50 ahead |
| A Explicit fact recall | 61.93 | |
| B Relations and multi-hop reasoning | 57.68 | |
| C Time and event sequence | 56.59 | |
| D1 New value overrides old | 77.6 | tied 4th |
| D2 Contradiction resolution | 76.4 | tied 4th |

The evaluation ended on 11-04 and the leaderboard came out on 11-19 after the platform
re-graded. The preliminary average shown when our run finished on 11-02 was 54.37
([technical report §4.1](docs/TECHNICAL_REPORT.md), in Chinese); the leaderboard is the
final figure.

**Cost**

| | Submitted configuration |
|---|---|
| Suite size | Textual Memory track: 6,000+ evaluation instances, about 300M tokens ([official description](https://github.com/AML-memory/agent-memory-leaderboard)) |
| Compiler | DeepSeek v4.1 flash (1017), 2 calls per window |
| Embeddings | qwen3.7-text-embedding-flash, cached locally |
| Spend | compiler about ¥130 + embeddings about ¥5 ≈ **¥135** (about ¥0.45 per million tokens of the suite) |
| Run | a free cloud instance (7.8 GB RAM), 31 h 39 min |

The estimate during development was ¥3,000–5,000. The saving comes from the structure,
not from doing less:

- **What was understood once is not understood again.** The understanding is kept at
  compile time (SBs, stretch summaries); Search hands over those products instead of having
  a model reread the raw text for every question.
- **An SB carries the gist; details stay in the source.** An SB says who did what about
  which matter; numbers, code and steps are not copied in but reached by index. Compiler
  output stays short (BEAM held-out set: 660k input tokens produce 40k output tokens).
- **Windows close at natural breakpoints, two calls each.** One call cuts SBs; one cuts
  stretches and writes each closed stretch's summary once, never rewritten.
- **Search calls no compiler model.** Readout is retrieval plus following the index;
  embeddings are cached by text hash, so restarts and repeated text cost nothing.

**What it cost, and the fix after the run**

The cheap compiler left marks: stretch summaries sometimes carry an old plan's numbers,
and knowledge-update questions pick the old value. But the post-mortem found the main loss
on explicit recall was in the readout: nine in ten gold evidence items reached the readout,
only four in ten as the original words; the rest arrived only as an SB's gist (the
submitted readout gave only the raw hits, and SBs reached through Lines or stretches
carried no source words, [technical report §5.2](docs/TECHNICAL_REPORT.md)). After the run,
SBs return with the words they point to. The compilation is unchanged and the cost barely
moves:

| Local public data (no evaluation data used) | Submitted | After | P1, top 100 raw |
|---|---|---|---|
| BEAM 100K held-out: gold evidence present as original words | 51.8% | **89.4%** | 79.4% |
| PersonaMem-v2 32K: same | 65.9% | **98.2%** | |

This is evidence recall, not answer score. The answer-score gain is still being measured:
precise readout moved the dev set 37.1 → 43.1 (stable over two answer passes); the other
changes stayed within answer noise ([research log](docs/DEVLOG.md)).

**Reading the result.** 59.78 is 2.5 points from the top five, and both governance
sub-scores are among the leaders; the spend is a public, reproducible number. Most entries
have not published their cost, so no cost-effectiveness optimum is claimed here — just one
point others can compare against: the Full track for ¥135, with code and configuration on
[`feat/aml-20261013`](https://github.com/Jasonatafricanow/MR-Mem/tree/feat/aml-20261013).
Re-runs are welcome, and so are other teams' cost figures.


## Start here

Python 3.12+ and Git are required. From the repository root:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[aml,lce]"
python scripts/smoke_offline.py
```

Installation downloads dependencies. The smoke test uses synthetic messages, temporary
storage and loopback HTTP only: **no model calls or model charges**. It checks Add, Search
and user isolation, not semantic quality. Core-only users can install with `pip install -e .`.

To connect real providers, copy `examples/runtime.aml-lce.json` to `runtime.json`, edit the
endpoints/models for your account and set the named environment variables. `.env.example`
lists names only; `.env` files are not loaded automatically. Set `MRMEM_AML_TOKEN` to your
own service access token, then run `mr-mem-aml serve --config runtime.json`. Clients use
`Authorization: Bearer <service-token>`. Default binding is loopback.

Provider-backed Add/Search and `check` may incur charges. `check --check-decision` also
probes the optional decision provider. `/health` itself is passive. Decision failures and
the default 15-second total deadline leave the ordinary retrieval response available.
Remove `decision` from the configuration to disable that provider.

See [deployment](docs/AML_SUBMIT_LCE.md), [HTTP/configuration](docs/AML.md),
[frozen definitions](docs/LCE_FROZEN.md), and [privacy](SECURITY.md).
Reported benchmark scores are development measurements, not independent validation.

For development: `pip install -e ".[dev,aml,lce]"`, then `ruff check src tests`,
`python -m pytest tests`, and `python scripts/smoke_offline.py`.

## Repository map

| Path | What it is |
|---|---|
| `src/mr_mem/memory` | MemoryCore: canonical semantic memory, admission through SourceRefs, BM25 / vector / RRF retrieval, Threads |
| `src/mr_mem/integrations`, `src/mr_mem/lce` | Path A: mature Threads and complete cognition handed to LCE Lines; one unified read path |
| `src/mr_mem_aml` | The AML service: raw store, compiler (SBs, stretches), Line tree, readout (`line_read.py`), HTTP server |
| `tests/` | Contract tests for the above |
| `docs/` | Design, decisions, reports — start from the [documentation map](docs/README.md) |

## Architecture

MR-Mem is a sidecar: when a host already keeps the conversation, the host's store stays the
authority for the raw text and MR-Mem keeps only what it compiled, with references back.
Standalone (as in AML) it keeps the raw messages itself. The boundaries — what MR-Mem owns,
how sources are admitted, how Thread and LCE consume the same semantic substrate — are in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Path A is described in
[docs/PATH_A.md](docs/PATH_A.md), the AML adapter in [docs/AML.md](docs/AML.md), server
deployment in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Status

Research code by one author, developed in the open. `main` holds the measured version;
`dev` carries work in progress behind switches that default off. What was tried, kept and
dropped is in the [research log](docs/DEVLOG.md). No license has been chosen yet.
