# AML 提交：MR-Mem + LCE（本地准备，2026-10-31）

部署和公网接入由用户那边负责；这里是本地已经准备好、验证过的东西，推到远端照着起服务就行。

## 起服务

```bash
pip install -e .[lce]
export ARK_API_KEY=...          # 方舟（豆包编译、豆包向量）
export MRMEM_AML_TOKEN=...      # Add/Search 的鉴权 token（Bearer / Token / X-Api-Key 都认）
mr-mem-aml check --config examples/runtime.aml-lce.json   # 先自检编译模型和向量
mr-mem-aml serve --config examples/runtime.aml-lce.json
```

`examples/runtime.aml-lce.json` 里要按远端改的：`data_dir`、`embedding.cache_path`、`host`（公网要 `0.0.0.0`）、`port`、`compile_concurrency`。

## 配置里定下来的东西

| 项 | 值 | 为什么 |
|---|---|---|
| 编译模型 | 方舟 `doubao-seed-2-1-pro`，关闭深度思考 | 所有验证都在它上面做的；换模型要重新验证 |
| 向量 | 方舟 `doubao-embedding-vision`（2048 维） | 同上；P1 融合检索和 LCE 共用 |
| `keep_raw_on_compile_failure` | true | 编译失败也让 Add 返回 200，原文照样可检索（退回 P1） |
| `compile_before_search` | true | 平台不会调 flush；Search 前先把这个用户还没收口的尾巴窗口编完（Search 最多 30 分钟） |
| `compile_window` | 软 4 万 / 硬 8 万字符，自然断点收口 | 窗口收口的那次 Add 会等编译完成再返回（1–3 分钟） |
| `line_read` | relay（原文命中 → SB → 沿承接取局部 + 精确原文片段），助手部分 `told` | 见下面的验证；段承接（`carry: threads`）和融合（`raw_fill`）在定版编译上没有帮助，保留为开关、默认关 |

## 验证（holdout beam_100k 60 题，DeepSeek 答题，每组两遍）

| 组 | 平均 | 每题字符 |
|---|---|---|
| P1（三算法融合检索，原文） | 47.2 | 4.8 万 |
| LCE + 融合（v11 编译） | 56.3 | 4.8 万 |
| LCE 用段承接 + 融合（v13 编译） | 54.2 | 5.0 万 |
| **最终配置（v14 编译 + relay 读出）** | **58.0**（56.9 / 59.1） | 2.0 万 |

pm2_32k（33 题）：和 P1 打平（56.1 / 57.6）。

## 本地契约自测

`mrmem-eval/local/smoke_local.py <runtime.json> [port]`：本地起服务，按官方格式发两块 Add、一次 Search，检查 success、三个 ID 原样回显、`data` 数组不超过 top_k、每条有 id 和 content、别的 user_id 查不到。2026-10-31 通过：Add 0.8–1.4 秒，Search 30 秒（含尾巴补编）。

## 已知限制

- 人与人聊天（LoCoMo）的编译场景没做，按"人和助手"编；编得不好时融合会补原文，接近 P1。
- 正式题库的编译总时长和费用取决于题库总量和并行度，按本地实测大约每路每分钟 5 万字符。
- 答题模型由平台锁定（官方脚本默认 Qwen3-14B），本地用的是 DeepSeek；弱答题模型下的效果待测。
