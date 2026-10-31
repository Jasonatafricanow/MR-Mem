# AML 提交：MR-Mem + LCE（本地准备，2026-10-31）

部署和公网接入由用户那边负责；这里是本地已经准备好、验证过的东西，推到远端照着起服务就行。

## 起服务

```bash
pip install -e .[lce]
export QWEN_PLAN_API_KEY=...    # 阿里百炼 Token Plan（deepseek-v4.1-flash-1017 编译）
export DASHSCOPE_API_KEY=...    # 阿里百炼（qwen3.7-text-embedding-flash 向量）
export MRMEM_AML_TOKEN=...      # Add/Search 的鉴权 token（Bearer / Token / X-Api-Key 都认）
mr-mem-aml check --config examples/runtime.aml-lce.json   # 先自检编译模型和向量
mr-mem-aml serve --config examples/runtime.aml-lce.json
```

`examples/runtime.aml-lce.json` 里要按远端改的：`data_dir`、`embedding.cache_path`（放在持久盘上）、`host`（公网要 `0.0.0.0`）、`port`、`compile_concurrency`。

## 配置里定下来的东西

| 项 | 值 | 为什么 |
|---|---|---|
| 编译模型 | 阿里百炼 Token Plan `deepseek-v4.1-flash-1017`，`enable_thinking: false`（`QWEN_PLAN_API_KEY`） | 用户 10-31 定。同一段 200 条消息对比 qwen3.8-flash：第一步 JSON 读不出 1/13 对 9/21，编译 20 分钟对 37 分钟，20 题两遍 60.8 对 55.3（豆包 v14 53.9，P1 47.6）。方舟豆包额度已用完 |
| 向量 | 阿里百炼 `qwen3.7-text-embedding-flash`（1024 维，`DASHSCOPE_API_KEY`），`cache_path` 本地缓存 | 用户 10-31 定：四种向量的 P1 召回差距不大（1–2 题以内），它便宜得多；P1 融合检索和 LCE 共用。接口同一文本两次返回的向量不完全一样（余弦约 0.998），缓存让重启重建索引时不重付费、检索不抖动 |
| `keep_raw_on_compile_failure` | true | 编译失败也让 Add 返回 200，原文照样可检索（退回 P1） |
| `compile_before_search` | true | 平台不会调 flush；Search 前先把这个用户还没收口的尾巴窗口编完（Search 最多 30 分钟） |
| `compile_window` | 软 4 万 / 硬 8 万字符，自然断点收口 | 窗口收口的那次 Add 会等编译完成再返回（1–3 分钟） |
| `decision` | `decision-model-preview`（System One），走 Token Plan（`QWEN_PLAN_API_KEY`，和编译同一个 key），Search 时旁路判断，读出最前面加一条说明 | 用户 10-31 定。两条规则（mr_mem_aml/decision.py）：问"我有没有做过 X"，而人自己的命中消息里既说做过、又说从没做过 → 引用两句原话；问人自己的某个具体信息，而人从没说过（只在助手的建议里，或根本没有）→ 说明这一点。只判断不改证据，不写进编译产物；调用失败就不加。holdout 60 题两遍：52.6 → 58.5（矛盾 8 → 50，拒答 50 → 67）。每次 Search 1 次分流调用，是否题再加至多 30 次、具体信息题再加 1 次；全进程限 16 路并发。要关掉就删掉这一项 |
| 编译流程 | `step1: topic`，`layer2: stretches`：每窗 2 次调用（SB 一次、段一次） | 定义见 docs/LCE_FROZEN.md：SB 是在聊的那件事，段摘要按 前因-谁-何时-做了什么-后果 写一次 |
| `line_read` | relay（原文命中 → SB → Line 上的局部，只带检索命中的原文），助手部分 `told`，全部段摘要（`thread_k: 60`、`thread_sbs: 0`），`marks: false`，`raw_fill: 0`，`hit_anchors: true` | 用户 10-31 定：不补位，证据已够（原文 → SB → Line、段摘要 → SB → 原文两条路）；晨昏线标记只在记忆侧用，不给答题模型看；每条原文命中都带出它的 SB，不受 `sbs` 上限截断（原来 1800 条前 30 命中里丢 208 条）。holdout 60 题两遍：P1 47.2，不补位 50.3，修正时间 + 决策旁路 58.5，再加 `hit_anchors` 60.1（时间推理两遍 33 → 50）；每题约 8.5 万字符 |

## 验证（holdout beam_100k 60 题，DeepSeek 答题，每组两遍）

| 组 | 平均 | 每题字符 |
|---|---|---|
| P1（三算法融合检索，原文） | 47.2 | 4.8 万 |
| LCE + 融合（v11 编译） | 56.3 | 4.8 万 |
| LCE 用段承接 + 融合（v13 编译） | 54.2 | 5.0 万 |
| v14 编译 + relay 读出（10-30，豆包向量） | 58.0（56.9 / 59.1） | 2.0 万 |

10-31 换成 qwen3.7 flash 向量（带缓存）和 ds1017 编译后重测，同一份缓存向量：P1 47.2；v14 + 补位 50.2；**最终配置（topic + 段，全部段摘要，不补位）50.3**（51.4 / 49.2），编译 237 秒、68 次调用。

pm2_32k（33 题）：和 P1 打平（56.1 / 57.6）。

## 本地契约自测

`mrmem-eval/local/smoke_local.py <runtime.json> [port]`：本地起服务，按官方格式发两块 Add、一次 Search，检查 success、三个 ID 原样回显、`data` 数组不超过 top_k、每条有 id 和 content、别的 user_id 查不到。2026-10-31 通过：Add 0.8–1.4 秒，Search 30 秒（含尾巴补编）。

## 已知限制

- 人与人聊天（LoCoMo）：没有助手、且消息开头能认出至少两个说话人名字时，第一步用人与人模板（一段话题来回一个 SB，用名字写谁对谁说）。平台若把一方当 assistant 发来，仍按人和助手编。
- 人在消息里贴整段文档、群聊导出时，第一步可能逐行切、撞 1.6 万输出上限，重试 3 次后该窗只留原文（退回 P1）；费 token，不影响可检索。
- 正式题库的编译总时长和费用取决于题库总量和并行度。ds1017 实测：单路约每分钟 2–3 万历史字符，长历史每个历史字符约 1.2 个输入 token（beam_1m 422 万字符：3 小时、519 万输入 + 87 万输出 token）。
- 答题模型由平台锁定（官方脚本默认 Qwen3-14B），本地用的是 DeepSeek；弱答题模型下的效果待测。
