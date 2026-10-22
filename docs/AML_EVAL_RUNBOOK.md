# MR-Mem Path A 对照评测执行指南

> 适用分支：`feat/aml-20261013`（AML 专属）。本指南写给执行评测的 agent，按顺序执行，遇到"停止条件"必须停下来向用户报告，不要自行变通。

## 0. 目标与原则

**要回答的问题：** LCE（Path A）在开放数据集上，相对 MR-Mem 自身检索基线，有没有真实增益。

**唯一的变量是读取方式。** 数据集只 Add（编译）一次，产生一个 `data_dir`；之后四组对照都读这同一个 `data_dir`，只换读取配置：

| 组 | `search_mode` | `read_policy.source_text_for_hits` | 含义 |
|---|---|---|---|
| **A** | `baseline` | `false` | 基线：只有 Canonical 检索结果 |
| **C0** | `path_a` | `false` | 加上 LCE 认知和 Thread（它们自带支撑 Block 和原文），普通命中不带原文 |
| **A+** | `baseline` | `true` | 基线 + 每条命中附带原文 |
| **C** | `path_a` | `true` | Path A + 每条命中附带原文 |

**主结论看 C 对 A+。** C 和 A+ 都给命中附带原文，差别只剩 LCE 认知和 Thread。
- C 显著好于 A+：LCE 有增益。
- A+ 好于 A，但 C ≈ A+：提升来自原文证据，不是 LCE。
- C0 对 A 只作参考：两者差别里混着"原文"和"LCE"两个因素。

**禁止事项（任何一条被违反，结果作废）：**
1. 四组之间不得修改代码、提示词、`ReadPolicy` 其他字段、embedding、HyDE、`top_k`、回答模型、评审模型、评审提示词。
2. 不得为任何一组重新编译。`data_dir` 在 Add 完成后只读。
3. 不得同时运行两个服务指向同一个 `data_dir`。
4. 不得更换或新增 embedding 模型、不得新增依赖。需要时停止并问用户。
5. 数据集、评测结果、`data_dir` 只放在评测服务器本地，不得提交到任何 git 仓库、不得上传。
6. 不得推送分支、不得开 PR。

## 1. 环境准备

下文命令以 Linux shell 为准。`$REPO` 是 MR-Mem 仓库检出目录，`$RUN` 是本次评测的运行目录，必须在仓库外，例如 `~/mrmem-eval-runs/<YYYYMMDD>-<数据集名>`。Windows 上把 `.venv/bin/` 换成 `.venv\Scripts\`，路径分隔符换成 `\`。

需要 Python ≥ 3.12 和 git（安装 LCE 依赖时要从 GitHub 拉取）。

```bash
git clone https://github.com/Jasonatafricanow/MR-Mem.git "$REPO"
cd "$REPO" && git checkout feat/aml-20261013 && git status --short
python3.12 -m venv .venv && . .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[lce]"          # 基线用了向量召回时改为 ".[lce,vector]"
python -m pytest -q                        # 必须全部通过；失败就停止并报告
```

已经检出过的，用 `git fetch && git checkout feat/aml-20261013 && git pull` 更新到最新。`git status --short` 必须为空。之后所有命令都在激活了 `.venv` 的 shell 里执行。

**编译器。** 使用用户指定的 LLM，通过 OpenAI 兼容端点接入（`compiler.type = "openai"`，支持的服务和写法见 [AML.md](AML.md) 第 4 节）。**用户没有指定用哪个 LLM、端点或 API key 环境变量时，停止并询问。** 不要用 `agy`、`grok` 这类命令行 agent 跑数据集：它们每次调用都有十几秒固定启动开销，小测试集就要十几个小时。一个数据集从头到尾只能用同一个编译器和模型。

**向量模型。** 必须和用户基线（D0）完全一致：
- 基线用的是 MR-Mem 本地 FastEmbed（如 `BAAI/bge-small-en-v1.5`）时，配置 `{"type": "fastembed", "model": <同一模型名>, "dimension": <同一维度>, "model_dir": <模型文件目录>}`。优先直接用基线所用的那份模型文件目录；没有的话用 `mr-mem-aml fetch-model --model <模型名> --dir <目录>` 下载。
- 基线用的是某个 HTTP 向量服务时，用 `{"type": "openai", ...}` 指向同一个服务和模型。
- 基线没用向量时，`"embedding": null`。

记录以下版本信息，写进最终报告：
- `git -C "$REPO" rev-parse HEAD`
- `src/mr_mem_aml/compiler.py` 里的 `PROMPT_VERSION`
- 编译器端点与模型（例如 `https://api.openai.com/v1` + `gpt-4o-mini`），`compile_concurrency`
- embedding 模型与维度（或"无，仅 BM25"）、HyDE 设置
- 服务器操作系统、Python 版本

## 2. 配置文件

在 `$RUN` 下建 5 个配置文件。它们**除 `search_mode` 和 `read_policy` 外必须完全相同**。

**第一步：确定基线设置。** 向用户确认（或从用户之前跑基线 D0 的记录里读出）：embedding 用的是什么（类型、模型、维度、模型文件或服务地址），HyDE 是 `off` / `fallback` / `augmented`。**确认不了就停止，问用户。** 设置不一致时，A 组就不等于用户已有的基线。

`$RUN/ingest.json`（以 gpt-4o-mini + 本地 bge-small 为例；`compiler` 换成用户指定的 LLM，`embedding` 和 `hyde` 按基线设置）：

```json
{
  "data_dir": "data",
  "compiler": {
    "type": "openai",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini",
    "api_key_env": "OPENAI_API_KEY",
    "json_mode": true,
    "timeout": 300
  },
  "embedding": {
    "type": "fastembed",
    "model": "BAAI/bge-small-en-v1.5",
    "dimension": 384,
    "model_dir": "/path/to/the/baseline/bge-small-model-files"
  },
  "hyde": "off",
  "compile_concurrency": 8,
  "search_mode": "path_a",
  "lce_enabled": true,
  "assistant_policy": "attributed",
  "context_messages": 20,
  "read_policy": {"source_text_for_hits": false},
  "host": "127.0.0.1",
  "port": 8765
}
```

再复制出 4 个组配置，只改两处：

| 文件 | `search_mode` | `read_policy` |
|---|---|---|
| `$RUN/arm-A.json` | `"baseline"` | `{"source_text_for_hits": false}` |
| `$RUN/arm-C0.json` | `"path_a"` | `{"source_text_for_hits": false}` |
| `$RUN/arm-Aplus.json` | `"baseline"` | `{"source_text_for_hits": true}` |
| `$RUN/arm-C.json` | `"path_a"` | `{"source_text_for_hits": true}` |

`data_dir` 是相对路径，以配置文件所在目录为基准，所以 5 个文件都指向同一个 `$RUN/data`。

预检：

```bash
mr-mem-aml check --config $RUN/ingest.json
```

`lce` 必须是 `importable`，`compiler` 必须以 `ok` 开头（括号里是一次最简调用的耗时，记进报告），配置了 embedding 时必须是 `ok (N dims)` 且维度正确。任何一项失败：停止，报告原文输出。

## 3. 测试集

使用用户提供的测试集：从几个开放数据集里抽出的小型混合集，用来模拟 AML。**整套直接跑完，不再抽子集，也不需要中途等批准。**

- 这个小测试集的作用是判断 LCE 有没有增量：有增量才考虑跑全量，没有就不跑。所以报告里的结论要如实反映置信区间，不要因为样本小而夸大或回避结论。
- `judged` 文件的 `category` 填"来源数据集/题型"（例如 `longmemeval/temporal-reasoning`），这样比较结果能按来源拆开看。
- 记录每次 Add 的耗时，写进报告，供用户估算全量成本。

## 4. Add 阶段（编译一次）

1. 启动服务（前台占用一个终端，或作为后台进程，记下进程以便停止）：

   ```bash
   mr-mem-aml serve --config $RUN/ingest.json
   ```

   确认 `GET http://127.0.0.1:8765/health` 返回 200，且 `lce_enabled` 为 `true`。

2. 用用户现有的 AML 式测试管路发送全部 Add 请求到 `http://127.0.0.1:8765/add`。要求：
   - **不同用户可以并发，同一用户必须串行。** 管路并发数设为与 `compile_concurrency` 相同（例如 8）；同一 `user_id` 的分块按时间顺序发送，上一块返回 200 之后再发下一块。
   - **按 AML 标准分块，一次 Add 发一块，每块 20 条消息。** 具体做法：
     - 每个会话（`session_id`）单独分块，一块不跨会话；
     - 按消息顺序依次装入当前块，装满 20 条就封块发送；
     - 如果还没满 20 条，块内总词数就要超过 2,000，则在这条消息之前封块（AML 规则：20 条或 2,000 词，先到为准）；
     - 会话末尾不足 20 条的余数单独成一块；
     - 不得逐条发送，也不得把整个会话一次发完。块的大小决定每次编译看到的内容，必须和 AML 正式评测一致。
   - `timestamp` 用 Unix 毫秒，数据集里有时间就必须带上。
   - `user_id` 按数据集的隔离单位：每题独立历史的（LongMemEval）每题一个 `user_id`；共享对话的（LoCoMo）每段对话、每个说话人按用户现有管路的做法。**和基线跑法保持一致。**
   - `request_id` 必须稳定、可复现（例如 `<数据集>:<user_id>:<分块序号>`），不能用随机数。重跑时同一 `request_id` 会直接返回，不再调用模型。
3. 返回码处理：
   - `200`：成功（`success: true`，`replayed: true` 表示之前已完成）。
   - `503`：按 `Retry-After` 等待后用**同一个请求**重试；同一请求连续失败 5 次，记录下来，继续下一个。
   - `422`：同一 `request_id` 发了不同内容，说明管路有 bug。**停止，报告。**
   - `400`：请求格式错误。**停止，报告。**
4. 全部发完后，**把全部 Add 再完整发送一遍**。已完成的会立刻返回 `replayed: true`，之前失败的会续跑。重复直到没有失败为止（最多 3 遍；仍有失败就报告失败的 `request_id` 列表）。
5. 收尾：

   ```bash
   curl -s -X POST http://127.0.0.1:8765/admin/flush -H "Content-Type: application/json" -d "{}"
   ```

   然后 `GET /health`，记录 `thread_failures`（非 0 要写进报告）。**停止服务**，再统计：

   ```bash
   python "$REPO"/scripts/aml_eval/lce_stats.py $RUN/ingest.json
   ```

   保存输出到 `$RUN/lce_stats.json`。如果 `lce_lines` 为 0，说明编译器没产出任何认知线，后面的对照没有意义：**停止，报告。**

6. 快照：把 `$RUN/data` 整个复制为 `$RUN/data.snapshot`。之后只读。

## 5. Search 阶段（四组）

准备 `$RUN/questions.jsonl`，每行一个问题：

```json
{"id": "q_001", "query": "原始问题文本", "user_id": "与 Add 时一致", "options": ["选择题才有"]}
```

`query` 必须是数据集里的原始问题，不得改写、不得附加答案或提示。

对每一组，**依次**执行：启动该组配置的服务 → 跑检索 → 停止服务。四组用同一个端口，同一时间只有一个服务。

```bash
mr-mem-aml serve --config $RUN/arm-A.json
```
```bash
python "$REPO"/scripts/aml_eval/search_arm.py --arm A --mode baseline --questions $RUN/questions.jsonl --out $RUN/results/A.search.jsonl
```

停止服务后换下一组：

| 组 | 配置 | `--arm` | `--mode` | 输出 |
|---|---|---|---|---|
| A | `arm-A.json` | `A` | `baseline` | `results/A.search.jsonl` |
| C0 | `arm-C0.json` | `C0` | `path_a` | `results/C0.search.jsonl` |
| A+ | `arm-Aplus.json` | `A+` | `baseline` | `results/A+.search.jsonl` |
| C | `arm-C.json` | `C` | `path_a` | `results/C.search.jsonl` |

说明：
- `search_arm.py` 固定 `top_k=100`，可断点续跑（已写入的 `id` 会跳过）。
- 如果用户的测试管路自己调用 Search，也可以直接让管路连这个服务：组配置里已经写死了 `search_mode`，管路不需要发任何特殊请求头。此时要保证每组的检索结果按同样格式保存下来。
- 如果配置了 HyDE，每次检索会调用一次编译器模型，结果有随机性，四组之间会多一层噪声。在报告里注明。

## 6. 回答与评审

使用**和用户基线完全相同**的回答模型、回答提示词、评审模型、评审提示词和检索结果拼接方式。参考：AML 官方仓库 `AML-memory/agent-memory-leaderboard` 的 `data/<数据集>/pipeline.py`，按 `data` 的顺序把每条 `content` 拼成记忆文本（`created_at` 的用法与基线一致）。

输入：每组的 `results/<组>.search.jsonl` 里的 `data`。
输出：每组一个 `results/<组>.judged.jsonl`，每行至少包含：

```json
{"id": "q_001", "is_correct": true, "category": "题型（数据集提供时）"}
```

评分是连续值的数据集，用 `"score": 0.0~1.0` 代替 `is_correct`。四组必须覆盖同一批 `id`。

## 7. 比较

```bash
python "$REPO"/scripts/aml_eval/compare_arms.py --arm A=$RUN/results/A.judged.jsonl --arm C0=$RUN/results/C0.judged.jsonl --arm A+=$RUN/results/A+.judged.jsonl --arm C=$RUN/results/C.judged.jsonl --search A=$RUN/results/A.search.jsonl --search C0=$RUN/results/C0.search.jsonl --search A+=$RUN/results/A+.search.jsonl --search C=$RUN/results/C.search.jsonl --pair C:A+ --pair A+:A --pair C0:A --pair C:A --out $RUN/report.json
```

输出包括：
- 各组准确率和按题型的准确率；
- 每组配对差：均值、配对 bootstrap 95% 置信区间、胜/负/平、McNemar 精确检验 p 值；
- 上下文统计：平均条数、平均 token 估算、有多少题读到了 LCE 认知或 Thread、有多少题可能超过回答模型的 117,760 token 上限被截断。

## 8. 报告（交给用户）

写 `$RUN/REPORT.md`，包含：

1. **版本**：第 1 节记录的全部版本信息，测试集的组成（各来源数据集和题数）。
2. **Add 阶段**：请求数、总耗时、平均/最长单次耗时（供估算全量成本）、重试和最终失败数、`thread_failures`、`lce_stats.json` 全文。
3. **主结论**：`C vs A+` 的均值差、95% 置信区间、p 值，一句话判定：
   - 置信区间下限 > 0：LCE 有增益；
   - 区间跨 0：未证明有增益；
   - 上限 < 0：LCE 有害。
4. **完整表格**：四组准确率、按题型准确率、所有配对比较。
5. **上下文**：每组的上下文统计；`may_truncate` 非 0 时单独说明。
6. **基线核对**：A 组准确率与用户之前跑出的基线分数对比。两者编译器不同，有差距是正常的，但要报告差多少。
7. **案例**：各挑 5 题，"C 对、A+ 错"和"C 错、A+ 对"，列出问题、标准答案、两组的回答，以及 C 组读到的 LCE 认知条目（`id` 以 `lce:` 开头的 `content`）。
8. **异常**：所有偏离本指南的地方、所有失败和重试、所有人工干预。

## 9. 停止条件汇总

遇到以下任何一项，停止并向用户报告，不要自行处理：
- 预检失败；用户未指定编译用的 LLM；无法确定基线的 embedding / HyDE 设置。
- Add 返回 `400` 或 `422`；重跑 3 遍后仍有失败。
- `lce_lines` 为 0。
- 四组的 `judged` 文件 `id` 集合不一致。
- 需要修改任何代码、提示词、配置字段、模型或依赖。
