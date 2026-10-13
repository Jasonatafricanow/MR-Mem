# AML 评测适配层（`mr_mem_aml`）

> **分支说明**：本文件只存在于 AML 专属分支 **`feat/aml-20261013`**。该分支基于核心分支 `feat/path-a-core-20261013`（Path A 核心，计划并入 `main`），只增加 AML 相关内容：`src/mr_mem_aml/`、`examples/`、`tests/test_aml_runtime.py`、`mr-mem-aml` 命令入口和本文档。Path A 的编译与读取逻辑全部来自核心，本分支不修改。

`mr_mem_aml` 提供 AML 式的 Add/Search HTTP 服务，用于在开放数据集和 AML 平台上做基线与 Path A 的配对评测：原文存储、语义编译器（agy / grok 命令行或 OpenAI 兼容端点）、可选向量召回与 HyDE、AML 官方接口契约。

## 1. 组成

| 模块 | 职责 |
|---|---|
| `source_store` | 原文按 request_id 只写一次；按 SourceRef 回查原文 |
| `compiler` | 一次模型调用产出 SemanticBlock、Thread 信号和首次即成熟信号；`PROMPT_VERSION` 标记提示词版本 |
| `embedding` | Gemini 或 OpenAI 兼容 embedding（可选） |
| `runtime` | Add / Search 管路：准入 Canonical → 向量投影 → 交给核心 `apply_block_signals`；检索交给核心 `PathAReader` |
| `server` | HTTP 服务，对齐 AML 官方契约 |
| `__main__` | `mr-mem-aml check / serve` |

## 2. 运行管路

```text
POST /add
  AMLSourceStore        原文按 request_id 只写一次；同一 request_id 内容不同 → 422
  SemanticCompiler      一次模型调用，同时产出 Block、Thread 信号、首次即成熟的 commit
                        结果先记入 journal；重试时复用，不再调用模型
  Canonical 准入         按语义身份幂等
  向量投影（可选）        完成前不返回 200
  Path A（核心）         path_a_ingest.apply_block_signals：Thread 维护、成熟交接、commit / revise
                        成熟判断沿用两次独立交互规则；全部按 processing_input_id 幂等
  → 200 时，内容已持久化且可检索

POST /search            {"query","user_id","top_k"} → {"data":[{"id","content","score","created_at"}]}
  请求头 X-MRMem-Search-Mode: baseline | path_a   单次请求覆盖配置里的读取模式
POST /admin/flush       重扫所有 Scope，重试仍未交接的成熟 Thread
GET  /health
```

失败语义：
- 模型调用或 LCE 失败，**都不会回滚 Canonical**。请求保留为可恢复状态，返回 503 和 `Retry-After`，用同一个 request_id 重试即可续跑。
- Thread 维护失败只记日志并计入 `/health` 的 `thread_failures`，不阻塞写入。

## 3. 评测建议：三组对照

同一个 `data_dir` 只编译一次，三组只在读取方式上不同：

| 组 | 配置 | 回答的问题 |
|---|---|---|
| A | `X-MRMem-Search-Mode: baseline` | 当前水平 |
| A+ | `baseline`，并设 `read_policy.source_text_for_hits=true` | 只加原文证据，能提升多少 |
| C | `path_a`（可再开 `source_text_for_hits`，与 A+ 对齐） | LCE 本身的增益：看 C 相对 A+ 的差值 |

如果 C 只比 A 好、不比 A+ 好，说明提升来自原文证据，而不是 LCE。

## 4. 配置与启动

```bash
python -m pip install -e ".[lce,vector]"      # vector 仅在启用向量召回时需要
mr-mem-aml check --config runtime.json        # 预检：LCE、编译器、embedding
mr-mem-aml serve --config runtime.json
```

`examples/` 目录下有 agy、grok 和 OpenAI 兼容端点三种编译器配置。主要字段：

| 字段 | 说明 |
|---|---|
| `data_dir` | 数据目录；相对路径以配置文件所在目录为基准 |
| `compiler` | `{"type":"command","argv":[...]}`：默认用 stdin 传提示词；argv 中可用 `{prompt}` 或 `{prompt_file}` 占位。或 `{"type":"openai","base_url","model","api_key_env"}` |
| `embedding` | `null` 表示只用 BM25；`{"type":"gemini","model","dimension","api_key_env"}`；或 `{"type":"openai",...}` |
| `hyde` | `off` / `fallback` / `augmented`，使用编译器同一个后端 |
| `search_mode` | 默认读取模式：`path_a` 或 `baseline` |
| `lce_enabled` | `false` 时不写 LCE（纯基线运行） |
| `assistant_policy` | `attributed`：助手消息里有信息价值的内容可以编译成 Block，并标明来自助手；`context_only`：只编译用户消息 |
| `context_messages` | 每次编译带入的同一会话之前的消息条数（只作上下文，不可引用） |
| `read_policy` | 见核心文档 [PATH_A.md](PATH_A.md) 第 4 节 |
| `auth_token_env` | 设置后，所有 POST 请求需要鉴权，接受 AML 的三种方式：`Authorization: Bearer <token>`、`Authorization: Token <token>`、`X-Api-Key: <token>`。`GET /health` 不鉴权 |

注意：
- **基线配置要和你之前跑基线时保持一致**，特别是 embedding 模型和 HyDE 开关，否则 A 组不等于你已有的基线。
- 服务是单线程的。Add 会阻塞在模型调用上，评测时建议先完成全部 Add，再做 Search。

## 5. 与 AML 官方接口契约的对齐

依据：AML 官方接口指南（agentmemoryleaderboard.ai/api-guide）和官方仓库 `AML-memory/agent-memory-leaderboard`（`1b8142b`）里的回答与评审管路。官方仓库没有 Add/Search 适配示例，只公开了各数据集的回答、评审脚本。

| 契约 | 本实现 |
|---|---|
| Add 请求 `request_id / user_id / session_id / messages[role, content, timestamp?]` | 一致；`timestamp` 缺失时用接收时间 |
| Add 成功返回 200，且写入已持久化、可立即检索 | 一致；返回 `{"success": true, "request_id", "user_id", "session_id"}` 原样回显，另带 `blocks` 等附加字段 |
| Add 重试（408/409/425/429/5xx，最多 32 次，同一 request_id 和内容） | 同内容重放返回 200（`replayed: true`），不再调模型；编译或 LCE 失败返回 503 + `Retry-After` |
| 同一 request_id 内容不同 | 返回 **422**（永久错误）。不用 409，因为 AML 会把 409 重试 32 次 |
| Search 请求 `query / user_id / top_k`，可选 `options`（选择题选项） | 一致；`options` 接受但不参与检索，两组对照一致 |
| Search 返回 `{"data": [{id, content, score, created_at}]}`，按相关度排序，不超过 `top_k`，`content` 非空 | 一致 |
| 鉴权 `Token` / `Bearer` / `X-Api-Key`，health 为不鉴权 GET | 一致 |
| 单次请求最长 30 分钟 | agy 每次 Add 约 15 秒，grok 37–132 秒 |
| 回答模型输入上限约 117,760 token，超出时平台按排名保留前缀 | Path A 模式下投影排在最前，不会被截掉；附带原文时总长度会变大，排在最后的 Canonical 结果可能被截掉 |

注意：
- **并发。** 服务是单线程顺序处理的，并发请求在监听队列里排队（队列长度 128）。评测表单里的 Add 并发建议设为 1：同一会话的分块必须按顺序写入，否则编译时拿到的上下文和 Thread 演进顺序会错。
- **正式提交需要公网 HTTPS。** AML 不接受回环、内网地址，也不接受 URL 里带凭据。正式评测前需要在服务前面加 HTTPS 反向代理或隧道，并设置 `auth_token_env`。
- **官方回答提示词**把记忆称为"episodic raw observations"，冲突时"prefer the most recent"。已接受认知带有 `revision N | as of <时间>` 标注，`created_at` 是其支撑证据中最新一条的时间，因此时间上和原文一样可比。
- **双说话人数据集**（LoCoMo、CL-Bench）的官方管路按两个说话人分别检索、分别拼接。按 `user_id` 隔离已经覆盖这种用法；如果平台对两个说话人各 Add 一遍，编译成本会翻倍。
- 评测结束后 30 天内删除评测数据：删除 `data_dir` 即可。

## 6. 已知限制

- 编译质量完全取决于所用模型和提示词（`PROMPT_VERSION`）。换模型或改提示词后，需要重新编译整个数据集，不能和旧结果混用。
- 编译器会拿到该用户已有的问题列表（KNOWN_LINES），以便复用同一个问题，让后续信号落到同一条认知线上。
- 本适配层**自存原文**（`aml_source.sqlite`），因为 AML 只通过 Add 推送对话，没有宿主存储可回查。这是 ADR-0001 的无宿主例外，**不适用于 agent 接入**；agent 接入应由宿主保有原文，通过 `render_source` 回查。
- Path A 本身的限制见 [PATH_A.md](PATH_A.md) 第 5 节。
