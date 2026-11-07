# AML 评测适配层（`mr_mem_aml`）

> **分支说明**：本文件只存在于 AML 专属分支 **`feat/aml-20261013`**。该分支基于核心分支 `feat/path-a-core-20261013`（Path A 核心，计划并入 `main`），只增加 AML 相关内容：`src/mr_mem_aml/`、`examples/`、`tests/test_aml_runtime.py`、`mr-mem-aml` 命令入口和本文档。Path A 的编译与读取逻辑全部来自核心，本分支不修改。

`mr_mem_aml` 提供 AML 式的 Add/Search HTTP 服务，用于在开放数据集和 AML 平台上做基线与 Path A 的配对评测：原文存储、语义编译器（agy / grok 命令行或 OpenAI 兼容端点）、可选向量召回与 HyDE、AML 官方接口契约。

## 1. 组成

| 模块 | 职责 |
|---|---|
| `source_store` | 原文按 request_id 只写一次；按 SourceRef 回查原文 |
| `compiler` | 补全后端（命令行 / OpenAI 兼容端点）和唯一的编译路径：每个窗口先由第一步写出 SB，再由段层切段、给收口的段写摘要（冻结定义 §2，每窗 2 次调用） |
| `moves` | 编译提示词与解析：聊天场景按在聊的那件事切 SB（`STEP1_TOPIC`），工作场景按人的回应切（`STEP1`），段层（`STRETCH`） |
| `embedding` | 本地 FastEmbed（如 bge）、任意 OpenAI 兼容 `/embeddings` 端点、Gemini，或火山方舟多模态向量（`ark`，如 doubao-embedding-vision，带本地向量缓存）（可选） |
| `raw_arm` | 原文检索（`raw_base=true` 时启用）：每条消息一个单元，BM25 + 向量，RRF k=60、每路候选 200，即 P1 基线协议；单元保留所属消息的 record id，用于向上找到编译产物 |
| `runtime` | Add / Search 管路：按窗口编译 → 准入 Canonical → 向量投影 → 长到 Line 树上；Search 走 `line_read`（原文命中 → SB → Line 和段），`baseline` 模式和不设 `line_read` 时交给核心 `PathAReader` |
| `line_read` | Path A 读出：原文命中挂在所属 SB 下、带消息时间，SB 带它投影的原话，沿 Line 和段带出经过 |
| `server` | HTTP 服务，对齐 AML 官方契约 |
| `__main__` | `mr-mem-aml check / serve` |

## 2. 运行管路

```text
POST /add
  AMLSourceStore        原文按 request_id 只写一次；同一 request_id 内容不同 → 422
  原文检索单元（raw_base）  编译前就写入原文索引，原文立即可检索
  → 200 时，原文已持久化且可检索；编译按窗口进行（compile_window），background_compile 时在后台
  编译窗口               在 soft 之后的第一个自然断点收口（会话切换、长停顿、场景的回合边界），最迟 hard
  第一步（SB）            一次模型调用写出本窗的 SB；模型可停在 SB 收口处，没编完的尾巴进下一窗
                        结果先记入 journal；重试、重启时复用，不再调用模型
  Canonical 准入         按语义身份幂等；向量投影（可选）
  段层                  一次模型调用把本窗 SB 切成段、接上未收口的段或更早的段，给收口的段写摘要
  Line 树               SB 按段内顺序和段的承接长到 Line 上（晨昏线：已发生 / 进行中）
  编译失败               keep_raw_on_compile_failure / skip_rejected_compiles 时只保留原文、计数，Add 仍返回 200

POST /search            {"query","user_id","top_k"} → {"data":[{"id","content","score","created_at"}]}
  请求头 X-MRMem-Search-Mode: baseline | path_a   单次请求覆盖配置里的读取模式
POST /admin/flush       把所有用户（或指定用户）剩下的消息编完，尾巴也编
GET  /health
```

`raw_base=true` 时，Search 的基线是原文检索列表（与 P1 完全一致），Path A 模式在其上加入向上找到的投影（见 PATH_A.md §4）。多选题的选项按 `append_options` 追加到查询。

失败语义：
- 原文存好即返回 200；窗口编译失败**不会回滚 Canonical**，也不让 Add 失败：计入 `/health` 的 `window_failures`，由下一次 Add 或 flush 重试（开 `keep_raw_on_compile_failure` 时直接只留原文）。
- 后台编译积压过多时（`background_compile.max_backlog`），Add 等待后返回 503，由 AML 重试。

## 3. 评测建议：三组对照

同一个 `data_dir` 只编译一次，三组只在读取方式上不同：

| 组 | 配置 | 回答的问题 |
|---|---|---|
| A | `X-MRMem-Search-Mode: baseline` | 当前水平 |
| A+ | `baseline`，并设 `read_policy.source_text_for_hits=true` | 只加原文证据，能提升多少 |
| C | `path_a`（可再开 `source_text_for_hits`，与 A+ 对齐） | LCE 本身的增益：看 C 相对 A+ 的差值 |

如果 C 只比 A 好、不比 A+ 好，说明提升来自原文证据，而不是 LCE。

完整执行步骤（配置、试跑、Add、四组检索、回答评审、比较、报告）见 [AML_EVAL_RUNBOOK.md](AML_EVAL_RUNBOOK.md)，配套脚本在 `scripts/aml_eval/`。

## 4. 配置与启动

```bash
python -m pip install -e ".[aml,lce,vector]"      # vector 仅在启用向量召回时需要
mr-mem-aml check --config runtime.json        # 预检：LCE、编译器、embedding
mr-mem-aml serve --config runtime.json
```

`examples/` 目录下的配置：
- `runtime.openai-compatible.json`：在线 LLM（gpt-4o-mini）+ 本地 bge-small 向量。**数据集评测推荐用这一类。**
- `runtime.local-llm.json`：本机 Ollama / vLLM 一类的 OpenAI 兼容服务，编译和向量都走本地。
- `runtime.agy.json`、`runtime.grok.json`：命令行 agent。每次调用都有固定启动开销（agy 实测约 16 秒，回一个 `{"ok": true}` 也要这么久），只适合小规模冒烟测试。命令需在 PATH 上，否则 `argv` 第一项写完整路径。

主要字段：

| 字段 | 说明 |
|---|---|
| `data_dir` | 数据目录；相对路径以配置文件所在目录为基准 |
| `compiler` | **OpenAI 兼容端点**：`{"type":"openai","base_url","model","api_key_env"?,"json_mode"?,"max_tokens"?,"temperature"?,"headers"?,"extra_body"?,"timeout"?}`。适用于 OpenAI、DeepSeek（`https://api.deepseek.com`）、硅基流动（`https://api.siliconflow.cn/v1`）、OpenRouter（`https://openrouter.ai/api/v1`）、xAI（`https://api.x.ai/v1`）、Gemini（`https://generativelanguage.googleapis.com/v1beta/openai`）、Ollama（`http://127.0.0.1:11434/v1`）、vLLM、LM Studio 等。`api_key_env` 省略时不发鉴权头（本地服务）；`json_mode: true` 请求 JSON 输出（端点支持时打开）；`extra_body` 原样并入请求体。429 按 `Retry-After` 退避，400/401/403/404/422 直接失败不重试。**命令行**：`{"type":"command","argv":[...]}`，默认用 stdin 传提示词，argv 中可用 `{prompt}` 或 `{prompt_file}` 占位 |
| `embedding` | `null`：只用 BM25。**`{"type":"fastembed","model","dimension","model_dir"}`**：MR-Mem 自带的本地 ONNX 向量（与 MR-Mem 基线用的是同一个实现），如 `BAAI/bge-small-en-v1.5`（384 维）；模型文件用 `mr-mem-aml fetch-model --model <名称> --dir <目录>` 下载一次。`{"type":"openai","base_url","model","dimension","api_key_env"?,"send_dimensions"?}`：任意 OpenAI 兼容 `/embeddings`（OpenAI、硅基流动的 bge、Hugging Face TEI、Ollama、vLLM），返回维度与 `dimension` 不符时报错。`{"type":"gemini","model","dimension","api_key_env"}` |
| `compile_concurrency` | 同时进行的模型调用数（不同 `user_id` 之间并行），默认 4，范围 1–64。按端点的限流设置 |
| `hyde` | `off` / `fallback` / `augmented`，使用编译器同一个后端 |
| `search_mode` | 默认读取模式：`path_a` 或 `baseline` |
| `context_messages` | 每次编译带入的同一会话之前的消息条数（只作上下文，不可引用） |
| `read_policy` | 见核心文档 [PATH_A.md](PATH_A.md) 第 4 节 |
| `auth_token_env` | 设置后，所有 POST 请求需要鉴权，接受 AML 的三种方式：`Authorization: Bearer <token>`、`Authorization: Token <token>`、`X-Api-Key: <token>`。`GET /health` 不鉴权 |

编译窗口、场景、编译输入、读出（`line_read`）、后台编译、决策旁路等字段，是否建议开、实测效果和成本，见 [CONFIG_GUIDE.md](CONFIG_GUIDE.md)。

注意：
- **基线配置要和你之前跑基线时保持一致**，特别是 embedding 模型和 HyDE 开关，否则 A 组不等于你已有的基线。
- **并发模型。** 所有存储（Canonical、原文、LCE、向量索引）只由一个线程读写，保证顺序和一致性；耗时的模型调用在请求线程里并行，上限为 `compile_concurrency`。同一 `user_id` 的请求整体串行，保证每块编译时拿到前一块产生的上下文和认知线。客户端可以对不同用户并发发送，同一用户的分块仍需按顺序、等上一块返回再发下一块。
- **吞吐。** 编译阶段（模型调用）随并发线性提升；提交阶段（准入、向量、Thread/LCE）在单线程上串行。实测每个 Block 的提交约 60–80 毫秒（Windows，核心存储每次提交都刷盘，`synchronous=FULL`），即一块 20 条消息、出 5–10 个 Block 时约 0.5–0.8 秒。
- 评测时建议先完成全部 Add，再做 Search。

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
| 单次请求最长 30 分钟 | 在线 LLM 每次 Add 通常几秒到十几秒；命令行 agent 另加十几秒启动开销 |
| 回答模型输入上限约 117,760 token，超出时平台按排名保留前缀 | Path A 模式下投影排在最前，不会被截掉；附带原文时总长度会变大，排在最后的 Canonical 结果可能被截掉 |

注意：
- **并发。** 评测表单里的 Add 并发可以设为 `compile_concurrency`。AML 按会话顺序推送分块；不同对话之间并行编译，同一 `user_id` 内由服务端串行。
- **正式提交需要公网 HTTPS。** AML 不接受回环、内网地址，也不接受 URL 里带凭据。正式评测前需要在服务前面加 HTTPS 反向代理或隧道，并设置 `auth_token_env`。
- **官方回答提示词**把记忆称为"episodic raw observations"，冲突时"prefer the most recent"。已接受认知带有 `revision N | as of <时间>` 标注，`created_at` 是其支撑证据中最新一条的时间，因此时间上和原文一样可比。
- **双说话人数据集**（LoCoMo、CL-Bench）的官方管路按两个说话人分别检索、分别拼接。按 `user_id` 隔离已经覆盖这种用法；如果平台对两个说话人各 Add 一遍，编译成本会翻倍。
- 评测结束后 30 天内删除评测数据：删除 `data_dir` 即可。

## 6. 已知限制

- 编译质量完全取决于所用模型和提示词（`PROMPT_VERSION`）。换模型或改提示词后，需要重新编译整个数据集，不能和旧结果混用。
- 段层每窗只看未收口的那一段、最近的 8 个已收口段和最多 10 个进行中的步骤；更早的段要靠检索带回，不进编译输入。
- 本适配层**自存原文**（`aml_source.sqlite`），因为 AML 只通过 Add 推送对话，没有宿主存储可回查。这是 ADR-0001 的无宿主例外，**不适用于 agent 接入**；agent 接入应由宿主保有原文，通过 `render_source` 回查。
- Path A 本身的限制见 [PATH_A.md](PATH_A.md) 第 5 节。
