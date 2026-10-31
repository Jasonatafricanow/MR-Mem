# AML 提交：MR-Mem + LCE（本地准备，2026-10-31）

部署和公网接入由用户那边负责；这里是本地已经准备好、验证过的东西，推到远端照着起服务就行。

## 起服务

```bash
pip install -e ".[aml,lce]"
export QWEN_PLAN_API_KEY=...    # 阿里百炼 Token Plan（deepseek-v4.1-flash-1017 编译）
export DASHSCOPE_API_KEY=...    # 阿里百炼（qwen3.7-text-embedding-flash 向量）
export MRMEM_AML_TOKEN=...      # Add/Search 的鉴权 token（Bearer / Token / X-Api-Key 都认）
mr-mem-aml check --config examples/runtime.aml-lce.json   # 会调用编译和向量服务，可能计费
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
| `compile_window` | 软 4 万 / 硬 8 万字符，自然断点收口；`max_messages: 200`；`max_chars_per_user: 10000000` | 不开 `background_compile` 时，窗口收口的那次 Add 会等编译完成再返回（1–3 分钟）。一个窗口最多 200 条消息（第一步的输出随条数增长，500 条短消息曾撞 1.6 万 token 上限）。一个用户的历史编译到 1,000 万字符（约 200 万 token）为止，之后只留原文、照样可检索（等同 P1）：评测一次性灌入全部历史，编译费就是读的每个 token，而 BEAM 10M 这类超长历史一项就占文本赛道约三分之一的 token、约 3% 的题；BEAM 1M（约 470 万字符）不受影响 |
| `compile_input` | `{"assistant_head": 600}` | 第一步只读助手每条话的开头和结尾共 600 字符，代码块读前 200 字符；人的话一字不动，检索和读出仍用完整原文。助手的 SB 本来只写"给了什么"，内容留在原文。BEAM 100k 同一时段两遍：不压缩 50.1，压缩 50.9；输入 token −65%、输出 −29%。按此外推，加上封顶，文本赛道 Full 编译约 ¥210–240（输入每百万 ¥1、输出每百万 ¥4 的非高峰价，误差约 ±30%） |
| `background_compile` | `{"search_wait": 15, "idle_tail": 30, "max_backlog": 160000, "add_wait": 12}`（`deploy/runtime.server.json`） | 公网入口（InstaBox）约 20 秒就断开请求。Add 存完原文、建好检索索引就返回（AML 要求的"持久化且立刻可检索"），到期的窗口在后台按用户逐个编译；一个用户 30 秒没有新的 Add，就把尾巴窗口也编掉（曾设 5 秒：Add 要 5–7 秒时，定时器在两次 Add 之间就触发，Smoke 里一个 361 条消息的用户被切成 8 个窗口，原来是 2 个）。Search 最多等这个用户的后台编译 15 秒，到时还没编完就用已有结果回答（没编到的部分照样有原文命中）。向量按每批 10 条、4 路并行请求（`embedding.batch` / `embedding.workers`），原来每条消息单独一次请求。Add 的向量在它自己的请求线程上取好再入库：放在串行处理所有用户存取的 owner 线程上时，一个 Add 的向量请求会把别的用户的 Add 和 Search 都堵在后面（Smoke：Add 最长 27 秒，超过网关的 20 秒被断开） | 平台是一个样本一个样本地写完就问（Smoke：Add 与 Search 交错）。所以一个用户待编译的积压超过 16 万字符或 2×`max_messages` 条时，Add（已入库）等到它到达后的第 12 秒，还降不下来就返回 503（从到达算起，不从入库完算起，整个请求不超过网关的 20 秒），AML 会重试（Add 最多 32 次，重试是幂等重放），写入就跟着编译的节奏走，长历史写完时前面基本已经编完；否则 1M token 的历史几分钟就写完，它的题来时大半还没编。每个请求在日志里记一行：路由、用户、耗时、状态，Search 还记返回的 thread / line / raw 条数 |
| `decision` | `decision-model-preview`（System One），走 Token Plan（`QWEN_PLAN_API_KEY`，和编译同一个 key），Search 时旁路判断，读出最前面加一条说明 | 用户 10-31 定。两条规则（mr_mem_aml/decision.py）：问"我有没有做过 X"，而人自己的命中消息里既说做过、又说从没做过 → 引用两句原话；问人自己的某个具体信息，而人从没说过 → 说明这一点（`absence`，**默认关**：它几乎就是拒答题的答案，AML 规定 Search 不得生成最终答案；PersonaMem 上 62 题挂了 42 题，关掉后 58.1 → 58.9）。只判断不改证据，不写进编译产物；调用失败就不加。holdout BEAM 60 题两遍：52.6 → 58.5（两条都开；矛盾 8 → 50，拒答 50 → 67）。每次 Search 1 次分流调用，是否题再加至多 30 次；全进程限 16 路并发。要关掉就删掉这一项 |
| 编译流程 | `step1: topic`，`layer2: stretches`：每窗 2 次调用（SB 一次、段一次） | 定义见 docs/LCE_FROZEN.md：SB 是在聊的那件事，段摘要按 前因-谁-何时-做了什么-后果 写一次 |
| `line_read` | relay（原文命中 → SB → Line 上的局部，只带检索命中的原文），助手部分 `told`，全部段摘要（`thread_k: 60`、`thread_sbs: 0`），`marks: false`，`raw_fill: 0`，`hit_anchors: true` | 用户 10-31 定：不补位，证据已够（原文 → SB → Line、段摘要 → SB → 原文两条路）；晨昏线标记只在记忆侧用，不给答题模型看；每条原文命中都带出它的 SB，不受 `sbs` 上限截断（原来 1800 条前 30 命中里丢 208 条）。holdout 60 题两遍：P1 47.2，不补位 50.3，修正时间 + 决策旁路 58.5，再加 `hit_anchors` 60.1（时间推理两遍 33 → 50）；每题约 8.5 万字符。`thread_span`：每条段摘要带它首尾一步的位置和时间（定义 §4），事件顺序两遍 54 → 67。`forget_scope`：人的遗忘请求从长期指令里单独列出，标明每条只管它说的那一件（PersonaMem-v2 32k 62 题两遍：P1 56.5，LCE 52.4 → 58.1；中性偏好 50 → 64） |

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


## 发布版本的运行保护

- 不补位时，没有可用 SB 的原文命中仍直接返回，不会因编译失败变成空结果。
- 决策旁路判断的就是读出交出去的那批原文命中（同一个检索文本，选项也在内；同样的条数）。
  判断规则和说明措辞保持实测过的版本：缺失判断看每条命中的前 500 字符（合计至多 15000），
  矛盾判断看人的每条命中的前 6000 字符。曾改成"只看完整消息、超过 15000 字符就不判"
  "矛盾说法要先比较时间与范围"，在留出集上缺失说明一题都不再触发，60 题 60.1 → 55.5，已撤回。
- 命中的原文整条给出，不按所选 SB 的指针裁剪（一条消息切进几个 SB 时，原来只给被选中 SB 的那部分）。
- `decision.total_timeout` 默认 15 秒，包含排队、调用与退避的等待；到期返回基础读出。
  已发出的底层请求可能稍后才结束，但不会再发起过期重试。
- `/health.decision` 给出配置状态、请求/说明/失败/超时计数及最近结果；不主动调用模型。
  `check --check-decision` 显式开启额外服务探测，可能计费。
- `auth_token_env` 已配置而环境变量缺失或为空时，启动失败，不会悄悄变成匿名服务。
- 免费检查入口为 `python scripts/smoke_offline.py`。上述保护只经过本地模拟与回归测试，
  尚未重新付费测分；本文历史分数不能直接当作修改后成绩。
