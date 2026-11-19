# MR-Mem

[English](README.md)

[![quality](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/quality.yml/badge.svg?branch=main)](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/quality.yml)
[![privacy](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/privacy.yml/badge.svg?branch=main)](https://github.com/Jasonatafricanow/MR-Mem/actions/workflows/privacy.yml)

**一个以原话为权威、把理解当作可追溯投影的记忆系统。** MR-Mem 把每条消息都作为原文证据保存，
把对话编译成语义块（**SB**，一件正在聊的事），用承接把 SB 连成 **Line**（一件事怎样往前推进），
再把 Line 切成**段**，每段只写一次"前因 → 谁 → 何时 → 做了什么 → 后果"的摘要。回答问题时按这套结构取：
原文命中找到它的 SB，SB 找到它所在的段，每个 SB 再按指针取回它编译时依据的原话。编译出来的东西
不替代原文，每个投影都能追回原文。

```text
原文 ──BM25 + 向量（RRF）──▶ 原文证据            （P1 基线）
  │ 按窗口编译，在自然断点收口
  ▼
SB：一件在聊的事 ── 带着指向原话的指针
  │ 承接："这一步是在那一步的基础上推进的"
  ▼
Line ──▶ 段：前因 → 谁 → 何时 → 做了什么 → 后果      （LCE）
```

| | |
|---|---|
| **在哪里测过** | AML 2026 第二轮 Full（11-19 正式榜单：综合 **59.78**，第 26 名；记忆治理 D1、D2 均并列第 4；见[下文](#aml-2026-成绩与成本)），之后在本地公开的 BEAM、PersonaMem-v2、CLBench 上 |
| **`main` 上是什么** | 参赛版本，加上赛后测出增益的改动：SB 顺着连线带出、读出里写上消息时间、精准读出（只有搜到的 SB 带原话；开发集 37.1 → 43.1，两遍答题都稳定） |
| **怎么走到这里的** | [研究开发日志](docs/DEVLOG.md)：做过什么、放弃了什么、为什么 |
| **设计** | [冻结定义](docs/LCE_FROZEN.md) · [LCE 设计稿](docs/LCE_DESIGN.md) · [技术报告](docs/TECHNICAL_REPORT.md) · [配置说明](docs/CONFIG_GUIDE.md) |

| 分支 | 内容 |
|---|---|
| `main` | 上面这个版本 |
| `feat/aml-20261013` | AML 参赛版本，冻结 |
| `dev` | 进行中的开发；试验开关默认关闭 |

成绩来自公开数据上的开发测量和一次正式评测，不是独立验证。

## AML 2026 成绩与成本

AML 只比记忆系统本身：平台通过 Add 写入对话、按题调用 Search，用平台锁定的答题模型和提示词作答、评分。
各家用的答题模型相同，差别只在记忆侧交出去的材料，以及为此付出的成本。

**正式榜单（Full，2026-11-19 发布）**

| | 得分 | 位置 |
|---|---|---|
| 综合 | **59.78** | 第 26；第 5 名 62.28，相差 2.50 |
| A 显式事实召回 | 61.93 | |
| B 关系与多跳推理 | 57.68 | |
| C 时间与事件序列 | 56.59 | |
| D1 新值覆盖 | 77.6 | 并列第 4 |
| D2 矛盾消解 | 76.4 | 并列第 4 |

评测 11-04 结束，平台复评后 11-19 出榜。11-02 跑完时平台给出的初步平均分是 54.37（[技术报告 §4.1](docs/TECHNICAL_REPORT.md)），以榜单为准。

**成本**

| 项 | 参赛配置 |
|---|---|
| 题库规模 | 文本记忆赛道 6,000+ 评测实例、约 3 亿 token（[官方说明](https://github.com/AML-memory/agent-memory-leaderboard)） |
| 编译模型 | DeepSeek v4.1 flash（1017），每个窗口 2 次调用 |
| 向量 | qwen3.7-text-embedding-flash，带本地缓存 |
| 费用 | 编译约 ¥130 + 向量约 ¥5 ≈ **¥135**（按题库规模折合约 ¥0.45 / 百万 token） |
| 运行 | 免费云主机（7.8G 内存），31 小时 39 分 |

开发时的预估是三五千元。成本降下来靠的是结构，不是少做事：

- **推理过的不再推理。** 编译时就留下对话的理解（SB、段摘要），Search 时交出编译产物，不在每次查询时让模型重读原文。
- **SB 只写主干，细节留在原文。** SB 写"谁、做了什么、哪件事"，数字、代码、步骤不抄进去，靠索引回原文。编译的输出因此很短（BEAM 留出集：66 万输入 token 只产生 4 万输出 token）。
- **按自然断点开窗，每窗 2 次调用。** 一次切 SB，一次切段并给收口的段写一次摘要；段摘要写一次就定下，不反复重写。
- **Search 本身不调用编译模型。** 读出是检索加沿索引展开；向量按文本哈希缓存，重启和重复文本都不重复付费。

**代价和赛后的补救**

便宜的编译模型确实留下了痕迹：段摘要偶尔写进旧计划的数值，知识更新题会选旧值。
但复盘发现，显式召回失分的主因在读出：九成标准证据进了读出，只有四成是原话，其余只以 SB 的概括出现
（参赛版只给检索命中的原文，经 Line、段摘要找到的 SB 不带原话，[技术报告 §5.2](docs/TECHNICAL_REPORT.md)）。
赛后让 SB 带着它指向的原话返回，编译不变、成本几乎不涨：

| 本地公开数据（不使用评测数据） | 参赛版 | 赛后 | P1 前 100 条原文 |
|---|---|---|---|
| BEAM 100K 留出集：标准证据以原话进入读出 | 51.8% | **89.4%** | 79.4% |
| PersonaMem-v2 32K：同上 | 65.9% | **98.2%** | |

这是证据召回，不是答题分数。答题分数的提升还在验证：精准读出在开发集上 37.1 → 43.1（两遍稳定），
其他几项改动的答题增益在噪声范围内（[研究开发日志](docs/DEVLOG.md)）。

**怎么看这个成绩。** 59.78 和第 5 名差 2.5 分，记忆治理两项进了前列；花费是一个公开、可复现的数字。
其他参赛系统大多没有公布成本，所以这里不宣称成本-效果的最优，只给出一个可以对照的点：
¥135 跑完 Full，代码和配置在 [`feat/aml-20261013`](https://github.com/Jasonatafricanow/MR-Mem/tree/feat/aml-20261013)。
欢迎复测，也欢迎其他团队公布自己的成本。


## 安装与免费冒烟测试

需要 Python 3.12+ 和 Git。在仓库根目录执行：

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[aml,lce]"
python scripts/smoke_offline.py
```

安装会下载依赖；冒烟测试只用合成消息、临时数据库和本机 HTTP，**不调用模型、不消耗模型额度**。
它验证 Add、Search 和用户隔离，不代表语义编译或评测效果。只使用基础 MemoryCore 时安装 `pip install -e .` 即可。

## 连接模型服务

将 `examples/runtime.aml-lce.json` 复制为仓库根目录的 `runtime.json`，按自己的服务修改模型与端点。
示例通过环境变量读取凭据，没有内置密钥。`.env.example` 仅列变量名称，程序不会自动加载 `.env`。

PowerShell 设置环境变量的形式：

```powershell
$env:QWEN_PLAN_API_KEY = "<你的编译服务密钥>"
$env:DASHSCOPE_API_KEY = "<你的向量服务密钥>"
$env:MRMEM_AML_TOKEN = "<你自己生成的服务访问令牌>"
mr-mem-aml serve --config runtime.json
```

模型名称和服务可用性由你的账号决定。Add、Search 可能调用收费服务；`check` 也会调用编译与向量接口。
`check --check-decision` 才额外探测决策服务。默认仅监听 `127.0.0.1`；配置了鉴权变量却未设置时会拒绝启动。
客户端请求携带 `Authorization: Bearer <访问令牌>`。

当前示例采用 topic SB + stretches 两阶段编译。命中原文没有可用 SB 时直接保留原文；其余读出沿结构索引展开。
决策旁路可通过删除 `decision` 配置关闭，失败或超时不影响基础检索；默认总等待上限 15 秒。
`/health` 提供旁路状态与失败计数，不会主动调用模型。

## 开发检查

```bash
python -m pip install -e ".[dev,aml,lce]"
ruff check src tests
python -m pytest tests
python scripts/smoke_offline.py
```

## 仓库地图

| 路径 | 内容 |
|---|---|
| `src/mr_mem/memory` | MemoryCore：canonical 语义记忆、经 SourceRef 写入、BM25 / 向量 / RRF 检索、Thread |
| `src/mr_mem/integrations`、`src/mr_mem/lce` | Path A：成熟的 Thread 和完整的认知交给 LCE 的 Line；统一的读出路径 |
| `src/mr_mem_aml` | AML 服务：原文存储、编译器（SB、段）、Line 树、读出（`line_read.py`）、HTTP 服务 |
| `tests/` | 上面各部分的契约测试 |
| `docs/` | 设计、决策、报告，从[文档地图](docs/README.md)开始读 |

## 架构与边界

MR-Mem 是 sidecar：宿主已经保存会话时，原文的权威留在宿主，MR-Mem 只保存它编译出的东西，并带着指回原文的引用；
像 AML 这样单独运行时，原文由它自己保存。MR-Mem 管什么、来源怎样写入、Thread 和 LCE 怎样共用同一层语义，
见[架构与边界](docs/ARCHITECTURE.md)（英文）。Path A 见 [PATH_A.md](docs/PATH_A.md)，AML 适配层见 [AML.md](docs/AML.md)，
服务器部署见 [DEPLOYMENT.md](docs/DEPLOYMENT.md)，隐私与发布规则见 [SECURITY.md](SECURITY.md)。

仓库不包含会话数据库、评测原始数据、供应商密钥或模型缓存。外部服务配置和本地评测成绩不能保证其他账号或数据集上的效果。

## 项目状态

个人研究项目，公开开发。`main` 是测过的版本；`dev` 是进行中的工作，新功能都在默认关闭的开关后面。
做过什么、留下什么、放弃了什么，见[研究开发日志](docs/DEVLOG.md)。许可证尚未选定。
