# 文档地图 · Documentation map

[返回 README](../README.md)

按用途分组。"状态"一栏说明这份文档现在还算不算数：**现行**是当前的权威，**记录**是某个阶段的报告或决策过程，**历史**已被后来的文档取代、只作留痕。

## 先读这些

| 文档 | 内容 | 状态 |
|---|---|---|
| [LCE_FROZEN.md](LCE_FROZEN.md) | LCE 的冻结定义：原文、SB、Line、段、两条时间轴。改动要先改这里 | 现行 |
| [LCE_DESIGN.md](LCE_DESIGN.md) | 设计稿 v8：核心模型、读出、容易犯的错，每条设计都标了实现状态 | 现行 |
| [DEVLOG.md](DEVLOG.md) | 研究开发日志：每个阶段做了什么、放弃了什么、为什么 | 现行 |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | AML 2026 第二轮技术报告：系统结构、开发测量、正式评测和复盘 | 记录（10-31 – 11-02） |

## 结构与边界

| 文档 | 内容 | 状态 |
|---|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | MR-Mem 作为 sidecar 的边界：管什么、来源怎样写入、Thread 和 LCE 怎样共用语义层（英文） | 现行 |
| [ADR-0001](ADR-0001-SIDECAR-MEMORY-BOUNDARY.md) | sidecar 记忆边界和原生来源接入的决策 | 现行 |
| [ADR-0002](ADR-0002-SEMANTIC-DELTA-V1-SOURCE-ADMISSION.md) | Semantic Delta V1 的原生来源写入 | 现行 |
| [canonical-semantic-projection.md](canonical-semantic-projection.md) | canonical SB 投影的契约（PR #12、#13） | 现行 |
| [PATH_A.md](PATH_A.md) | Path A 核心：canonical SB → Thread → LCE，以及统一读出 | 现行 |
| [EXTRACTION_BOUNDARY.md](EXTRACTION_BOUNDARY.md) | 从 Mind Runtime 抽出时划的边界 | 历史（10-01） |
| [SOURCE-REF-ACCEPTANCE.md](SOURCE-REF-ACCEPTANCE.md) | issue #3 的验收记录 | 历史（10-02） |
| [LCE_PORT.md](LCE_PORT.md) | 把 LCE 移植进 MR-Mem 的方案和旧实现审核 | 历史（10-14） |

## 运行与配置

| 文档 | 内容 | 状态 |
|---|---|---|
| [AML.md](AML.md) | AML 适配层：HTTP 协议、配置项、运行方式 | 现行 |
| [CONFIG_GUIDE.md](CONFIG_GUIDE.md) | 每个配置项开不开、实测效果、成本；试过没采用的也在里面 | 现行 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | 用 Docker Compose 部署到服务器 | 现行 |
| [AML_EVAL_RUNBOOK.md](AML_EVAL_RUNBOOK.md) | 跑配对评测的执行指南（写给执行评测的 agent） | 现行 |
| [AML_SUBMIT_LCE.md](AML_SUBMIT_LCE.md) | 正式评测前的本地准备和验证 | 记录（10-31 – 11-01） |

隐私与发布规则见仓库根目录的 [SECURITY.md](../SECURITY.md)。
