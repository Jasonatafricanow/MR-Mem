# MR-Mem

[English](README.md)

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
| **在哪里测过** | AML 2026 第二轮（2026-11-02 正式评测，平均 **54.37**），之后在本地公开的 BEAM、PersonaMem-v2、CLBench 上 |
| **`main` 上是什么** | 参赛版本，加上赛后测出增益的改动：SB 顺着连线带出、读出里写上消息时间、精准读出（只有搜到的 SB 带原话；开发集 37.1 → 43.1，两遍答题都稳定） |
| **怎么走到这里的** | [研究开发日志](docs/DEVLOG.md)：做过什么、放弃了什么、为什么 |
| **设计** | [冻结定义](docs/LCE_FROZEN.md) · [LCE 设计稿](docs/LCE_DESIGN.md) · [技术报告](docs/TECHNICAL_REPORT.md) · [配置说明](docs/CONFIG_GUIDE.md) |

| 分支 | 内容 |
|---|---|
| `main` | 上面这个版本 |
| `feat/aml-20261013` | AML 参赛版本，冻结 |
| `dev` | 进行中的开发；试验开关默认关闭 |

成绩来自公开数据上的开发测量和一次正式评测，不是独立验证。


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

## 文档与边界

- [HTTP 协议与配置](docs/AML.md)
- [部署说明](docs/AML_SUBMIT_LCE.md)
- [冻结语义定义](docs/LCE_FROZEN.md)
- [技术报告](docs/TECHNICAL_REPORT.md)：开发评测，不能当独立泛化验证
- [隐私与发布规则](SECURITY.md)

仓库不包含会话数据库、评测原始数据、供应商密钥或模型缓存。外部服务配置和本地评测成绩不能保证其他账号或数据集上的效果。

服务器拉取部署见 [Docker Compose 部署说明](docs/DEPLOYMENT.md)。
