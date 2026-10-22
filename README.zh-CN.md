# MR-Mem

[English](README.md) | [简体中文](README.zh-CN.md)

从 Mind Runtime 拆出的独立语义记忆 sidecar。宿主原生会话库保留原文权威；MR-Mem 保存
接受的语义记忆，以及可追溯、可重建的 Thread / LCE 派生状态与检索索引。

## 已实现的链路

- SourceRef 来源、版本与 scope 校验；语义准入、持久化、失效和幂等恢复。
- BM25、可选向量检索、RRF 融合和可选 HyDE；未配置模型时不假装语义编译成功。
- Thread / Path A 树图写入，以及带来源、时间和预算约束的 Line 相邻步骤读出。
- 原生宿主批次接入：只保存来源元数据，支持编辑/删除传播、取消旧批次和重启恢复。
- 独立 AML 适配器与 HTTP 入口。AML 缺少宿主原文库时自行保存来源，不能把该方式
  当成原生宿主必须复制整份历史的要求。

## 安装与验证

```sh
python -m pip install -e ".[dev]"
python -m pytest
```

旧 LCE 集成需要 `.[lce]`；向量后端需要 `.[vector]`。树图核心与原生批次入口位于本仓库。
[总装说明](docs/ASSEMBLY.zh-CN.md) 包含算法尚未填入时的使用方式；原生入口配置示例为
`examples/native-tree.disabled.json`，具体契约见 [原生来源接入](docs/NATIVE_INGEST_CONTRACT.md)。

## 权威与完成度

`SemanticAdmissionService.admit` 接受宿主已经认可的语义候选，并重新验证当前来源。
派生图、嵌入、摘要和检索结果不获得独立事实权威。原文发生变化后，旧支持必须失效。
具体边界见 [ADR-0001](docs/ADR-0001-SIDECAR-MEMORY-BOUNDARY.md)。

工程接线通过固定夹具验证，不代表 SemanticBlock 边界、归属或时态算法完成验收。
Path B 跨经历假设仍是 [设计稿](docs/LCE_DESIGN.md) 中的第二阶段。真实模型质量、
完整历史策略重建与生产宿主长期运行不能由测试数量代替。

MR 的情感、appraisal、intent、宿主交付与在线 Body 推理不属于 MR-Mem core。
