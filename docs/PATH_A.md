# MR-Mem Path A（核心能力）

本文说明 MR-Mem 的 Path A 核心：Canonical SemanticBlock → Thread → LCE（Line / Worktree / Baseline）的编译路径，以及统一读出口。核心与宿主无关，可并入 `main`，供 agent 集成和评测适配层共用。

依据：Mind-Runtime ADR-0033（`e12ec94`）、MEMORY_ARCHITECTURE_V1、lce-binding.md、MR-Mem ADR-0001/0002、LCE `SEMANTIC_BLOCK_DESIGN_FREEZE_2026-09-25`。LCE 锁定在 `5228475`。

> AML 评测适配层（`mr_mem_aml`）只在专属分支 **`feat/aml-20261013`** 上，说明见 [AML.md](AML.md)。核心分支不含这部分。

## 1. 组成

全部位于 `mr_mem.integrations`，属于可选依赖（`pip install -e ".[lce]"`），`mr_mem.memory` 核心不引用它们。

| 模块 | 职责 | 来源 |
|---|---|---|
| `lce_path_a.LcePathA.compile` | 成熟 Thread → `PrecomputedDraftInput` → Worktree → Baseline；Baseline 被接受后删除 Thread；lineage 为 `mr-thread:<thread_id>` | 移植自 Mind-Runtime `LceThreadProjectionCompiler` / `open_lce_thread_handoff` |
| `lce_path_a.LcePathA.commit` | 首次出现就已完整的认知，直接进入同一条 lineage | **新增**，见第 3 节 |
| `lce_path_a.LcePathA.revise` | 已接受的认知线收到 resolve 或成熟信号时，从 HEAD 继续修订 | **新增**，实现 V1 §7「B1 + M4 → B2」，见第 3 节 |
| `lce_path_a.LcePathA.accepted` | 列出每条认知线的 HEAD，含支撑、修订号、来源（thread / commit / revision）、证据是否完整 | 新增 |
| `path_a_ingest.apply_block_signals` | 把编译器给每个 Block 的 Thread 信号、首次即成熟信号路由到 Thread 维护和 LCE | 新增；所有宿主共用同一套路由语义 |
| `path_a_read.PathAReader` | 统一读出口：已接受认知（含支撑与原文）→ 活跃 Thread（含支撑）→ Canonical 检索；`baseline` 模式只返回最后一部分 | 移植自 Mind-Runtime `retrieval_composition.py` |

## 2. 宿主如何接入

宿主负责三件事，核心负责其余：

1. **原文。** 宿主保有原文（ADR-0001：session history 是唯一原文权威）。准入 Canonical 时给出 `SourceRef`；读取时通过 `render_source(ref)` 回调按引用取回原文，取不到返回 `None`。
2. **语义编译。** 宿主的模型把一段对话编译成 SemanticBlock，并为每个 Block 给出至多一个 Thread 信号（`ThreadSignal(action, question, summary, mature)`）和至多一个首次即成熟信号（`CommitSignal(question, summary)`）。Block 准入 Canonical 后，用 `BlockSignals(memory_id, content, thread, commit)` 交给 `apply_block_signals`。
3. **检索栈。** 宿主选择 `MemoryRetrievalService` 的 provider（BM25、向量 RRF、HyDE 等），交给 `PathAReader`。基线模式就是这一检索栈本身。

```python
from mr_mem.integrations.lce_path_a import LcePathA
from mr_mem.integrations.path_a_ingest import BlockSignals, ThreadSignal, apply_block_signals
from mr_mem.integrations.path_a_read import PathAReader, ReadPolicy

lce = LcePathA(canonical_path=db, lce_root=root / "lce", scope=scope)
result = apply_block_signals(canonical=core.canonical, products=core.products,
                             lce=lce, scope=scope, blocks=signals, at=now)
# result.promotion_failures > 0：用同一输入重试即可（全部幂等）

reader = PathAReader(canonical_path=db, retrieval=retrieval, lce=lce,
                     policy=ReadPolicy(), render_source=host_render)
items = reader.search(scope, query, limit=20, mode="path_a")
```

失败语义：Thread 和 LCE 都是派生投影，失败**不会回滚 Canonical**。
- Thread 维护失败只记日志，结果里 `thread_failed=True`。
- 成熟 Thread 交接失败时 Thread 保留，下一次任意 `apply` 的无状态重扫会重试。
- `commit` / `revise` 失败计入 `promotion_failures`，按 `processing_input_id` 幂等，重放不会产生多余修订。

## 3. 与原实现的差异（需要正式 ADR 确认）

1. **首次即成熟的直接提交（`commit`）。** 语义编译器判定某个 Block 已经包含完整、自洽的推理判断时，直接交给 LCE，不经过 Thread。
   - 与 Thread 成熟的关系见第 4 点：首次即成熟针对的是一个 Block 本身就是完整判断的情况；Thread 成熟针对的是由多个 Block 演进到结束的事。
   - lineage 和同一问题的 Thread 共用，所以之后的 Thread 成熟会在这条 lineage 上产生新修订，而不是另起一条。
   - 依据：2026-10-13 用户意图（"单次成熟"），以及 ADR 输入材料 §3、§9。
2. **已接受认知线的继续修订（`revise`）。**
   - Mind-Runtime 的问题：Thread 编译后被删除，之后对同一条线的 resolve 会因为找不到 Thread 而被丢弃；重新开启的 Thread 又必须重新凑齐两次独立交互。结果 Baseline 一直停在第一次成熟时的内容。
   - 本实现：已有 HEAD 的认知线收到 resolve 或成熟信号时，以"HEAD 的支撑 + 新 Block"产生新修订，内容使用编译器在同一次调用里给出的理解，不另外调用模型。非 settle 的信号（如 `track`）仍然开 Thread。
   - 已有 HEAD 的线再次经 Thread 成熟时，交接的支撑同样合并 HEAD 的支撑，新修订仍然引用旧证据。
   - 依据：MEMORY_ARCHITECTURE_V1 §7。
3. **证据回溯不到时不丢弃认知。** `accepted()` 对支撑不再 ACTIVE 的 HEAD 照常返回，标记 `evidence_complete=False`；读出口在认知和其支撑 Block 上都注明 "original evidence unavailable"。Mind-Runtime 的做法是整条跳过。依据：2026-10-13 用户规则（原文是权威证据，投影不消失，只有系统级人工删除才删）。
4. **不产出 SemanticPoint。** MR-Mem 单独使用时，Point 没有消费方；编译器直接按语义完整性产出 Block。
5. **Thread 成熟 = 这件事在当前时间下结束了（替换 ADR-0033 的"两次独立交互"规则）。**
   - Thread 是进行时，LCE Path A 是过去时。编译器判断一件事已经结束（`resolve`，或 `track` 带 `mature`），Thread 就成熟、交给 LCE；不再要求支撑来自两次不同的交互。
   - 例："这电脑好牛，我想买" → "没钱，不买"：两个 Block，可以在同一轮里说完，事情结束即成熟。以后又想买，是另一件事：把这条线召回来对照，在同一条线上产生新修订。
   - 只有新支撑、没有结束信号，不算成熟。
   - 依据：2026-10-14 用户决定（治理设计稿 v5 §4.0、§4.4）。正式 ADR 待补。
6. **编译时看到哪些线（`path_a_lines.select_known_lines`）。**
   - 编译器判断新内容是接续已有的线还是另开新线，靠的是它看到的投影层（LCE 认知线 + Thread）。每次只读与当前相关的部分，来源有三种：最近活跃（按最近一次挂上 Block 的**消息时间**）、内容唤醒（用这批新消息检索各条线的问题、摘要和支撑 Block：BM25，配置了向量时再加余弦，RRF 融合）、时间唤醒（摘要里写明的时间 `YYYY-MM-DD[ HH:MM]` 已到或 24 小时内将到、且还没结束的线）。
   - 没被读到的线就是这次休眠，仍在投影层里，读出时照样能被查询召回。
   - 每条线连同状态（open / settled）和摘要一起给编译器。线的归属由编译器决定：Path A 入口只按问题文本精确匹配 Thread，不再用"词重叠 ≥ 0.30"（它会把"买电脑"和"买手机"并成一条）。
   - 所有时间都用消息时间，回放可确定。依据：治理设计稿 v5 §4.3。


## 4. 读出口的选择规则（固定，未针对测试集调参）

- 先跑宿主的检索栈。基线模式就是这一步的结果。
- 宿主也可以提供自己的原文检索（`raw_search`，如 AML 适配层的 P1 协议原文检索）。这时基线就是原文命中列表，投影**向上**找到：原文命中 → 由这条原文编译出的 Block → Block 所在的 Thread / LCE 认知线；这是"投影 → 原文"回溯验证链接的反方向。
- 一条已接受认知或活跃 Thread 被选中的条件，满足其一即可：
  - 它的某个支撑 Block 排在本次检索结果的前 `support_rank_window`（默认 10）名之内；
  - 配置了 embedding，且它自身文本与查询的余弦相似度 ≥ `min_cosine`（默认 0.60）。
- 排序：两个条件同时满足（支撑在前列且自身文本与查询相近，两路互相印证）的排最前，然后按支撑 Block 的最好名次，再按向量相似度。最多取 `lce_limit`（5）条认知、`thread_limit`（2）条 Thread。
- 开着的 Thread 读作"进行中的事"：标注 `Ongoing matter, not concluded | last update <最新证据时间>`，支撑 Block 按时间先后排成轨迹。
- 每条被选中的投影，后面依次跟着它的支撑 Block（≤ `support_limit`）和这些 Block 对应的原文（≤ `source_limit`）。支撑 Block 的顺序：先排本次检索里名次靠前的，其余按时间从新到旧。
- 如果某个开着的 Thread 所在 lineage 的 HEAD 比它更新（例如这条线已经被修订过），这个 Thread 不再展示，避免同一条线同时出现新旧两种理解。新旧按双方最新证据的消息时间比较（LCE 的接受时间和宿主的活动时间不是同一个时钟）。
- 之后用 Canonical 检索结果补满 `limit`，已经出现过的 Block 不重复。
- `source_text_for_hits=True` 时，普通检索命中也附带原文，两种模式同样生效；用于把"LCE 的增益"和"原文证据的增益"分开评估。
- `stamp_raw_hits=True` 时，原文命中前加消息时间 `[YYYY-MM-DD HH:MM]`，两种模式同样生效（治理设计稿 G0）。

## 5. 已知限制

- 线的唤醒质量（同一件事有没有被拆成多条线）还没有专门评测；评测时用线清单人工或抽样复核。
- `revise` 沿用编译器给出的当前理解，不检查它和旧版本是否矛盾。认知纠正和负约束账本（ADR-0033 9/28）没有实现。
- 每条认知线最多保留 100 条支撑；超出时丢弃最早的。
- 投影的选择依赖底层检索质量：检索排不出来的事实，对应的认知也不会被选中。证据已丢失的认知，其支撑不再参与检索，只能通过向量相似度被选中；只用 BM25 时它们保留在 LCE 中但不会被读出。
- 编译质量完全取决于宿主所用模型和提示词。
