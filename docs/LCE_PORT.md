# LCE 移植进 MR-Mem：方案与旧实现审核（草案）

- 日期：2026-10-14
- 状态：方案已确认（第 4 节），按第 5 节实施。
- 依据：`MEMORY_GOVERNANCE_DESIGN_DRAFT_2026-10-13.md` v6（下称"设计稿"），尤其 4.0、4.4、4.5、5.x。
- 移植来源：LCE `origin/master` = `5228475`（MR-Mem-pathA 的 venv 里装的就是这个提交）。本地 LCE 仓库检出的是实验分支，不作为来源。
- 移植目标：MR-Mem-pathA。LCE 和 MR-Mem 是两个独立项目；本方案把 LCE 的代码移植进 MR-Mem，MR-Mem 不再通过安装包依赖 LCE。

---

## 1. 架构基线（以设计稿为准）

```
原文（唯一权威）
  └─ MR-Mem 编译 → SB（投影最低层，事件被切成的点）
       └─ LCE（整体概念），三部分并列：
            ├─ Thread：进行时，正在进行的分支（worktree）
            ├─ Path A：过去时，已锁定的树
            └─ Path B：未来时的可能性，从过去时推出，可证伪
```

**Path A 和 Thread 是显性的工作，由 LLM 在上下文里串联：**

- 话题开始时，检索把合适的线（Thread 分支、Path A 树上的相关节点）放进编译输入，即 `KNOWN_LINES`（设计稿 4.3）。
- 接哪条线、从哪个节点分叉、这一步锁定了没有，都由编译器（LLM）读语义判断。
- 这里**不用**向量互近邻、证据收敛、Pareto 竞争来决定挂线。

**LCE 的结构发现属于 Path B：** 互近邻轨迹、证据收敛、Surface、灵感素材。它们从已经形成的过去时里发现隐含的结构和跨经历的关联，产出可证伪的假设，不决定 Path A 的接线。

---

## 2. LCE 模块分类与审核

审核标准：是否符合设计稿 v6 的逻辑。结论分三类：**移植**、**移植但改造**、**不移植**（MR-Mem 已有对应物，或属于 LCE 单独运行时才需要的部分）。

### 2.1 Path A / Thread 用到的部分

| 模块 | 内容 | 审核结论 |
|---|---|---|
| `cognition/line_graph.py` 的 `LineGraphStore` | Line 的稳定身份；节点 = SB；父子边；分支（多子）、汇合（多父，默认关闭）；拒绝成环和冗余捷径边；成员和边按知道时间做修订，可退役重建，历史保留 | **移植。** 这就是设计稿 4.5 的树。父边就是 Thread 指向分叉点的索引。 |
| `line_graph.py` 的 `LineGraphView` | 按知道时间截断的可见视图；`frontier`；`paths_to_frontier`；`raw_closure`（任一节点追回全部原文） | **移植。** 回溯顺着边走，不做全库检索（4.5）；原文溯源对应双向索引的向下方向（5.5）。 |
| `line_graph.py` 的 `CallableLineProjector` | 在线上找和当前内容最相关的节点，取周围有界的一小段，带原文溯源 | **移植但改造。** 现在只按向量找节点、只取固定 6 个节点。读出要"按问题取树上对应部分"（4.5）：问整段经历时取整棵树这一层的概括，问吃饭时取吃饭分支。需要的是按问题选子树，而不是固定大小的邻域。 |
| `line_graph.py` 的 `LineAssembler` | 新线的身份认定要有足够独立证据；只有唯一的非劣候选才能继承旧线；拒绝自动合并 | **不用于 Path A。** 这是 Path B 的收敛逻辑。Path A 由编译器决定接线，直接调用 store 的原语（建节点、加边）。保留给 Path B。 |
| `cognition/worktree.py`、`promotion.py`、`core/engine.py`、`store/sqlite_store.py`（Worktree → Baseline） | OPEN 草稿积累"重复支撑、多点支撑"后按策略晋升为 Baseline；Baseline 是一个区域一段文本，带修订链 | **改造，见第 4 节第 1 条。** 设计稿的锁定是"这一步在当前时间点状态锁定了没有"，由编译器逐步判断，不是靠支撑次数。Baseline 是"一段文本 + 修订链"，不是树。 |
| `cognition/rejection.py` | 否认记录：按支撑签名禁止同样的证据再推出被否认的结论，可以重开 | **移植。** 移植后它就在 MR-Mem 里，满足"拒绝记录只有一处"（设计稿 3.3）。G4 不再新做。 |
| `cognition/invalidation.py` | 原文失效时，定向传播到受影响的推导 | **移植。** 对应"原文丢失时投影保留、标灰"（设计稿 1）。审核点：是否会把投影隐藏（与"不隐藏"冲突）。 |
| `contracts/baseline.py` 等契约 | 数据契约 | 跟随上面的取舍移植需要的部分。 |

### 2.2 Path B 用到的部分（第二阶段移植）

| 模块 | 内容 | 审核结论 |
|---|---|---|
| `structure/trajectory.py` | 互近邻轨迹提议，Point-Cloud 慢路径 | Path B：从过去时里发现隐含轨迹。 |
| `cognition/convergence.py` | 去中心化证据收敛，Pareto 竞争 | Path B：对推出的结构做证据认定。 |
| `structure/surface.py` | 跨 Line 的路径形状比对 | Path B：跨经历的相似结构。 |
| `cognition/inspiration.py` | 关联素材、延伸素材（A→B→C 推 D?） | Path B：未来时的可能性（设计稿 4.0 的买手机推荐小米）。 |
| `structure/discovery.py`、`structure/frontier.py`（06R 旧路径） | 快照式结构观察、frontier 吸收打分 | 先审核是否还需要。trajectory 运行时文档已把它列为待退役的兼容路径。 |

### 2.3 不移植

| 模块 | 原因 |
|---|---|
| `reference_memory/*`（LCE 自带的原文与 SB 存储） | MR-Mem 本身就是这个底座。 |
| `semantic/*`（LCE 自带的 SB 编译器） | MR-Mem 的编译器是唯一编译权威。 |
| `integrations/mr_mem*.py`、`cognition/block_adapter.py`、`cognition/external.py` | 两个项目之间的桥。移植后不再需要。 |
| `read_api.py`（词重叠的 Understanding 读接口） | 由 MR-Mem 的读出口取代。 |
| `runtime.py`、`testing/*` | LCE 单独运行的组装和测试替身。测试按移植后的接口重写。 |

---

## 3. MR-Mem-pathA 里与之重复、需要替换的部分

| 现有代码 | 问题 | 移植后 |
|---|---|---|
| `memory/threading.py`、`memory/product.py` 的 `MemoryThread`（ADR-0033） | 另一套 Thread 存储和成熟规则；一条 Thread 只有一组扁平的支撑（最多 8 个），没有树，没有分支 | 由 Line 树上正在进行的分支取代 |
| `integrations/lce_path_a.py` | 每条 Thread 在 Baseline 里开一条扁平的线；修订就是追加文本 | 由 Line 树取代 |
| `integrations/path_a_ingest.py` | 把编译器信号转成 Thread 事件和 commit / revise | 改为：编译器信号 → 在树上建节点、加边、标锁定 |
| `integrations/path_a_read.py` | 平铺版和"一件事一条"版，都是重复造轮子 | 由改造后的局部投影取代 |
| `integrations/path_a_lines.py`（KNOWN_LINES 选取：最近活跃 + 内容唤醒 + 时间唤醒） | **方向对**，就是"话题开始时检索合适的线放进来" | 保留；候选从 Thread 改为树上的分支和节点 |
| `mr_mem_aml/compiler.py` 的线信号（thread / commit） | 只有"接哪条线、结束没有"，没有"从哪个节点分叉"，也没有逐步锁定 | 输出格式改为：这个 SB 接在哪个节点下（父节点）、这一步锁定了没有 |

---

## 4. 已定（用户 2026-10-14 确认）

1. **锁定不需要另外表示。** Thread 和 Path A 本身就是这个区分：一个节点在 Thread（正在进行的分支，worktree）里就是进行时，被合并进 Path A 的树就是已锁定。锁定 = 编译器判断这一步锁定后，把它从 Thread 合并进树（相当于 git 提交），不加状态字段。LCE 旧的"Worktree 靠重复支撑晋升为 Baseline"的策略不用于 Path A：晋升条件是编译器的锁定判断，不是支撑次数。
2. **同一个 SB 里有锁定和未锁定的部分时，由编译器切成两个 SB**（"约了"与"明天吃饭"），符合设计稿 4.1。

---

3. **没有 main。** Thread = 未 commit；Path A = 已在该分支上 commit。分支不合并进主干，去中心化；不同维度的树通过共享 SB 交汇（`LineGraphStore` 已支持一个 Block 属于多条 Line）。编译器对一个 SB 可给出多个挂接。
4. **失效传播（`invalidation.py`）不移植到 Path A：** 它把受影响的快照、草稿、Baseline 标记重建并隐藏，与"标灰不隐藏"冲突；Line 视图读节点时直接查原文是否还在。其快照部分随 Path B。

## 4.5 审核发现（逐模块读代码，2026-10-14）

| # | 位置 | 旧实现 | 与设计的冲突 | 移植时的处理 |
|---|---|---|---|---|
| 1 | `LineGraphView._node_visible` / `state_for_node_at_cutoff` | 节点可见要求自身原文有效**且所有祖先可见**；上游一个 SB 原文失效，整棵子树隐藏 | 违反"原文丢失时投影保留、标灰、降置信度，不隐藏"（设计稿 1） | 可见性只看成员关系是否有效；原文失效的节点和其后代照常可读，标为证据不完整 |
| 2 | `CallableLineProjector.project_for_block` | 用当前 SB 的向量找最近节点作锚点，按图距离取固定 6 个节点 | 设计要"按问题取树上对应部分"（4.5）：问整段经历取整棵树的概括，问一个局部取那条分支；锚点应来自原文命中向上的反向索引（5.5） | 锚点 = 查询命中的原文 → 编出的 SB → 所在节点；范围按问题取子树/分支，不是固定个数 |
| 3 | `LineAssembler`（新线身份认定、唯一非劣继承、拒绝自动合并） | 向量 + 证据收敛决定挂线 | 属于 Path B；Path A 由编译器决定 | Path A 直接用 store 原语；`LineAssembler` 随 Path B 第二阶段移植 |
| 4 | `ensure_node` / `add_edge` 的时间 | `knowledge_at` 缺省取 `datetime.now()` | 回放要确定，时间按消息算（设计稿 4.3） | 一律显式传入消息对应的 known_at；`created_at` 只作审计 |
| 5 | `rejection.py` | 按"区域 + 内容哈希 + 支撑签名"精确匹配 | LLM 改个说法就能绕过同一条被否认的推断 | 保留精确匹配作最后一道闸；编译时把相关拒绝记录随 KNOWN_LINES 一起交给编译器（设计稿 4.3 已写"同一次检索带出拒绝记录"）。记录只此一处 |
| 6 | 派生指纹 `derivation_fingerprint` | 记录向量/算法版本 | Path A 的派生来自编译器 | Path A 的指纹 = 编译提示词版本 |
| 7 | 进行中的分支（Thread）存在哪 | LCE 的 Worktree（`DraftRevision`）是给 Baseline 的一段候选文本，不是一串 SB 节点 | 设计要 Thread = 从树上某节点分出、正在进行的一串 SB | 复用 MR-Mem 现有 `MemoryThread`（问题、摘要、成员），改造：加分叉点索引（树上节点 ID），成员不再限 8 个；不新建第三套存储 |

---

## 5. 实施顺序（确认后）

1. 移植 `LineGraphStore`、`LineGraphView`、`rejection`、`invalidation` 进 MR-Mem，按 MR-Mem 的存储和契约改写；带上原有测试。
2. 编译器输出改为"父节点 + 这一步是否锁定（锁定则合并进树，否则留在 Thread）"；KNOWN_LINES 的候选改为树上的分支和节点。
3. 读出改为按问题取子树投影。
4. 删掉 MR-Mem 侧重复的 Thread、扁平 lineage、读出代码。
5. 只编译一段 BEAM 100k 对话，把树拿给用户看：节点、分支、锁定、索引。
6. 用户认可后，再全量重编和跑端到端。Path B（2.2 节）作为第二阶段。
