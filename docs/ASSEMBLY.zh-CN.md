# 编译器未接入时的工程总装

[English](ASSEMBLY.md)

`examples/assembly-no-compiler.json` 使用显式的 `compiler.type=disabled`。
现有原文存储、canonical 库、原文检索、服务线程和恢复路径可以装配；不生成占位语义，不调用模型。
这属于独立 AML 接入模式；有原生持久历史的宿主仍应使用 SourceRef，不应为总装复制整份对话库。

`/health` 的 HTTP 存活检查仍可成功，但内容报告 `degraded`、`compiler_configured=false`、
`semantic_validation=not_checked`。`/search` 可以读取已有数据；`raw_base=true` 时可检索待处理原文。
`/add` 保存原文后返回可重试的 503，不能把“保存原文”报告为“语义处理完成”。重复交付不重复保存。
窗口模式同样保留待处理输入。`/admin/flush` 在编译器未配置时拒绝执行；此时禁止启用依赖模型的 HyDE。

算法完成后，使用已有 command/openai 后端，或通过 `MemoryRuntime(config, compiler=implementation)`
注入。先关闭旧单写入实例，再打开同一目录，重试原 request_id 和相同 payload；窗口尾部由 flush 处理。
原始时间、来源身份和待处理数据延续，后续 admission、projection、Thread/Path A 使用已有实现。
`compiler_configured=true` 只说明实现已接入，不表示语义质量通过验收。

测试用明确标注的已接受夹具代替推理算法，验证无编译器时持久保存、重启恢复、去重、用户隔离。
原研发工作区没有修改。原有未提交内容在隔离副本中单独形成保留快照；本次没有采纳编译可靠性研究原型。

`path_a_engine=tree`、`search_mode=path_a` 现在会读取已有 Line 图，raw 和 canonical 两条检索路径均已接入。
命中的 Block 会携带受预算约束的相邻步骤及明确记录的推进边，保留 flat 和 baseline 模式。
`read_policy.max_characters` 限制最终搜索文本总长度。读取不需要编译器，图和时态库只读打开。
MR 通过 `MR_MEM_LINE_ROOT=<data_dir>/lce_tree` 消费同一套读出逻辑。
知识时间、用户范围、隐藏策略和预算规则见 [Line 读出契约](LINE_READ_CONTRACT.md)。

有原生历史的宿主现在可使用 `NativeMemory`：侧库仅保存来源版本、批次和接受的编译结果，正文由
宿主按 scope 实时读取。原文编辑/删除自动传播到 canonical 失效；未完成的旧批次取消后重新登记
当前输入，重启可恢复，未配置编译器时保留待处理状态。见 [原生接入契约](NATIVE_INGEST_CONTRACT.md)
和 `examples/native-tree.disabled.json`。Hermes 回合接线位于配套 MR 分支的
`docs/assembly-native-memory.md`。Path B 仍是现有设计明确保留的第二阶段，没有被总装验收代替。
