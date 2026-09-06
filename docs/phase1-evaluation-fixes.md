# 第一批评估修复：评分版本 2

日期：2026-09-05。

本批修复评分正确性、失败记录、检索标注口径和统计边界。核心包仍仅使用
Python 标准库；测试依赖仍为 pytest。没有重新运行付费模型，也没有改写
已经保存的历史 benchmark results。

## 1. 轨迹和任务断言

`TrajectoryStep.tool_calls` 保存同一模型轮的全部调用；每次调用分别保留
`id/name/arguments/observation/status/error/ok`。`step.calls` 和
`outcome.tool_calls` 提供兼容视图，原来的 `action` 仍可读取。
`tool_calls=None` 表示旧格式；显式 `[]` 表示没有记录到执行，不能用挂起的
`action` 占位符伪造一次完成的调用。没有记录结果时不假定成功。

`RuleScorer` 返回三个指标：

| 指标 | 定义 |
|---|---|
| `run_completed` | `success=True` 且 `stop_reason=finished` |
| `answer_correct` | 只检查任务声明的答案要求 |
| `rule_pass` | 运行完成且答案检查通过 |

没有答案断言，或只有空的 `expect_substrings` 时，后两项为 `None`，不再
当作正确答案。动态星期任务目前只确定性检查工具要求，答案正确性仍未评估。

保留 `expect_substrings`，其中数字按完整数值匹配：期望 `391` 不接受
`3910`、`1,391` 或 `-391`，可以接受等值的科学计数表示。普通文本子串
只是轻量断言，不能识别所有否定表达或事实矛盾。

新增声明：

```json
{
  "id": "price",
  "prompt": "Return JSON with total and currency.",
  "expect_number": {"value": 19.9, "abs_tol": 0.01, "path": "/total"},
  "expect_fields": {"/currency": "USD"}
}
```

`expect_number` 也接受一个数值；没有 `path` 时要求答案只有一个数值 token，
所以任务应明确要求只返回最终数值。允许多个数值的解释应改用结构化字段。
`expect_fields` 使用 JSON Pointer；`expect_set={"values":[...],"path":"/..."}`
检查 JSON 数组的集合内容，忽略顺序和重复，布尔值不等同于数字。重复的
JSON 对象键、NaN 和非有限数值不能通过结构化答案检查。

工具使用保留 `used_expected_tool`，增加 `tool_contract_pass`。可声明：

```json
{
  "tool_contract": {
    "required": [
      {"name": "lookup", "arguments": {"key": "price"}, "max_calls": 1},
      {"name": "calculate", "max_calls": 1}
    ],
    "forbidden": ["purchase"],
    "order": [["lookup", "calculate"]],
    "dependencies": [
      {"source": "lookup", "target": "calculate", "argument": "/price"}
    ]
  }
}
```

`required.arguments` 是对象子集；`min_calls` 默认 1，统计参数匹配且成功的
调用，`successful=false` 可接受失败尝试；`max_calls` 统计所有同名尝试。
依赖默认精确匹配，也可声明 `match="contains"`，用于计算表达式包含查询
结果的检查。依赖匹配此前最近一次成功来源的结果，来源必须位于更早的
模型轮。同批预生成的参数不能证明使用了该批刚产生的观察。

`contains` 只能证明数值或文本出现在参数里，不能证明任意程序的完整语义；
要求严格数据流时优先采用结构化参数和 `equals`。这些指标依赖适配器如实
记录轨迹，不提供外部副作用的事务性或恰好一次执行保证。

`tool_error_free` 保留出错事实。`tool_recovery_pass` 只在发生错误时适用：
后续同工具成功且任务完成，可以视为恢复；`allow_errors` 明确允许任务预期的
工具失败。未恢复错误和 fatal 错误仍影响流程分。`trajectory_score` 是上述
流程信号的组合，不再被解释为答案正确率。

## 2. 裁判及失败分母

所有内建裁判要求完整维度、有限数值且位于 `[0,1]`，禁止布尔值冒充数字。
拒绝缺失维度、额外维度、重复键、顶层数组和多段 JSON。为兼容既有服务，
仍接受带说明前缀的单个 JSON 代码块。二值裁判必须返回真正的 `bool`。
会话裁判现在能看到逐轮工具结果和错误，包括结果文本中的来源标识。

`TaskResult`、`TurnRecord` 和 `ConversationResult` 保存评分状态及错误。
构建、执行、适配失败不会让后续独立任务丢失；会话内执行失败后，依赖它的
后续轮标为 `blocked`，下一段独立会话继续。单独的评分器失败不破坏会话历史。
`KeyboardInterrupt` 等控制信号继续向上传播。

报告区分：

| 状态 | 含义 |
|---|---|
| `valid` | 有效分数 |
| `not_applicable` | 评分器明确返回 None，无适用断言 |
| `failed` | 评分器调用、输出校验或指标冲突失败 |
| `execution_error` | 没有可用于评分的执行结果 |
| `blocked` | 依赖的会话步骤已失败 |
| `missing` | 自定义评分器未声明且该记录没有提供指标 |

`aggregate()` 仍返回有效分数的平均值，必须连同 `metric_summary()` 阅读。
后者保存每指标的 `planned/valid/not_applicable/failed/execution_error/blocked/missing`。
`coverage=valid/(planned-not_applicable)`；未执行时适用性未知，保守计入未覆盖部分。
不能只凭有效分数均值判断整体质量。`execution_summary()` 另外报告执行成功率、
执行错误、评分错误和未执行评分器数量；Judge 故障不自动等同于 Agent 答错。

自定义评分器应声明 `metric_names`，使全部失败时仍可确定分母。未声明的
旧评分器仍支持动态数值键，但首次有效输出前无法知道其指标名。裁判可在
构造时显式提供 `dimensions`；内建 builder 会自动提供。

`render()` 和 `dump()` 默认包含上述分母。导出增加 `schema_version=2`、
`scoring_version=2` 及标准化轨迹。全部内容先验证 JSON，再以临时文件原子
替换输出，避免序列化错误破坏已有报告。手工构造的非法 Scorecard 会明确
拒绝导出。恢复实验、重新评分命令及完整模型配置清单留到第二批。

## 3. 检索口径

排名按 ID 的第一次出现稳定去重，再计算 Recall/MRR/nDCG；`k` 必须是正整数。
`relevant_ids` 兼容原来的集合，也接受 `{"doc": grade}`，grade 为有限非负数。
nDCG 使用 trec_eval 的线性 gain，并保留 graded relevance；重复文档不增加分数。

没有正相关标注时，低层指标返回 `None`，这是相对旧版本返回 0 的显式变化。
`evaluate_retrieval` 保留全部查询、ID、状态和检索计数。一次查询检索失败后
继续，其指标为空并记录错误。`average()` 只统计有效行，`counts()` 同时报告
计分、失败、不适用、未标注和无正相关标注的查询数。

NFCorpus 两个脚本使用原始 document 级 qrels，把检索出的 chunk 映射回文档后
评分。文档未成功入库时仍保留原标注和查询，另外报告缺失文档及逐查询覆盖率。
没有自行生成 chunk 级人工标签。检索结果仍受当前管线的 chunk 候选及输出
预算限制，因此并不等同于另一套完整文档排序系统的官方榜单成绩。

分组统计使用同一次执行的观测。配对比较按 query ID 对齐，输出有效配对数量
和排除的查询 ID；有效交集上的质量差异必须连同失败计数解释。

## 4. 统计方法

`bootstrap_ci` 保留连续任务级指标的 percentile bootstrap，并检查有限输入、
有效重采样次数和置信度。独立样本单位应是任务或查询，不能把同一任务的
重复运行或改写当作额外独立题目。

`wilson_ci(successes, n)` 用于二元通过率。例如 50/50 的 95% 区间约为
`[0.9287,1]`；Safety 汇总已接入。连续常数样本的 percentile 区间仍可能是
零宽，不应把该方法直接用于宣称全成功的二元总体概率等于 1。

`paired_permutation_test` 在小样本时枚举全部配对标签交换；大样本 Monte Carlo
采用 plus-one 修正。两对均提升时精确双侧 p 值为 0.5。该检验要求零假设下
每对的条件标签可交换，配对本身不能消除固定顺序、服务漂移或相关样本的影响。
旧 `paired_bootstrap_test` 保留兼容名称并发出弃用提示，实际执行新检验。
结果记录真实方法；不能继续按旧名称解释其统计意义。

## 验证与后续范围

离线回归覆盖正常数据以及故意错误的轨迹、答案、裁判、查询和统计输入。
命令行 bare baseline 与实际兄弟仓库 ReActAgent 的双工具 Mock 调用均已验证。

历史报告未重新计分，v2 修改过的任务声明也不能直接与旧成绩作差。
长程记忆、A2A、Skill、人工裁判校准、重复运行和公平对照实验属于后续批次。
本批没有承诺终止任意阻塞的自定义执行函数；执行器级 deadline 和取消回收
需要在后续实验运行流程中实现。

运行离线测试：

```text
python -m pytest -q tests
```

最终本地验证（Windows / Python 3.14）：252 项通过，0 失败、0 警告。
bare baseline 命令行 4/4 通过，JSON 报告成功写入系统临时目录；实际
ReActAgent 加离线双调用 Mock 验证了第二次工具失败的完整适配。
`git diff --check` 通过。测试未调用真实模型，未重新生成历史 benchmark 成绩。
