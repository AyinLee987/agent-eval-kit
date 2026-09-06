# Agent 设计用例与执行轨迹

本批聚焦 Agent 设计，暂不包含 Skill。五个方向各 10 条，共 50 条具体测试规格，阅读 [用例清单](../benchmarks/agent_design/CASES.md)，完整输入、环境、故障和验收条件保存在 [cases.json](../benchmarks/agent_design/cases.json)。

本页保留首次基线与接入过程；后续修复复测、最终预算补测和仍未解决的失败见 [修复验证摘要](agent-design-repair-results.md)。最终评估库离线回归为 691 项通过，下面的早期测试数量对应各自历史阶段。

## 交付与执行状态

| 内容 | 当前状态 | 能说明什么 |
|---|---|---|
| 50 条用例规格 | `specified`，可验证格式与数量 | 已定义测试方法；不表示故障环境和执行适配器全部实现 |
| 本地 Trace 记录与 HTML 查看器 | 已实现 | 能逐步查看真实执行的输入输出、错误和耗时 |
| 真实 ReAct / 编排器的离线示例 | 固定模型响应驱动，无 API 调用 | 验证运行机制与轨迹接入；不构成真实模型成功率 |
| A2A 协议用例 | 已写规格，待实现相应传输与协议适配 | 本地编排器测试不能替代跨进程 A2A 协议验证 |
| 当前 Agent 的实际基线 | 已执行37条：29条真实模型、8条固定刺激驱动原生运行时 | 单次机制验收；另13条能力不支持 |
| 50 条完整对照实验 | 未执行 | 本次只有单基线单trial，不能证明架构收益 |

所有状态修改都要求在用例独立的模拟服务、临时目录或临时数据库内执行。用例里的故障脚本只改变环境响应，不替被测 Agent 决定重试、去重或下一步动作。没有相应能力时报告 `unsupported` 并单列覆盖率，不能把未执行算成通过。

## 评估方法

用同一个模型、同一任务和同一初始环境进行配对对照；固定基础提示词、工具契约、故障触发规则和预算，只改变待测机制。不同条件必须重新创建环境，不能复用已经消耗过的故障计数或已经写入的记录。

优先验收外部状态、必要约束与终止结果。例如，写入响应丢失后重试是否产生两条记录，由模拟数据库判断；不能仅凭 Agent 声称“完成”判通过。只有具有业务必要性的前后依赖才约束调用顺序，允许其他有效路径。

机制测试使用固定模型响应驱动真实 Agent，可以隔离运行时问题。真实模型实验另行运行，比较整体任务成功率、故障恢复率、重复写入率、约束满足率，以及全部子 Agent 的总 Token、调用次数和端到端时间。重复运行按任务或场景族聚合，不能把同一场景的重复 trial 当作独立样本。

用例规格不是现有 `experiment --tasks` 接受的问答任务；直接提取 `prompt` 会丢失故障与环境。后续执行适配器必须读取完整 case，在评估端建立环境和状态判定器，仅把用户提示和可见工具信息提供给 Agent。标准验收、故障脚本和预期状态不能进入模型提示。

```powershell
python -B -m agent_eval design-cases
```

这条命令验证 50 条规格，不运行模型或 Agent。

## 轨迹记录范围

`TraceRecorder` 使用一次评估一条 Trace，关联 `task_id`、`condition`、`trial_id`，每次尝试有独立 ID。Span 通过 `parent_span_id` 表达父子关系；模型调用、工具调用和上下文处理在调用发生时记录开始和结束。

| 层级 | 记录内容 |
|---|---|
| 评估运行 | 用户输入、关联 ID、规范化结果、整体执行状态 |
| Agent | 输入、初始预算、终止原因、步数、用量、最终输出 |
| 模型调用 | 实际 messages 与工具定义、模型返回的文本与 tool calls、提供方返回的 usage、错误和耗时 |
| 工具分发 | 工具参数、调用 ID、返回观察、可恢复错误、预算状态 |
| 工具实际执行 | 底层执行尝试及原始异常；参数校验失败不会伪造底层调用 |
| 上下文处理 | 整理或压缩前后的 messages，以及其内部已接入的模型调用 |
| 子 Agent | 显式传递父 Span 并包装 Worker 后，记录 Worker 内部调用树 |

记录的是程序可以观察到的消息和行为，不是模型不可见的内部思维链。默认保留提供给记录器的完整内容，并对常见密钥字段、Bearer 和明显密钥文本脱敏；任意自然语言中的隐私信息不保证自动识别。

目前适配原生 ReActAgent 的同步 `run`，以及现有 DeepSeek 评估外层包装。异步／流式入口、提供方内部 HTTP 重试、未接入的远程进程、自定义记忆服务或工具内部网络请求没有自动全覆盖。覆盖范围会写入 `trace_limitations`。原生适配器使用该项目的内部执行边界，升级 companion 后需要重跑兼容测试。

## 使用方式

现有 durable experiment 配置增加：

```json
{
  "trace": {
    "enabled": true,
    "instrumentation": "adapters.trace_agent:trace_agent"
  }
}
```

把这段合并到原有配置顶层，保留 `conditions`、预算和评分配置。`instrumentation` 可省略，此时只记录评估运行整体；不会自动获得工具和模型调用详情。原有提供方的 outcome adapter 保持原样，评估器负责把 Trace 关联写入结果。

```powershell
python -B -m agent_eval experiment --tasks <任务文件> --config <配置文件> --out <新实验目录>
python -B -m agent_eval trace-view <实验目录> --out <输出页面.html>
```

每个实验会生成 `traces/<trace_id>.jsonl` 和同名 `.manifest.json`，`report.json` 的逐题记录中有 `trace` 链接。HTML 是独立离线页面，可搜索用例、展开调用树、查看输入输出和关联分数。数据留在本地，页面不调用网络，也没有接入 Langfuse 服务。

离线执行示例：

```powershell
python -B -m benchmarks.agent_design.trace_smoke --harness-path ../agent/agent-harness-from-scratch --out results/agent-design-trace-smoke
python -B -m agent_eval trace-view results/agent-design-trace-smoke --out results/agent-design-trace-smoke/index.html
```

示例使用真实 ReActAgent／本地编排器和固定模型响应，不读取 API 密钥、不产生模型费用。其报告中的检查只验证该示例的机制断言，不能视为 50 条规格已验收，也不能视为真实模型的评估成绩。

## 中断与完整性

事件逐条追加并 flush，清单记录已写入数量和未闭合 Span。进程被超时终止后，可读取已写入事件；缺失的结束事件标记为不完整，不补造成功或时间。文件缺失、序号缺口、损坏行、日志写入失败都有诊断。

`trace.complete=true` 仅表示记录完整。任务发生致命错误但已记录全部结束事件时，轨迹仍可完整，任务仍是失败。日志 I/O 失败降级为诊断，不改变 Agent 返回结果。flush 不等同于断电级持久化，机器突然断电仍可能丢失尚未落盘的数据。

恢复和重评分沿用已保存的轨迹。中断任务再次执行会创建新 Trace，保留旧尝试链接，避免拼接两次执行。启用记录会有磁盘和序列化开销；对照条件应使用相同配置，实验协议会区分是否开启记录。

## 与 Langfuse 的关系

这里实现的是本地 Trace／嵌套 Span 记录和查看，结构借鉴通用可观测性方式。Langfuse 把模型、工具等 observation 嵌套在同一 trace 下，并支持 OpenTelemetry 接入；见 [官方数据模型](https://langfuse.com/docs/observability/data-model)。

本地 JSONL 不是 OTLP，也不是 Langfuse 直接导入格式。本批没有安装或部署 Langfuse、配置账号、上传数据。需要接入时可以另加受控 exporter，映射 Trace／Span ID、时间、输入输出和 usage，并处理短进程退出前 flush，见 [官方 SDK 文档](https://langfuse.com/docs/observability/sdk/overview)。

## 本次验证（2026-09-06）

完整离线回归与打包检查：**535 passed**。50 条规格通过数量、结构和文档一致性验证。5 个固定响应驱动的真实运行示例全部通过机制检查，5 条轨迹全部闭合，共记录 63 个 Span；模型 API 调用为 0。

示例报告与独立页面保存在本地 `results/agent-design-trace-smoke-verified/` 下的 `report.json` 和 `index.html`。这些结果位于 Git 忽略目录，远端仓库和发布包不包含它们；上面的离线命令可生成新的示例。已人工查看页面布局，调用树能分别显示 Agent 错误状态和工具成功结果。

测试明确覆盖超时保留部分轨迹、恢复/重评分不重跑已完成任务、日志失败不改变结果、凭证脱敏、历史尝试关联，以及模型文本不能在导出页面中执行脚本。5 个示例中的预算保护检查通过，但该示例的 Agent 任务成功标记仍为 false；没有把保护机制生效混成任务完成。

## 实际基线运行与离线复核（2026-09-06）

执行适配器位于 `benchmarks/agent_design/live_*.py`。工具/状态数据均为独立沙箱；真实模型自主决定工具调用，故障环境不替它重试。精确循环、token边界、取消和deadline用例需要可重复的刺激，因此BUDGET-03至10使用固定响应驱动真实运行时，单独统计，不称为模型实跑。

| 方向 | 机制通过 | 失败 | 未完整评定 | 不支持 |
|---|---:|---:|---:|---:|
| 工具恢复 | 6 | 1 | 3 | 0 |
| 状态与幂等 | 7 | 0 | 0 | 3 |
| 预算与终止 | 6 | 4 | 0 | 0 |
| 上下文与记忆 | 7 | 3 | 0 | 0 |
| A2A协议 | 0 | 0 | 0 | 10 |

29条真实模型用例为22通过、4失败、3未完整评定；8条确定性运行时用例为4通过、4失败。DeepSeek返回95次有效usage，全部响应模型为`deepseek-v4-flash`，非思考模式；输入66,780、输出8,275，共75,055 token。固定模型的虚拟用量不计入费用。37条轨迹全部完整，548个Span，没有日志诊断错误。

本轮发现并修正评分器假阴性：说明文字包围唯一完整JSON对象、TOOL-10错误匹配问题类型、STATE-06合法状态别名、CONTEXT-04只读约束同义表达。对相同已保存答案、实际工具观察和最终状态做纯函数重评分，未重新运行Agent或工具，额外API调用为0。原始统计15通过/19失败/3未评原样保留；修正后26通过/8失败/3未评。11条契约结果改变，另4条仅子指标改变。每个派生评分记录来源清单和评分代码哈希。评分修正不是Agent能力提升，后续实验应冻结修正后的oracle。

未完整评定的3条：TOOL-03模型一次给出合法参数，未触发参数错误；TOOL-05直接选副本，未触发主服务超时；TOOL-06内容恢复成功，但没有完整的输出schema验证机制。未支持的13条是STATE-03崩溃窗口持久化恢复、STATE-09租约重分配/fencing、STATE-10持久outbox，以及A2A的10条协议规格。

“机制通过”包括正确失败，例如权限永久拒绝后终止，或补偿完成后如实报告订单失败；不等于业务任务成功。状态用例的服务端幂等由沙箱提供，不能归因成Agent默认具备幂等。部分截止时间用虚拟环境时钟；BUDGET-06/07/09测真实墙钟。包含摘要的根总量统一按提供方真实usage验收，不能漏算摘要费用。单条请求可能越过观察用量阈值，保护器不是严格账单上限。

首次运行（会产生模型费用，仅读取环境或companion `.env`中的指定DeepSeek凭证，不复制凭证）：

```powershell
python -X utf8 -B -m benchmarks.agent_design.live --harness-path ../agent/agent-harness-from-scratch --out results/experiments/my-design-run --workers 4
```

增加`--preflight`只执行TOOL-01；随后加`--resume`继续其余用例，已完成的用例不重跑。源码变化会拒绝续跑；原运行完成后仅修评分器时，使用离线派生报告，不覆盖原始结果：

```powershell
python -X utf8 -B -m benchmarks.agent_design.rescore_live --source results/experiments/my-design-run --out results/experiments/my-design-run-reviewed
```

本轮新增与修正的针对性离线验证：119项通过，覆盖真实运行入口、独立状态oracle、JSON歧义拒绝、超额记账、硬中断用量恢复与不重跑的离线评分。

本地复核目录为 `results/experiments/agent-design-deepseek-v4-flash-20260906-reviewed/`，包含 `REPORT.md`、`index.html` 和 `BADCASES.md`；保留的原始目录名去掉 `-reviewed` 后缀。这些完整记录没有上传到 Git。公开的后续结果和失败分析见 [修复验证摘要](agent-design-repair-results.md)。
