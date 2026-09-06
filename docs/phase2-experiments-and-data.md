# 第二批：公开数据与可复现的实验流程

本批新增 1,600 条带标准答案的公开样本，并实现实验清单、逐次持久化、恢复、重评分、配对比较、人工复核模板和重复并发测量。代码测试和本地样例不需要模型 API。本批没有重新测量真实模型，也没有改写历史结果。

## 新增数据

| 数据集 | 开发集 | 冻结测试集 | 测什么 | 自动评分 |
|---|---:|---:|---|---|
| GSM8K | 400 | 400 | 多步算术推理 | 最终数值正确率、输出格式 |
| HotpotQA distractor | 400 | 400 | 根据多个文档回答问题并指出证据 | 答案 EM/F1、支持句 EM/F1、联合指标 |

四份 JSONL 总计 6,074,994 字节，约 5.79 MiB。文件在 `benchmarks/public_data/{gsm8k,hotpotqa}/{dev,test}.jsonl`。

- GSM8K 的开发集来自官方 train，测试集来自官方 test。
- HotpotQA 两份都来自官方带标签的 validation；这里的 test 是本地留出集，不是官方隐藏测试集。两份各有 326 条 bridge、74 条 comparison。
- 固定来源 revision、SHA256、采样 seed；同题和数字替换后的重复题跨划分交集为 0。HotpotQA 还排除了跨划分共享上下文文档标题。这个规则不能排除所有语义近似题，也不能排除模型预训练接触过公开 benchmark。
- 提示词只含问题和提供的上下文。标准答案、参考解法、支持句标签保留在评估端，不传给 Agent。
- `manifest.json` 记录来源、下载时间、排除原因和划分检查；`source_lock.json` 固定下载哈希。缓存重建已验证字节一致。已有冻结测试内容变更时默认拒绝覆盖。
- GSM8K 使用 MIT 许可；HotpotQA 数据及其派生字段使用 CC BY-SA 4.0，许可原文和归属说明随数据保存。

来源、论文和许可详见 [DATASET_CARD](../benchmarks/public_data/DATASET_CARD.md)。原始文件已下载到忽略的 `.cache/`；验证整理后的数据不需要网络或第三方库：

```powershell
python -m benchmarks.public_data.prepare --verify
```

需要重新下载或重建时，先安装 `benchmarks/public_data/requirements.txt` 中的 Parquet 读取依赖；随后运行 `python -m benchmarks.public_data.prepare`。已有完整缓存可加 `--offline`。数据扩充不等于运行时测试覆盖：任务恢复、失败隔离和并发仍由独立的受控测试覆盖；A2A 协议与 Skill 选择/权限等测量留待相应功能实现。

## 安装与一次完整的离线流程

Python 3.11+；核心包只依赖标准库。源码目录中运行：

```powershell
python -m pip install -e ".[test]"
python -B -m pytest -q -p no:cacheprovider
python -m agent_eval experiment --tasks benchmarks/experiments/smoke_tasks.json --config benchmarks/experiments/offline.json --out results/experiments/example --max-runs 4
python -m agent_eval resume results/experiments/example
python -m agent_eval compare --a results/experiments/example --a-condition bare_a --b results/experiments/example --b-condition bare_b --metric answer_correct
python -m agent_eval rescore results/experiments/example --out results/example-rescored.json
```

`offline.json` 是两个相同的无模型基线，每题各重复 3 次，三题共 18 次运行。它只验证实验流程，不是 Agent 改进的证据。已有实验目录不会覆盖；重复执行上述示例时请换新目录，或使用 `resume`。

公开数据可直接作为输入，先生成清单而不执行任何 Agent：

```powershell
python -m agent_eval experiment --tasks benchmarks/public_data/gsm8k/dev.jsonl --config benchmarks/experiments/offline.json --out results/experiments/gsm8k-plan --plan-only
```

这会计划 400 题 × 2 条件 × 3 次 = 2,400 次运行。实际评估时先用开发集调整配置，再冻结模型、提示词和工具配置后运行测试集。GSM8K 对比指标为 `public_numeric_exact`；HotpotQA 可分别比较 `public_answer_f1`、`public_evidence_f1`、`public_joint_f1`，不要把两类任务不同指标混成一个总准确率。

## 实验保存了什么

每个实验目录包含：

- `manifest.json`：任务/数据哈希、实验配置、代码哈希、Git HEAD/dirty 状态、Python/依赖版本、时间预算、随机执行顺序。
- `tasks.json`：当时的完整评估任务快照，含评分参考信息。
- `runs.sqlite3`：预先登记的 `(condition, task_id, trial_id)` 唯一记录，事务提交每次运行的状态和结果。
- `report.json`：可重建的报表，按条件列计划数、完成数、执行/评分失败、缺失分母、已观测 token 和耗时。

任务与 trial 构成配对块，块的顺序及块内条件顺序由固定 seed 打乱。这个 seed 控制实验调度，不代表模型提供方的随机性也被固定。每次都创建新的 Agent；只把 `prompt` 传入 `run`。工厂/适配器用 `module:attribute` 引用，工厂参数通过 `conditions[].config` 传入。

接入项目现有 Agent 有两种方式：提供一个可导入的 `build_agent(**config)`，或使用 `adapters.experiment_agent:build_agent`，在 config 中给出 `llm_factory`、`tools_factory` 及相应配置；适配器使用 `adapters.react_agent_adapter:adapt`。外部代码库路径加入 `source_roots`。模型名、temperature、工具集版本和 token/step 预算应显式写在工厂配置或 metadata 中；后者只作记录，真正执行限制须由工厂设置。凭据通过环境读取，不能写入会保存的 JSON。

默认评分器是规则、工具、轨迹评分；公开数据自动增加 PublicDatasetScorer。自定义评分器配置形如：

```json
{"factory":"my_eval.judges:build_judge_scorer","config":{"model":"your-judge","rubric_version":"v1"},"metric_names":["judge_pass"]}
```

工厂返回实现 `score(task, outcome)` 的 scorer。显式 `metric_names` 保证整个评分进程超时或构建失败时也能登记失败分母。自定义 scorer 依赖、裁判提示词和版本要写入配置并纳入 `source_roots`。默认规则重评分是离线的；显式选择联网裁判仍会调用 API。

## 中断、超时和恢复语义

Agent 执行和评分各有独立子进程期限。Windows 使用 Job Object，POSIX 使用进程组；返回结果前会终止残留子进程并等待主 worker 退出。这里提供进程生命周期隔离，不是执行不可信代码的安全沙箱。

先提交 Agent 输出，再执行评分。评分中断后恢复评分，不再次执行 Agent。已完成的错误也是终态数据，`resume` 不会自动挑失败样本重跑。普通中断在确认本地 worker 清理后保留可恢复记录；突然崩溃遗留的 running/scoring 记录默认拒绝重试，需要核对外部副作用再显式 `--recover-interrupted`，记录的 worker 仍活着时仍拒绝。无法确认清理成功时也保留不确定状态和 PID。

不能承诺远程工具“恰好执行一次”：本地进程退出不等于远端请求回滚。需要写外部系统的工具应自行支持幂等键。源码或保存任务发生变化后不允许继续向同一实验追加结果，应该创建新实验。重评分允许用新评分实现派生新文件，保留原数据。

超时或崩溃可能没有最终 token 用量，`observed_tokens` 仅为已观测数，`cost_unknown_runs` 明确计数；不能把未知费用当作零。无模型基线的 tokens 是词数近似，不是模型账单。耗时同时保存父进程总墙钟时间和 worker 内部耗时，启动与清理开销可见。

## 怎样比较与校准

`compare` 核对数据、任务/trial/group 身份、评分定义和时间预算。默认要求完整有效的配对数据；明确加 `--allow-incomplete` 才生成有缺失的诊断分析，整组排除规则与覆盖率会一并报告。真实错误答案的 0 分会保留；执行或评分不可用不会假装成错误答案或默默删除。

先对同题的重复运行求均值，再对同 group 的题求均值，以 group 为推断单位；没有分组时以 task 为单位。报告 B−A、改善/退化项、配对置信区间和置换检验，并列 token/耗时。推断仍依赖组间独立性与配对标签可交换等实验假设。更换内置评分实现后，可将两个实验用同一配置重评分，再比较派生报告。

旧 `intervention_ladder/analyze.py` 也会保留含错误的条件、拒绝静默覆盖同名条件，并检查任务/trial 覆盖。旧数据缺少任务内容哈希和分组身份时只提供描述性结果，不能据此补造可信的显著性证据。历史 RESULTS.md 中的数字仍是历史记录。

人工复核与裁判分析：

```powershell
python -m agent_eval review-template results/experiments/example --dimensions judge_pass --out results/judge-review.json
# 人工查看任务、答案和轨迹，填写 reference_scores/reviewer/human_reviewed。
python -m agent_eval calibrate --records results/experiments/example --gold results/judge-review.json --out results/judge-calibration.json
```

模板不含裁判分数，默认所有标签为空。真实分析需对含相应 judge 指标的运行导出并复核；上面的无模型样例不会生成 judge_pass。输出哈希可防止将旧答案的人审标签套用到新答案。数值维度支持 MAE/容差一致率，离散维度支持混淆矩阵与 kappa。没有人工标签就报告未校准；`synthetic_program_check` 只验证程序，不能充当人工校准证据。

## 并发与 CI

```powershell
python -m agent_eval concurrency --case-factory benchmarks.runtime_regression.cases:make_cases --workers 3 --repeats 3 --seed 0 --out results/runtime-regression.json
```

每个重复块随机先串行还是并行，每个条件/重复都获得新任务状态。保存排队、执行、总耗时、错误、答案质量及 SLA。只有配对批次全部成功、质量通过且满足 SLA 才发布合格的 speedup CI；含失败的原始时延只作诊断。线程 benchmark 的 timeout 是事后 SLA 判定，会等线程真实结束；需要强制终止的 Agent 实验使用前述进程执行器。

真实 multi_agent_latency 入口已沿用相同配对重复与质量检查逻辑；未定义答案断言的 datetime 任务会标记质量未知，不假装通过。真实模型结果需重新运行后才能给出。

GitHub Actions 配置 Windows/Linux × Python 3.11/3.14，安装依赖后只跑离线测试。wheel 带库、CLI 和受控小样例，不带原始下载缓存或历史结果；公开 JSONL 在源码仓库中使用。这里已在本机验证 wheel 构建及安装后导入/CLI；远端 CI 需推送后才会实际触发。

## 本次验证记录

2026-09-06，Windows / Python 3.14：**406 passed，0 skipped**。构建测试使用本机已有的 setuptools/wheel，没有联网安装构建依赖。测试涵盖原有功能回归、公开数据/评分、计划持久化、恢复不重复执行、评分中断恢复、真实子进程及后代超时清理、清理状态不确定时保留 PID、版本比较、校准与 wheel 导入。

另外实际运行了以下离线流程，结果文件在本机 `results/` 下并被 Git 忽略：

- `experiments/phase2-smoke-20260906/`：三题、两个相同基线、各三次；先完成 4 次，再恢复到完整 18 次。
- `phase2-comparison-20260906.json`：全部三组配对完整，相同基线的答案分数差为 0。只作程序检查。
- `phase2-rescored-20260906.json`：从保存轨迹重评 18 条，原始结果保留。
- `phase2-review-20260906.json`：18 条未标注的复核模板，没有伪造人工分数。
- `experiments/phase2-public-smoke-20260906/`：公开开发集计划 2,400 次，仅执行两个无模型样例验证加载/评分；其余明确记为 pending。它不是完整 benchmark 结果。
- `phase2-runtime-20260906.json`：6 个受控工作负载 × 3 对批次，两臂都保留失败/超时/错误答案；合格 speedup CI 按预期不发布。

公开数据 `prepare --verify` 确认四份各 400 条。没有调用真实模型或付费 API，没有更改历史结果文件或 companion agent 源码。
