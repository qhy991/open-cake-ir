# AKA qualified-parent IR review：现状与 Lab 计划

状态：2026-09-04（JST）更新。Phase A 已终结；Lab admission 已物化；两条 canary 和两批 5 并发 B200 任务已通过。

## 结论

AKA v6 的 677 个 `qualified` parent 已完成一轮逐条 Open-Cake IR 审查：676 条形成模型结果和独立 verifier 记录，1 条在模型运行前遭遇基础设施故障。676 条中，439 条的**分类结论**被接受，237 条因 reviewer 或 schema 问题被拒绝。

“分类被接受”不等于“可在 GPU 上执行”。当前只有 57 条同时满足 `schedule`、`expressible` 和静态 `lower passed`，可以进入 Lab admission；它们仍需通过独立 oracle 的 B200 完整输出正确性和 sanitizer，才能成为动态有效结果。现已对 12 条 fixed-instance 任务建立该动态证据，剩余 45 条仍待逐条 admission/evaluation；这不是 corpus 覆盖率、性能或训练证据。

## 这项工作的意义

AKA 在这里是 Open-Cake 的外部 challenge corpus，而不是自动进入 Compiler 的测试集。它的价值是用真实、来源完整的 CUDA parent 检查四个边界：现有 Schedule 能否表达；缺口是否真的属于 IR；问题是否应由 Program 或 Workload Contract 负责；静态可 lower 的实现能否在目标 GPU 上保持完整语义。

这使 IR 演进建立在重复、可验证的真实缺口上，避免“一条任务增加一个 primitive”；也把静态表达、GPU 正确性、性能优化和训练资格分开，保留正例、负例和未知结果。

## 当前结果与类别

| 类别 | 数量 | 含义 | 下一责任方 |
| --- | ---: | --- | --- |
| `schedule / expressible / lower passed` | 57 | 当前 IR 能描述该固定 parent，且生成路径通过静态 assessment/lowering；尚未证明 GPU 正确 | Lab admission |
| `ir_gap / not_expressible` | 326 | reviewer 判断完成该 parent 需要现有 IR 无法表达的语义；候选 primitive 仍只是提案 | IR owner：先聚类、复核、审批 |
| `program_composition` | 27 | 需要多 kernel 顺序、reset 后 accumulate、workspace、host scalar 或统一 ABI 等程序级组合 | Program/Executable 层 |
| `insufficient_evidence` | 21 | 当前材料不足以完整确定语义、owner 或可表达性 | 补源码或语义证据 |
| `workload_evidence` | 8 | 缺口已定位在 Workload Contract，例如 shape、dtype、标量、输入域、oracle 或容差不完整 | Workload owner |
| reviewer rejection | 211 | 模型给出的 Schedule 被确定性 verifier 拒绝，常见原因是 access map、shape、dtype、subrange、writer 或 reduction 约束错误 | 修正 authoring；保留真实负例 |
| schema rejection | 26 | 模型输出违反审查 schema 的条件约束，不是 IR 或 GPU 结论 | 修 reviewer 输出合同 |
| infrastructure failure | 1 | 沙箱挂载在模型启动前失败，没有形成可审查结果 | 新建显式 recovery attempt |

以上 439 条 accepted 等于前五类之和；它表示分类可信，不表示全部可执行。237 条 reviewer/schema rejection 加 1 条基础设施失败，共构成本轮保留的 238 个失败。

57 条 Lab 候选内部还有两个优先级：33 条标记为 optimization eligible，24 条为 optimization ineligible。该标记只用于排队；两组都必须先验证动态正确性。24 条原则上止于 correctness，33 条只有在 correctness、sanitizer 和重复性均通过后才能进入性能实验。

## 远端执行状态

- `admission-v1` 已从 676 条 verifier ledger 中确定性物化 57 条候选：33 条优化优先、24 条 correctness-only；253 个 parent/reference/harness 文件全部存在，自包含输入约 7.2 MiB。
- GPU Infra 固定为 `97a2bff`，远端部署重新通过 76/76 测试；独立 daemon、socket 和 state 与共享控制面分离。
- canary v1–v5 分别暴露 judge cwd、home 路径穿越、GPU Infra guard 路径、NumPy 依赖和虚拟环境符号链接解析问题；均保留为 `infra_error/unknown`，没有被重试或改写成 correctness 失败。
- 唯一成功 successor v6 在 NVIDIA B200/sm100 上完成 correctness、memcheck 和 racecheck，三个阶段均 `passed/valid`。四种固定 n=1024 输入在三个阶段共保留 12 个完整输出 artifact；独立标准库复核了 12,288 个元素，位级输出 mismatch、输入 mutation 和 ABI failure 均为 0。
- v6 不包含 benchmark，`frontier_eligible=false` 是预期结果；它只证明该 fixed-instance Cake lowering 和当前 Lab 路径有效。
- 第二条 sigmoid n=17 由远端 `gpt-5.6-sol/max` 从嵌入的冻结证据生成 evaluator，经模型外语法、字节不变、oracle、task schema 和资源模式审查后提交。task-owned runtime successor 在 B200 上完成 correctness、memcheck、racecheck，均 `passed/valid`；9 个完整输出 artifact 经独立 Python 数学/IEEE-float32 复核 306 个输出，oracle mismatch、输入 mutation 和 ABI failure 均为 0。
- 首批 5 并发覆盖 atan-gradient n=9、rowwise add 1×37、global-average-pool backward 1×1×1、rsqrt-gradient n=1 和 GELU-tanh n=4。五条均先通过远端 Sol/max authoring 与外部 verifier，再由 broker 执行；5/5 `completed/valid`，全部 correctness、memcheck、racecheck `passed/valid`。
- 首批 5 条共保留 57 个完整输出 artifact；独立 Python 复核 1,134 个输出，oracle mismatch、输入 mutation、ABI failure、memcheck error 和 racecheck hazard 均为 0。没有运行 benchmark；`frontier_eligible=false` 是预期。
- 第二批 5 并发覆盖 copy-tile 3×4×5、vectorized-copy n=1,048,576、row-replication 257×7×37、vector-add n=1 和 bit-preserving identity n=1。原 verifier 因 Codex 自动初始化目录产生 5 条 `filesystem_policy` rejection，均被保留；显式 v2 只允许已证明的 bootstrap 目录、限制模型变更到三个声明文件并修正过长 task ID，没有重跑模型。
- 第二批 GPU 5/5 `completed/valid`，全部 correctness、memcheck、racecheck `passed/valid`；33 个完整输出 artifact 经独立 Python 复核 13,382,418 个输出，oracle mismatch、输入 mutation、ABI failure、memcheck error 和 racecheck hazard 均为 0。
- 下一批最多 5 条已放行给独立协调任务；每条仍需单独通过 authoring verifier 后才可进入 GPU。当前动态有效数量为 12/57。
- 新增代码的 focused 文档/admission 测试为 9/9，远端 GPU Infra 为 76/76。组合 Torch、Triton 和 jsonschema 环境运行 678 个 Open-Cake contract tests，仅历史 G8 replay/custody 测试失败 1 项；本分支未修改该 Lab 实现或测试字节，因此该既有门禁不被本工作掩盖或修复。

## 下一步计划

1. **固化 Phase A。** 保留 677 条输入、676 条 ledger、原始 receipts 和失败分类；唯一基础设施失败使用新 identity 恢复，不覆盖旧记录，也不重跑 reviewer/schema 真失败。
2. **建立 Lab admission manifest。** 对 57 条逐条绑定 AKA item、parent、accepted receipt、Compiler revision 和 lowered artifact；检查工作负载、完整输出、副作用、独立 oracle、容差和 `sm_100a/B200` 目标。缺任一关键事实就退出 Lab 队列。
3. **去重并按风险分组。** 优先简单 copy/elementwise，其次 reduction，再处理 indexed/atomic/stateful；相同 parent、workload 和 Schedule 语义不重复占用 GPU。
4. **端到端 canary（已完成）。** v6 已验证 compile/launch、完整输出、memcheck、racecheck、结果文件权限和独立复核；该结果解除 5 并发的启动门槛。
5. **使用最多 5 并发（首批已完成）。** `max_in_flight_items=5` 表示最多五个条目同时处于准备、排队或执行状态，不表示同时占用五张 GPU。B200x4 上 exclusive GPU 阶段最多使用实际可用设备，第五条排队；sanitizer、benchmark 和 profiler 必须 exclusive，GPU 只由 broker 分配。
6. **先完成 correctness，再做优化。** 对 admission survivor 运行完整输出 oracle 和 sanitizer；仅其中最多 33 条 optimization 候选进入冻结 baseline 后的稳定性、paired timing 和 profiler 阶段。负结果和基础设施未知分别保留，不自动 retry、reroute 或取消已接受的 sibling。
7. **并行处理非 Lab 类别。** 聚类 326 个 IR gap，只为重复、源码完整且不可由现有 primitive 组合的最小语义提出 Compiler successor；27 个 composition 任务进入 Program 设计；29 个 evidence 任务先补合同；211+26 个拒绝用于改善 authoring/reviewer，而不是消耗 GPU。

## 为什么选择 5 并发

5 并发在吞吐和可审计性之间更合适：它远低于此前的大规模模型审查并发，便于定位首个动态分歧；在四卡节点上允许最多四个 exclusive GPU 阶段运行并保留一个准备或排队槽，不会把“控制器并发”误当成 GPU 数量；同时限制编译产物、sanitizer 日志和 mirror 对磁盘的增长。为消除对其他用户环境的依赖，远端新增约 5.6 GiB task-owned Torch/Triton runtime；第二批 1M 元素完整输出及 home 证据副本约增加 1.4 GiB，当前状态盘约 98% 已用、约 39 GiB 可用。后续必须继续限制 artifact 体积，只清理可重建缓存和重复临时产物，正式证据不可删除。

## 完成标准

本阶段完成不是“57 条都跑过”，而是每条都有唯一终态：`correctness_valid`、确定的语义/实现失败、明确的基础设施未知，或因合同不完整而未 admission。只有 `correctness_valid + sanitizer_valid + repeatability_valid` 的 optimization 候选，才允许进入性能资格；只有经过独立 IR 审批、完整 Corpus Gate 和目标 GPU 验证的重复 gap，才允许形成新的 Compiler Revision。
