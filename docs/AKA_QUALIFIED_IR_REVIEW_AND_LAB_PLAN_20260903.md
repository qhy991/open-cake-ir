# AKA qualified-parent IR review：现状与 Lab 计划

状态：2026-09-04（JST）更新。Phase A 已终结；Lab admission 已物化；两条 canary 和十一组最多 5 并发 B200 任务已执行，57 条均有终态。

## 结论

AKA v6 的 677 个 `qualified` parent 已完成一轮逐条 Open-Cake IR 审查：676 条形成模型结果和独立 verifier 记录，1 条在模型运行前遭遇基础设施故障。676 条中，439 条的**分类结论**被接受，237 条因 reviewer 或 schema 问题被拒绝。

“分类被接受”不等于“可在 GPU 上执行”。只有 57 条同时满足 `schedule`、`expressible` 和静态 `lower passed`，进入了 Lab admission；最终 56 条通过独立 oracle 的 B200 完整输出正确性、memcheck、racecheck 和模型外重算，1 条因 authoring schema、runtime custody 和 evidence surface 不合格而保持 GPU `not_run`。这证明 56 个 fixed instance 的当前 Cake lowering 有效，不是 corpus 覆盖率、性能、完整动态 parent 或训练证据。

这 57 条执行中暴露的真实问题均可定位为 reviewer/schema、evaluator/artifact hygiene 或 backend lowering/compile gate；修复后没有留下需要新增 Schedule IR 的动态失败。因此，**不能用这 57 条作为增加 IR 的依据**。326 个 `ir_gap` 的 proposal-only 语义聚类现已完成，但它只产生待 owner 审议的候选，未批准任何 Compiler 变更。

## 这项工作的意义

AKA 在这里是 Open-Cake 的外部 challenge corpus，而不是自动进入 Compiler 的测试集。它的价值是用真实、来源完整的 CUDA parent 检查四个边界：现有 Schedule 能否表达；缺口是否真的属于 IR；问题是否应由 Program 或 Workload Contract 负责；静态可 lower 的实现能否在目标 GPU 上保持完整语义。

这使 IR 演进建立在重复、可验证的真实缺口上，避免“一条任务增加一个 primitive”；也把静态表达、GPU 正确性、性能优化和训练资格分开，保留正例、负例和未知结果。

## 当前结果与类别

| 类别 | 数量 | 含义 | 下一责任方 |
| --- | ---: | --- | --- |
| `schedule / expressible / lower passed` | 57 | 当前 IR 能描述该固定 parent，且生成路径通过静态 assessment/lowering；尚未证明 GPU 正确 | Lab admission |
| `ir_gap / not_expressible` | 326 | reviewer 判断完成该 parent 需要现有 IR 无法表达的语义；296 个 exact names 已被提议归并为 202 个 semantic clusters | IR owner：复核重复簇、冲突簇和来源独立性后审批 |
| `program_composition` | 27 | 需要多 kernel 顺序、reset 后 accumulate、workspace、host scalar 或统一 ABI 等程序级组合 | Program/Executable 层 |
| `insufficient_evidence` | 21 | 当前材料不足以完整确定语义、owner 或可表达性 | 补源码或语义证据 |
| `workload_evidence` | 8 | 缺口已定位在 Workload Contract，例如 shape、dtype、标量、输入域、oracle 或容差不完整 | Workload owner |
| reviewer rejection | 211 | 模型给出的 Schedule 被确定性 verifier 拒绝，常见原因是 access map、shape、dtype、subrange、writer 或 reduction 约束错误 | 修正 authoring；保留真实负例 |
| schema rejection | 26 | 模型输出违反审查 schema 的条件约束，不是 IR 或 GPU 结论 | 修 reviewer 输出合同 |
| infrastructure failure | 1 | 沙箱挂载在模型启动前失败，没有形成可审查结果 | 新建显式 recovery attempt |

以上 439 条 accepted 等于前五类之和；它表示分类可信，不表示全部可执行。237 条 reviewer/schema rejection 加 1 条基础设施失败，共构成本轮保留的 238 个失败。

57 条 Lab 候选内部还有两个优先级：33 条标记为 optimization eligible，24 条为 optimization ineligible。该标记只用于排队；两组都必须先验证动态正确性。24 条原则上止于 correctness，33 条只有在 correctness、sanitizer 和重复性均通过后才能进入性能实验。

## IR-gap 聚类结果

326 个 verifier-accepted `ir_gap` 最初给出 296 个 exact candidate names，说明名称高度碎片化。远端 Sol/max 基于完整嵌入的 compact evidence 提出 202 个语义簇，模型外 verifier 证明 326 个 case 和 296 个 exact name 都恰好出现一次：

- 51 个 `repeated_candidate` 簇覆盖 174 条；
- 150 个 `singleton_or_distinct` 簇覆盖 150 条；
- 1 个 `conflicted_needs_review` 簇覆盖 2 条；冲突来自同名 `segmented_scan` 对边界元素包含规则的相反定义。

较大的重复候选包括 FP32 FMA（12 条）、typed runtime scalar（9 条）、computed FP32 atomic state add（7 条）、INT32-indexed FP32 atomic scatter-add（7 条）、segmented indirect FP32 sum（7 条）和 FP32 natural log（6 条）。这些计数只支持 proposal 排序；它们没有证明来源独立、现有 primitive 不可组合、typed/effect/verifier/lowering 闭包、GPU correctness 或性能，因此 `approval_granted=false`、`implementation_performed=false`、`release_performed=false`。

## 远端执行状态

- `admission-v1` 已从 676 条 verifier ledger 中确定性物化 57 条候选：33 条优化优先、24 条 correctness-only；253 个 parent/reference/harness 文件全部存在，自包含输入约 7.2 MiB。
- GPU Infra 固定为 `97a2bff`，远端部署重新通过 76/76 测试；独立 daemon、socket 和 state 与共享控制面分离。
- canary v1–v5 分别暴露 judge cwd、home 路径穿越、GPU Infra guard 路径、NumPy 依赖和虚拟环境符号链接解析问题；均保留为 `infra_error/unknown`，没有被重试或改写成 correctness 失败。
- 唯一成功 successor v6 在 NVIDIA B200/sm100 上完成 correctness、memcheck 和 racecheck，三个阶段均 `passed/valid`。四种固定 n=1024 输入在三个阶段共保留 12 个完整输出 artifact；独立标准库复核了 12,288 个元素，位级输出 mismatch、输入 mutation 和 ABI failure 均为 0。最初未落盘 standalone verifier JSON，现由保留工件透明重建为 `copy4-canary-v6-independent-verification.current.json`；其 `verification_timestamp=2026-09-03T19:57:51.539423+00:00`、`gpu_rerun=false`、`run_modified=false`，不得倒填为历史已有验证。
- v6 不包含 benchmark，`frontier_eligible=false` 是预期结果；它只证明该 fixed-instance Cake lowering 和当前 Lab 路径有效。
- 第二条 sigmoid n=17 由远端 `gpt-5.6-sol/max` 从嵌入的冻结证据生成 evaluator，经模型外语法、字节不变、oracle、task schema 和资源模式审查后提交。task-owned runtime successor 在 B200 上完成 correctness、memcheck、racecheck，均 `passed/valid`；9 个完整输出 artifact 经独立 Python 数学/IEEE-float32 复核 306 个输出，oracle mismatch、输入 mutation 和 ABI failure 均为 0。对应 standalone JSON 同样是当前从保留工件重建的 `sigmoid-runtime-v2-independent-verification.current.json`，使用相同 `verification_timestamp` 并显式声明未重跑或修改旧 run。
- 首批 5 并发覆盖 atan-gradient n=9、rowwise add 1×37、global-average-pool backward 1×1×1、rsqrt-gradient n=1 和 GELU-tanh n=4。五条均先通过远端 Sol/max authoring 与外部 verifier，再由 broker 执行；5/5 `completed/valid`，全部 correctness、memcheck、racecheck `passed/valid`。
- 首批 5 条共保留 57 个完整输出 artifact；独立 Python 复核 1,134 个输出，oracle mismatch、输入 mutation、ABI failure、memcheck error 和 racecheck hazard 均为 0。没有运行 benchmark；`frontier_eligible=false` 是预期。
- 第二批 5 并发覆盖 copy-tile 3×4×5、vectorized-copy n=1,048,576、row-replication 257×7×37、vector-add n=1 和 bit-preserving identity n=1。原 verifier 因 Codex 自动初始化目录产生 5 条 `filesystem_policy` rejection，均被保留；显式 v2 只允许已证明的 bootstrap 目录、限制模型变更到三个声明文件并修正过长 task ID，没有重跑模型。
- 第二批 GPU 5/5 `completed/valid`，全部 correctness、memcheck、racecheck `passed/valid`；33 个完整输出 artifact 经独立 Python 复核 13,382,418 个输出，oracle mismatch、输入 mutation、ABI failure、memcheck error 和 racecheck hazard 均为 0。
- 第三批 5 并发覆盖 complex-pair layout copy 1×2、cube-gradient n=1、softsign n=1、asin-gradient n=1 和 binary-add n=1。GPU 5/5 `completed/valid`，全部三阶段通过；60 个完整输出 artifact 经独立复核 132 个输出，所有错误计数均为 0。
- 第四批 5 并发覆盖 SELU n=11、channel-shuffle 1×6×8、channelwise affine 1×1、GELU-tanh n=4 和 ReLU n=17。affine 原 run 因 evaluator 未识别 canonical `RACECHECK SUMMARY` 保留为 `infra_error/unknown`；只修摘要解析的 model-free v2 successor 通过。统一复核 54 个完整输出 artifact、1,854 个输出，所有错误计数为 0。
- 第五批 5 并发覆盖 dual-output tile copy n=1、same-shape binary add n=1024、scale n=4099、rowwise broadcast-first add 2×257 和 scale2-axpy-scale n=257。旧 v2 verifier 因硬编码 oracle 字段名拒绝 5 条且未提交 GPU；alias-aware v3 只读取语义等价字段，没有放宽必需事实，5/5 accepted。唯一 GPU runs 全部 `completed/valid`，39 个完整输出 artifact 经独立标准库复核 152,199 个输出，oracle、输入 mutation、ABI、memcheck 和 racecheck 错误均为 0。首条提交成功后本地回执解析器把纯文本 run ID 误作 JSON，空旧台账和控制故障均被保留；该 run 没有重提，其余四条各提交一次。
- 第六批覆盖 softsign gradient n=257、FP16→FP32 n=63,490、reciprocal gradient n=1,024、token-position embedding 3×5×64 和 GELU-backward n=4,099。五条最终均 `completed/valid`；FP16 条目在首次提交前只补 canonical racecheck 摘要解析。GELU 原 run 的 correctness 通过，但 evaluator 的 PyTorch CUDA cache 在 `--leak-check full` 下产生 2 MiB finding，原 run 保留为 `rejected/invalid`；保留同一 leak gate、只显式释放未使用 cache 的新 identity 通过。统一复核 33 个 artifact、851,406 个输出，所有错误计数为 0。
- 第七批覆盖 identity n=1、vol2im identity 48 元素、global-average-pool backward 1×7×1、guarded INT32 store n=1,024 和 tanh-GELU n=23。五条最终均 `completed/valid`；GAP 原 run 的 artifact 缺 `performance_measured=false`，原 `completed/valid` 结果保留但未被外部接纳，只补证据字段的新 identity 通过。统一复核 39 个 artifact、28,707 个输出，所有错误计数为 0。
- 第八批覆盖 strided copy n=30、strided add n=30、scalar pow n=4,097、GELU-backward vec4 n=4,100 和 bias-sum 2×3×259。前四条 `completed/valid`，24 个 artifact 经独立复核 197,790 个输出，所有错误计数为 0。bias-sum 的生成 Triton 使用 `tl.arange(0,6)`，因非二次幂在 GPU compile 失败；node 的 `infra_error/unknown` 结果被细化为 `backend_compile_gate`，没有重试，也不计 IR 语义失败或动态有效。
- 最终 R1 覆盖 bias-sum successor、INT32 identity n=257、GELU-tanh n=4、PReLU 1×3×7 和 variance-affine triplet n=257。bias-sum 由远端 Sol/max 只把六行 reduction pad 到八行并屏蔽两条 lane；真实 GPU-free Triton compile、B200 三阶段和独立重算均通过。R1 五条最终共复核 39 个 artifact、19,974 个输出。
- 最终 R2 覆盖 probability cross-entropy gradient 3×5、softplus gradient n=524,417、四输出 BN parameter postprocess n=257、BF16 in-place bias-gradient state 和 instance-normalization coefficients n=257。BF16 条目的 v1 reviewer 错把“无 returned tensor”当成缺输出；detail-v2 保留拒绝，reviewer-v3 以完整 mutable `dbias` state 加空 tuple 为 observable，且 authoring/kernel/evaluator/oracle/task 均未修改。五条最终复核 39 个 artifact、6,319,488 个 tensor 输出和 144 个 BF16 mutable-state 元素。
- 最终 R3 覆盖 gamma/beta backward、paired partial-gradient reduction、warp-sum32、rows8 ordered sum 和 beta-zero addr outer。三条生成 kernel 先在 GPU-free Triton compile 暴露 store block/rank 或 outer broadcast 缺陷；各自由新远端 Sol/max kernel successor 修复，并重新通过 verifier、真实 compile、唯一 GPU 三阶段和独立重算。R3 五条最终复核 27 个 artifact、41,028 个输出；所有错误计数为 0，所有失败 predecessor 原样保留且不计 IR 失败。
- 唯一非动态有效条目 l000214 的最新 v4 仅为 `model_prepared`：`kernelctl task-check` 拒绝其缺失 `comparison/workloads`；stage 使用旧字段和另一用户 runtime 路径；oracle 使用旧 aliases；evaluator 依赖 `hashlib/base64/zlib` 压缩编码 16M 元素证据。模型外 verifier 将其终结为 `reviewer_schema_and_evidence_surface`、GPU `not_run`、IR gap 未建立。恢复必须使用新 Sol/max authoring，并先满足约 32 GiB 余量下的完整输出磁盘门槛。
- 当前终态为 56/57 dynamic valid、1/57 authoring rejection、0 unknown，且全部 `performance_measured=false`。
- 可发布数据位于 `docs/data/aka-qualified-ir-v6-review-20260904/`：包含 677-row Phase A index、57-row Lab terminal ledger、202-cluster proposal、manifest、dataset card 和最小证据 receipts。导出 verifier 重新证明 677/57 行唯一性、精确 cluster partition 和敏感模式扫描 0 findings。
- 新增代码的 focused 文档/admission 测试为 9/9，远端 GPU Infra 为 76/76。组合 Torch、Triton 和 jsonschema 环境运行 678 个 Open-Cake contract tests，仅历史 G8 replay/custody 测试失败 1 项；本分支未修改该 Lab 实现或测试字节，因此该既有门禁不被本工作掩盖或修复。

## 下一步计划

1. **固化 Phase A。** 保留 677 条输入、676 条 ledger、原始 receipts 和失败分类；唯一基础设施失败使用新 identity 恢复，不覆盖旧记录，也不重跑 reviewer/schema 真失败。
2. **建立 Lab admission manifest。** 对 57 条逐条绑定 AKA item、parent、accepted receipt、Compiler revision 和 lowered artifact；检查工作负载、完整输出、副作用、独立 oracle、容差和 `sm_100a/B200` 目标。缺任一关键事实就退出 Lab 队列。
3. **去重并按风险分组。** 优先简单 copy/elementwise，其次 reduction，再处理 indexed/atomic/stateful；相同 parent、workload 和 Schedule 语义不重复占用 GPU。
4. **端到端 canary（已完成）。** v6 已验证 compile/launch、完整输出、memcheck、racecheck、结果文件权限和独立复核；该结果解除 5 并发的启动门槛。
5. **使用最多 5 并发（首批已完成）。** `max_in_flight_items=5` 表示最多五个条目同时处于准备、排队或执行状态，不表示同时占用五张 GPU。B200x4 上 exclusive GPU 阶段最多使用实际可用设备，第五条排队；sanitizer、benchmark 和 profiler 必须 exclusive，GPU 只由 broker 分配。
6. **先完成 correctness，再做优化。** 对 admission survivor 运行完整输出 oracle 和 sanitizer；仅其中最多 33 条 optimization 候选进入冻结 baseline 后的稳定性、paired timing 和 profiler 阶段。负结果和基础设施未知分别保留，不自动 retry、reroute 或取消已接受的 sibling。
7. **审批 proposal-only IR clusters。** 优先复核 51 个 repeated clusters 的来源独立性、源码完整性和现有 primitive 可组合性；150 个 singleton 默认不扩展 IR，`segmented_scan` 冲突簇先解决边界语义。只有通过 typed/effect/verifier/analysis/lowering 与 positive/near-miss Corpus Gate 的 survivor 才能提出 Compiler successor；27 个 composition 任务进入 Program 设计，29 个 evidence 任务先补合同，211+26 个拒绝用于改善 authoring/reviewer。

## 为什么选择 5 并发

5 并发在吞吐和可审计性之间更合适：它远低于此前的大规模模型审查并发，便于定位首个动态分歧；在四卡节点上允许最多四个 exclusive GPU 阶段运行并保留一个准备或排队槽，不会把“控制器并发”误当成 GPU 数量；同时限制编译产物、sanitizer 日志和 mirror 对磁盘的增长。为消除对其他用户环境的依赖，远端新增约 5.6 GiB task-owned Torch/Triton runtime；第二批 1M 元素完整输出及 home 证据副本约增加 1.4 GiB，当前状态盘约 98% 已用、约 33 GiB 可用。后续必须继续限制 artifact 体积，只清理可重建缓存和重复临时产物，正式证据不可删除。

## 完成标准

本阶段已达到完成标准：57 条均有唯一终态，56 条为 `correctness_valid + sanitizer_valid + independent_recompute_valid`，1 条为明确的 authoring/reviewer rejection，0 条 unknown。只有其中标记 optimization eligible 且后续再取得 `repeatability_valid` 的条目，才允许进入性能资格；只有经过独立 IR 审批、完整 Corpus Gate 和目标 GPU 验证的重复 gap，才允许形成新的 Compiler Revision。
