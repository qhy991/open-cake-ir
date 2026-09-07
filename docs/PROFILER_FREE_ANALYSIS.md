# 无 GPU 的编译资源分析

`Compiler.profile` 和 `tools/report_schedule_profile.py` 可以在运行 kernel 之前提供
资源反馈。普通静态分析只需要 Python 3.10+；编译资源采集需要支持目标的 Triton 和
CUDA binary utilities，但无需 GPU，也不会初始化 Triton kernel handle。
已经采集的报告可以连同产物带到 macOS 等无 CUDA 环境复用。

## 三种输入阶段，同一份反馈

| 输入 | 可以得到什么 | 仍然不能得到什么 |
| --- | --- | --- |
| Schedule | 运算量、声明资源上界、同步和索引访问风险 | 后端寄存器和隐式共享内存 |
| 编译后的 CUBIN 与 launch 元数据 | 寄存器、静态／动态共享内存、栈和 local allocation，以及更紧的驻留上界 | 实际占用率、stall 百分比、缓存命中、运行时间 |
| 已保留的报告与对应产物 | 不加载 CUDA、Triton、PyTorch 的本地复用 | 另一套工具链重新编译后的资源分配 |

寄存器数量来自 `cuobjdump --dump-resource-usage`，动态共享内存来自编译元数据。
两者都不需要加载或执行 GPU kernel。`STACK` 和 `LOCAL` 保留 binary utility 给出的
字节含义，不能当作运行时 spill 次数或内存流量。
[NVIDIA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)
定义了这些输出字段。

驻留分析仍报告**上界**。实际寄存器和共享内存分配改善了输入，但分配粒度、driver
保留空间、shared-memory carveout、barrier、隐式 tensor memory 和调度可能进一步
降低并发度。不能把上界直接读成耗时或实际利用率。

`work.scheduled_transfer_payload` 另外计算显式 LOAD/STORE 在整个计划中的逻辑载荷，
包含每个分块和循环执行次数，并保留逐操作明细。例如 GEMM 的 A 会随 N 分块重复读取，
B 会随 M 分块重复读取；只统计输入 Buffer 的大小看不到这些复用成本。
访问掩码可能减少载荷，所以这里使用上界；persistent 计划按遍历的工作块计数，动态
停止循环使用真实可推导的 trip 分布。无法建模的访问（如原子读改写）明确返回未知。
这个数字**不包含后端复制、合并、缓存命中和事务粒度的效果**，不是实际 L2/DRAM
字节数，也不能直接除以 HBM 带宽当作运行时间。终端的 `IR read MiB<=` 显示该范围。

## 使用

以下命令在仓库根目录运行，使用已经满足依赖的 Python 环境。输入既可以是 JSON，
也可以是由当前 Python 前端接受的 `.py` Schedule。

只做静态分析：

```bash
python3 tools/report_schedule_profile.py --json \
  corpus/schedules/gemm-bias-b1-smoke.json
```

终端报告在每个 Schedule 的数值行下面展示驻留上界的约束资源、已分析范围、
Finding 的代码、位置、消息、类别、严重程度及阻止阶段，以及已有的 barrier／scoreboard 定性风险原因。
风险标签保持 `uncalibrated_risk`，不代表测得的 stall 比例或瓶颈排序。

JSON 外层仍为 `schema_version=1`；`rows` 和 `skipped` 中的 `findings` 现在保留
Compiler 的完整 Finding 对象，合并 `Assessment.findings` 与 `Assessment.guidance`，
由 `severity` 区分 blocking、report 和 hint；提示不成为阻止条件或测量结果。
需要代码列表的消费者应显式
读取每个对象的 `code`，不能再把对象当字符串。`profile` 保持其原有结构与数值；
已保留编译报告的复用只读取原有编译资源与产物字段，因此既有报告仍可用于该路径。

在具备工具链的 Linux 环境中编译和检查；输出目录必须是仓库外的新目录：

```bash
CUDA_VISIBLE_DEVICES= python3 tools/report_schedule_profile.py \
  --compile-to /tmp/cake-compiled-profile \
  --cuobjdump /usr/local/cuda/bin/cuobjdump \
  corpus/schedules/gemm-bias-b1-smoke.json
```

工具保留 `report.json`，以及按报告行编号的 `0000/lowered.py`、`kernel.ptx` 和
`kernel.cubin`。本轮采集支持 `sm_100a`、不使用多 CTA cluster 的 Triton 路线；CuTe DSL、NVCC 路线和
辅助 global scratch 尚未接入这个采集器。批量报告为不支持的后端保留静态分析，并
明确列出未采集的范围；编译错误仍会中止采集，不替换目标或伪造数值。

把整个输出目录带到无 GPU、无 CUDA 工具链的机器后，可复用原编译观察：

```bash
python3 tools/report_schedule_profile.py --json \
  --compiled-report /tmp/cake-compiled-profile/report.json \
  corpus/schedules/gemm-bias-b1-smoke.json
```

复用会核对保留的源码和 CUBIN 身份，再匹配当前生成源码、目标、kernel entry 与
launch 线程数；不按 Schedule 显示名称猜测匹配。记录携带原编译器与检查器版本。
它是该工具链和该二进制的观察，不保证另一套编译器会分配相同资源。
报告的数值来源是受信任的采集器；身份核对不等于对外部篡改数值的独立机器认证。

同一观察可以进入已有 agent 反馈：

```bash
python3 src/open_cake_ir/tasks/qsa/project_feedback.py compiler \
  --revision compiler/revision.lock.json \
  --compiled-report /tmp/cake-compiled-profile/report.json \
  corpus/schedules/gemm-bias-b1-smoke.json
```

Python 调用通过 `compiler.profile(assessment, compiled_resources=resources)`。
公开方法先重放 Assessment，防止把修改过的 Schedule 与旧编译观察组合。
省略 `compiled_resources` 就只做静态分析；不会隐式申请 GPU 或启动编译工具链。

## QSA 评测反馈

QSA evaluator 为已通过 Compiler 检查的节点保留局部 Finding，包括代码、位置、
消息和原有的两个阻止阶段标志。终态结果经过 `qsa_evaluation_feedback`、
或 `python -m open_cake_ir.tasks.qsa.project_feedback evaluation` 时，
各节点仍保留驻留分析的 `coverage` 与逐资源 `bounds`、profile 的 `abstentions`，
以及 NCU 指标已有的原因和缺失观察。上界、定性风险和未知值的含义不变。

此 Executor 后继版本将 `ncu_abstentions` 从指标名称字符串列表改为指标对象列表；
每个对象包含 `metric`、`estimate_kind`、`value`、`unit`、`coverage`、`reasons`
和 `missing`，未知值仍为 `null`。消费者应读取对象的 `metric` 取得名称，
不再把对象当字符串。已有的 `ncu_estimates` 继续承载非未知指标，不增加平行兼容字段。

输入 profile 若带有 `empirical_cost`，投影只保留 `kind`、`model_id`、
`model_compiler_revision_id`、`model_compiler_revision_sha256`、`target`、`covered`、
`predicted_kernel_us`、`empirical_range_us` 和 `reason`；缺失时不生成此字段。
估算仍以供应方声明的环境为条件。完整的 `context` 和 `reported_evidence` 留在
已有的源模型或报告中，通过模型及 Compiler 身份关联；它们允许供应方附带任意内容，
所以不进入 provider 反馈。此摘要规则同时适用于 Compiler 反馈和终态投影。
当前 QSA evaluator 未请求经验模型，
该能力只保证已有元数据的传递。反馈只包含可定位的摘要，不包含原始样本、完整报告、
生成源码或汇编，不产生新的瓶颈排序或测量。Direct CUDA 保留本臂已有诊断；
基础设施错误仍不可用于指导候选修改，后续请求继续使用原有 task 线程和计数。

## 与 cost model 的关系

这条路径补充后端真实资源特征，帮助定位瓶颈、比较候选和建立后续耗时模型。
它保留已否定的 wave 假设和旧校准结论，未改变公开排序的校准覆盖。
耗时、吞吐率和 stall 比例的数值预测需要独立的目标校准；最终候选仍由外部正确性及
匹配实测验收。当前工作继续推进有适用范围和误差证据的性能估算，而不把资源上界
直接作为运行时间。

## 显式传入经验耗时模型

`Compiler.profile` 现在可以通过同一入口接收外部经验模型。模型查询只使用 CPU，
不会加载 GPU 库或隐式采样，也不会修改已发布的 `calibration_coverage` 或
`Compiler.rank`。静态资源、编译资源和条件耗时预测可以在同一份反馈中查看：

```bash
python3 tools/report_schedule_profile.py --json \
  --cost-model /external/calibration/model.json \
  /external/candidates/schedule.json
```

Python 接口使用 `EmpiricalCostModel.load(path)`，然后调用
`compiler.profile(assessment, cost_model=model)`。现有 Python agent 反馈接口可以直接
消费该结果：`qsa_compiler_feedback(assessment, static_profile=profile.as_dict())`。
任务入口已迁入 QSA 目录，只保留 `compiler` 和 `evaluation`；Ralph 负责生成下一轮状态。

模型用一个通用表示覆盖不同算子：每条曲线固定一个完整 Schedule 模板，明确列出
共同变化的 Buffer 维度、允许的整数对齐和测量区间，再按维度大小插值。这里没有
按算子名称分派的成本公式。模板比较保留 JSON 的值类型；只允许显示名称与模型
明确绑定的尺寸变化。模板中这些尺寸是占位值，不必等于某个测量点的尺寸。
不同 Revision 内容身份、不同目标、范围外或未对齐的尺寸、模板差异，
以及同时命中多条曲线都会返回未覆盖原因。已提供编译资源时，后端编译器版本
与模型声明不一致也会拒绝估算。

模型根对象为 schema_version=2，包含 `model_id`、`compiler_revision_id`、
`compiler_revision_sha256`（来自 Assessment 的现有内容身份）、`target`、
`context`、`reported_evidence` 和 `curves`。`context` 声明 `timer`、`cache_protocol`、
`runtime`（至少有 `compiler_version`）及 `input_scope`。每条曲线包含 `template`、
`varying_dimensions`（`buffer` 和零起始 `dimension`）、`extent_multiple`、按 extent
严格递增的 `points`（`extent`、`kernel_us`）和 `relative_error_envelope`。
`points` 至少包含一个测量点。只有一个点时，仅覆盖该点的精确尺寸，返回该点的
耗时及既有误差范围，不做插值或外推；多个点时仍在测量区间内插值。
模型内容在读取时复制为不可变数据，后续修改输入对象不会改变预测。

JSON 中的 `empirical_cost` 给出 `predicted_kernel_us`、`empirical_range_us`、覆盖状态、
原因、上下文及外部报告的证据。它是 **external empirical cost**：Compiler 校验匹配
关系和插值计算，外部采集者仍负责实测、误差验证和来源真实性。模型声明的环境
不等于当前机器环境已经准入；缺少编译观察时，输出完全以声明的编译环境为条件。
经验残差范围不是置信区间，也不保证全部未测尺寸。该输入不会补写 NCU 的利用率
或 stall 百分比，更不会决定候选接受与发布。

终端同时显示条件估算的经验范围；未覆盖时保留具体拒绝原因，不用空白数值掩盖
目标、Revision、模板或范围的差异。完整上下文和来源仍由 JSON 中的既有字段提供。

校准必须在模型绑定的冻结 Compiler Revision 上完成。早期独立 FMA 原型的旧 schema
与 v46 测量不能仅改写版本名称后充当新 Revision 的模型；需要新的采集或经过明确
审查的兼容证据。本接口不内置未验证的系数。

并行分支曾产生同名 v49 而内容不同的 Compiler；模型因此必须同时绑定编号和
已有的内容身份。旧 schema 1 不会被默认为匹配，需要明确的兼容性验证和新模型
交付。这个身份字段只解决实际的冻结 Revision 对应关系，不建立新的文件摘要目录。
