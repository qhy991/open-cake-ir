# 显式 pointwise epilogue 融合 pass

Compiler 现在提供 `fuse_pointwise_epilogue`。调用者给出 producer 和 epilogue 两个
Schedule，并明确选择一个私有中间结果。pass 检查范围后返回新的融合 Schedule；
`assess()` 和 `lower()` 不会自动调用它。

它对应 OMOE recipe 中可执行的一个动作：将已舍入的 producer 结果直接交给
pointwise epilogue，删除中间全局写入和重新读取。是否值得融合、是否比现有 GEMM
更快仍由 Lab 的候选选择与共同评测决定，Compiler 不读取 recipe 或模型记录。

## 组合边界

当前 Schedule 描述一个 kernel，没有完整的跨 kernel 程序图。因此此 API 定义的是
显式函数组合 `epilogue(producer(inputs))`，不是对任意未提供程序图的优化。

调用者声明 `private_intermediate` 没有这两个阶段之外的消费者。Compiler 能检查
producer 只有这一份输出、epilogue 只有这一份输入，以及两者的访问和类型；它
无法证明外部未提供的程序没有其他消费者。若中间结果需要被其他调用者观察，
不得使用此组合。若 producer 还声明另一份输出，pass 会拒绝。

结果 ABI 是 producer 输入加 epilogue 的一个新输出，输入不变，输出要求独立存储。
consumer 的局部值及输出会被重命名以避免冲突。中间张量从这个新 ABI 消失，
两个原 Schedule 保留不变，调用者可以继续使用它们。

旧阶段的 Workload seal 不适用于新 ABI。结果 `metadata` 为空，必须由任务所有者
绑定完整组合的 Workload，才能进入 Lab 评测。pass 不替调用者伪造这个绑定。

## 首版支持范围

- 两阶段都已通过同一个 Compiler 的检查，并具有相同精确 Target 的 Triton 路线。
- 一行对应一个 program，完整行的 store、reload、最终 store 相互对应。
- 单个 role；warp 分配和 residency 承诺一致，不静默改变性能控制。
- producer 有一个最终 store；中间张量为 BF16 或 FP16，来自一个显式 cast。
- epilogue 只有 load、cast、elementwise 和最终 store，没有额外输入、归约或副作用。
- 不处理循环、持久调度、state、共享/TMEM 分配、同步、动态 extent、scale 关系或视图偏移。

FP32 中间张量目前拒绝。删除一个 FP32 store/load 可能改变后端指令收缩；identity
cast 既不构成可靠的舍入边界，也不是 IR 的合法规范形式。不能以数学公式相同
代替实际数值契约。BF16/FP16 原有的显式 cast 会保留，epilogue 读取它的结果。

匹配不到时返回 `applied=False`、`reason` 和说明，不修改输入或生成半成品候选。
匹配后会对完整结果重新 `assess`，使用现有分析更新寄存器压力、工作量和合法性。
这只是重新检查，不是性能资格。`FusionResult.schedule` 每次返回 assessment 的
独立 JSON 投影，避免候选字典和被检查的源字节变成两份可变权威。

## 使用

```python
from open_cake_ir.compiler import Compiler, frontend

compiler = Compiler.load(".", "compiler/revision.lock.json")
producer = frontend.read_schedule("examples/python/epilogue_producer.py").document
consumer = frontend.read_schedule("examples/python/epilogue_consumer.py").document
result = compiler.fuse_pointwise_epilogue(
    producer, consumer,
    private_intermediate="mid",
    schedule_id="fused_gemm_silu",
    entry_point="fused_gemm_silu",
)
if result.applied:
    schedule = result.schedule
    source = compiler.lower(result.assessment).source
else:
    print(result.reason, result.message)
```

示例的 producer 是 BF16 输入的 contraction+bias，显式舍入为 BF16 后写出；
consumer 重新读取并计算 SiLU。融合后保留这个 cast，原有 `mid` 参数、一次 store
和一次 reload 消失。这是标量乘法/归约表达的 contraction，不声称 Tensor Core
GEMM 或高性能 SwiGLU 已实现。

命令行可生成 checkout 外的候选 JSON、lowered source 和结果说明：

```bash
python3 tools/apply_epilogue_fusion.py \
  --producer examples/python/epilogue_producer.py \
  --epilogue examples/python/epilogue_consumer.py \
  --private-intermediate mid \
  --schedule-id fused_gemm_silu --entry-point fused_gemm_silu \
  --output /Users/haiyan-infiniai/.local/share/open-cake-ir/fused-gemm-silu-demo
```

使用项目支持的 Python >=3.10。输出目录必须是新的、规范的 checkout 外路径。
成功应用退出 0；不适用退出 2，并保留拒绝说明。

## 验证与经验边界

合同检查包括合法但不适用的 FP32 两阶段、不同 target、不同 warp/residency、不同
行映射、额外输入/输出、形状变化和非 pointwise consumer。反例需要先证明原阶段
本身能通过检查，再确认是 pass 自己的适用条件拒绝，而非借用另一个错误。

执行生成源码的 CPU 模型比较两阶段和融合结果，并通过 BF16/FP16 舍入中点检查
故意绕过 cast 的错误候选。它验证的是该 CPU 模型中的数据流和显式舍入，不验证
Triton/ptxas 的实际优化、硬件 transcendental 或 GPU 性能。

需要 GPU 正确性、timing 和 profiler 才能判断是否保留这个候选。低层案例只允许
在已声明相应参考权限的作者环境使用，不能偷偷加入 clean-start。

## English

`Compiler.fuse_pointwise_epilogue` explicitly composes two validated, row-owned Triton
Schedules across a caller-declared private BF16/FP16 intermediate. It preserves the
explicit cast, removes the global store/reload, alpha-renames the consumer, and assesses
the complete result. It is not an automatic graph pass; unknown external consumers and
physical aliasing are outside the supplied composition boundary. Old Workload metadata
is cleared because the resulting ABI is new. Unsupported patterns return a reason.
CPU/source checks establish no GPU, performance or model-level qualification.
