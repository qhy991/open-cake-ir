# AMD 入门：精确支持 gfx1151

[English](../GETTING_STARTED_AMD.md) · [中文首页](README.md) · [文档总目录](../README.md)

当前 AMD 路径只面向 **gfx1151、HIP 和 32 个线程一组的 wave32**。
Schedule 仍写 `lowering.backend=triton`，硬件事实统一由 Compiler 的 Target 负责。
生成源码、编译 GPU 二进制、算对答案、测量速度和分析瓶颈，各需要自己的证据。
当前 Compiler 版本看[自动生成的状态页](../../reports/current/STATUS.md)。

当前开发分支是 `codex/amd-gfx1151-consolidated`，它把已验收 AMD 改动 rebase 到 main 的
B300 支持与 task 分工上。[`tasks/amd`](../../src/open_cake_ir/tasks/amd)负责 AMD 的合同检查、
标准答案和搜索决定；[`tasks/workloads.py`](../../src/open_cake_ir/tasks/workloads.py)负责选择
明确登记的冻结合同。共同 Evaluation 只保留可复用的 HIP 产物和 profiler 工具，不负责分派 AMD 算子。
这些独立入口不会因为登记成功，就自动获得通用 TaskLab 或 native-tensor ABI。

## 现在能表达什么

| 路径 | 内容与边界 |
| --- | --- |
| FP32 SwiGLU | 输入是两张独立的表；有 CPU FP64 标准答案和只检查正确性的入口。 |
| FP32 RMSNorm 加权 | 先对每行归一化，再乘权重；输入的地址和内容必须保持不变。 |
| Q4_0/Q8_1 原始记录 | 用 UINT8/INT8 表达明确大小的字节记录，并检查访问范围。 |
| Q8_1 producer | 从浮点输入生成量化记录；实机检查需要逐字节核对全部 576 字节，包括 15 条全零填充记录。 |
| RMSNorm 单行搜索 | 四种 wave 数、固定基线、B/B ABBA 噪声检查和原始数据重放；正式 Search Contract 仍待冻结。 |
| AITER 与 rocprofv3 | 分别提供外部基线准备和 profiler 交接入口；不能据此宣布已经完成公平比较或提速。 |

本轮不实现 Q4_0/Q8_1 MMVQ consumer。要先在真实 gfx1151 上证明 producer 输出的字节正确，
再进入消费这些记录的计算阶段。这里的叶子算子结果，也不能替代完整 llama.cpp、模型层或服务结果。

## 不用 GPU，先检查并生成源码

先按[首次使用](../GETTING_STARTED.md)准备已有的项目 Python 环境。在仓库根目录执行：

```bash
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler assess --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/swiglu-b8-smoke-gfx1151.json

AMD_SOURCE_DIR=$(mktemp -d)
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler lower --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/swiglu-b8-smoke-gfx1151.json \
  --output "$AMD_SOURCE_DIR/swiglu.py"
```

命令检查完整计划，并把源码写到仓库外。它不会加载 HIP，也不会启动 GPU kernel。
输出中的执行要求必须保持为：

```json
{
  "target": "gfx1151",
  "triton_target": {"backend": "hip", "arch": "gfx1151", "warp_size": 32},
  "binary_role": "hsaco",
  "assembly_role": "amdgcn"
}
```

HSACO 是要执行的二进制，AMDGCN 是保留的汇编材料。目标、指令、类型或地址规则不匹配时，
诊断会指出位置；Compiler 不能私自换另一种 GPU。
gfx1151 尚无合格的驻留与排名校准。`RESIDENCY_TARGET_UNMODELED` 表示这部分没有模型覆盖，
不是资源一定够用。搜索准备会记录 `ranking_applied=false`，不会借用 NVIDIA 的校准数字。

## 为什么量化记录要写清楚

`ggml_q4_0_v1` 一条记录占 18 字节，`ggml_q8_1_v1` 占 36 字节。
[`ir/resources.py`](../../src/open_cake_ir/compiler/ir/resources.py)中的 relation 把“第几条记录”
和实际字节连接起来，不要求作者另学一套 layout 语言。

Q8 producer 由小操作组合而成：取绝对值、安全除法、明确的舍入、改变临时值的分组形状、
wave32 XOR 归约、类型转换和按字段存储。类型转换只有 `cast` 一种写法：

- FP32 到 FP16：`rounding=nearest_even, overflow=ieee`。
- FP32 到 INT8：`rounding=toward_zero, overflow=forbid`。
- Q8 存储按登记顺序接收 `d:fp16`、`s:fp16` 和 `qs:int8[32]`。

这里的 `forbid` 要求外部输入合同保证数值有限且在 INT8 范围内；Compiler 不会证明数值范围，
也不会自动把越界数压回边界。

基本含义看[操作积木](../wiki/primitives.md)，完整组合看
[Q8 producer 计划](../../corpus/schedules/packed-q8_1-producer-gfx1151.json)。
“能生成这些指令”还不是“GPU 已经算对了所有字节”。

## 真正运行之前

2026-09-07 的后继交付是静态与 CPU 合同迁移。**当前源码还缺新的实机 host capture 和
通过 admission 的 gfx1151 Executor。** 保留的 v1/v2/v3 描述文件属于历史源码，
不能把它们直接套到新源码上。

收集主机事实统一使用
[`capture_executor_host.py`](../../tools/capture_executor_host.py) 的 `--runtime-kind hip`。
它要求明确给出 Python 包、ROCm 与编译工具、`amd-smi`、运行库以及需要的 profiler 路径，
核对实际用到的环境，然后写入一个新的外部 JSON 文件。`--help` 只显示参数，不收集环境。
环境收集本身不证明 GPU 正确性或速度。

适用的验收门槛通过且实机动作得到授权后，把这份新文件交给
[`release_gfx1151_executor_cycle.py`](../../tools/release_gfx1151_executor_cycle.py)：

```bash
PYTHONPATH=src "$AMD_PYTHON" tools/release_gfx1151_executor_cycle.py \
  --project-root . --host-environment "$AMD_HOST_CAPTURE"
```

`AMD_PYTHON` 是刚才记录的 ROCm Python 路径，`AMD_HOST_CAPTURE` 是外部收集文件。
工具从所有保留的发布描述文件推导下一个身份，先生成临时候选，再核对源码、真实主机和精确 Target，
最后以禁止覆盖的方式安装。缺少收集文件或 admission 失败时，已有发布文件保持原样。
不能手工挑版本号；这个发布流程也不运行正确性或计时 workload。

随后才是[SwiGLU 正确性](../../examples/gpu/swiglu_amd_quickstart.py)、
[RMSNorm 正确性](../../examples/gpu/rmsnorm_amd_quickstart.py)和
[Q8 字节检查](../../examples/gpu/llama_q8_1_amd_quickstart.py)。
各入口的准备分支只用 CPU。实机尝试还要求发布过的 Compiler、干净的源码、匹配的 Executor，
以及仓库外的新 evidence 目录；失败记录也必须保留。

## 搜索怎样避免把噪声当成提速

单行方案固定 `row_tile=1`，只比较 `num_warps={1,2,4,8}`；
基线固定 `row_tile=64, num_warps=4`。正式 Search Contract 未冻结前，不能宣布优化结果已验收。
已有检查路径保留以下规则：

- 两个 Workload case 都要算对，输入地址和内容不变，fallback 次数为零。
- 先用同一份基线和自己比较，即 B/B ABBA。它若产生明显假胜出或波动过大，就停止性能判断。
- 候选确认保留原始配对样本，使用约定的 0.05 CV 和 1.05 倍门槛，允许别人重新计算决定。
- 只有封存的叶子计时胜出才进入独立 rocprofv3 目录。Profiler 重放原决定并核对相同产物和 kernel
  启动，不把 profiler 里的时长当成普通计时样本。

汇编里读到的 VGPR、SGPR、LDS 等只是编译产物事实。它们不自动给出真实 occupancy，
所以保留 `occupancy_derived=false`。计时胜出还需要 profiler 证据，才继续考虑匹配的外部基线比较。

## 怎样保留旧结果

已验收的 refresh 提交 `b2d0a42409afcdef05e66da4723df44a6a09dfef` 以
`ee233f48824a7d1955e449db9636da974d93253f` 为基线，仍由
`codex/amd-gfx1151-refresh-20260907` 保留。更早的 AMD 来源
`d32b92f886e21adb51aa5e6648a0a3fbc4562565` 保存在备份分支
`codex/amd-consolidated-before-sync-20260907`。
历史 Executor 必须和原始完整 Git 源码一起回放。
[v1 外部绑定](../../inventory/AMD_GFX1151_EXECUTOR_V1_BINDINGS_20260826.json)、
[v2 发布](../../inventory/AMD_GFX1151_EXECUTOR_V2_RELEASE_20260826.json)、
[v3 发布](../../inventory/AMD_GFX1151_EXECUTOR_V3_RELEASE_20260826.json)保持原字节、日期和证据含义。
这些记录不会自动证明本轮新源码正确。

共同 runtime 的 B200 后继只更新源码绑定，并明确保留基线 v51 的主机声明。
这一步没有重新验证实机；真正执行前仍须通过主机 admission。
另一个 B300 描述文件虽然保留了更高的历史编号，也不能因此改变本轮采用的主机声明。

后续同步固定使用 main `6d9a198e69098cd51d427b9bffbe967a6f62de90`。
main 和已验收 refresh 曾分别发布不同的 B200 v53 描述文件。main 保留原路径，
AMD 原字节以
[`open-cake-ir-b200-v53-amd-refresh.json`](../../runtime/executors/open-cake-ir-b200-v53-amd-refresh.json)
保留身份。回放该变体要使用完整 `b2d0a424` 源码和原始
`runtime/executors/open-cake-ir-b200-v53.json` 路径；别名不代表 rebased runtime。
[Compiler v57](../../compiler/releases/v57/README.md)的原批准同样保留，
变化后的组合源码还需要协调后继发布并重新独立审查。

另一个 B300 分支的 `bfc446e384b87634daf5a289a35bc1e36bbad9ff` 已经使用 Compiler v56 和
B200 Executor v52。这里仅保留[身份材料](../../compiler/releases/v56/README.md)，没有迁入其实现。
旧 AMD 分支发生过冲突的 Compiler v28 archive 仍由原 Git 提交保存，也不覆盖 main 的 archive。
后继身份与独立批准分别遵循
[ADR 0049](adr/0049-released-executor-descriptors-reserve-their-identities.md)、
[ADR 0050](adr/0050-released-compiler-locks-reserve-their-identities.md)、
[ADR 0052](adr/0052-independent-agent-release-review.md)。
