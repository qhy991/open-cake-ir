# MetaX C550：当前任务能否运行

本页是接入前的固定源码调查。后续实现及实际验收范围见
[C550 当前可用路径](metax-c550.md)，不要把这里的初始拒绝计数当作当前状态。

2026-09-20。检查对象为 `7e20cf2d` 的任务、Compiler 和 Executor，以及
`c550-1` 上的 C550 / MACA 环境。该提交承接 `main@f09a3e85`，仅增加上一轮
MetaX 调查工具和文档。此轮从它创建本地平台分支 **`metax`**；原调查分支
`codex/metax-c550-support-20260920` 保留，主 checkout 没有切换。

**当前没有任务能沿 Open-Cake 的正式执行路径直接在 C550 上运行。**
拒绝发生在设备注册、任务构造和 Target / Executor 解析处，尚未进入 GPU
编译或 launch。这个结论是当前软件路径的可运行性检查，不是 C550 算力不足，
也不表示这些任务的数学运算不能移植到 C550。

## 检查范围和结果

检查使用干净提交的 detached worktree：
`/private/tmp/open-cake-metax-survey-verify-20260920`。
执行解释器为仓库已存在的 `.venv/bin/python`（Python 3.14.6）。
第一次未指定解释器的命令在该目录选中了系统 Python 3.9.6，于 import 时因
`dataclasses.field(kw_only=...)` 失败；那次没有产生任务结果，也没有安装或
修复环境。随后显式使用项目现有解释器执行检查。

| 检查边界 | 数量 | MetaX 结果 | 对照或范围说明 |
| --- | ---: | --- | --- |
| `tools/launch_task.py` 实际 CLI choices | 46 | 46 个任务工厂均拒绝 `triton-metax` | B300：46 个都生成 lowering；DCU：29 个生成 lowering，17 个因 Target 未声明 `cast` 被拒绝 |
| `tools/check_flashinfer_tasks.py` 的完整 pack | 26 | 0 个 MetaX lowering | B300：26 个都生成 lowering；其中 17 个已在上行，另外 9 个为 launch plan，不能把两行相加为 72 项 |
| AKA v3、DeepSeek-V4、独立 add-RMSNorm 工厂 | 9 | 9 个均拒绝 | AKA v3 6 个只接受 B200；V4 2 个只接受 B200；add-RMSNorm 要求已注册 backend |
| 其余注册 operator | 8 | 无 MetaX 执行路线，源码审查 | 2 个旧 tile 合同及 DSA、Flash KMeans、KDA、KDA megaop、QSA、TinyGEMM 六类专用路径；未当作统一 CLI launch 运行 |
| 已提交 workload 文档 | 57 | 没有 MetaX workload | 11 个显式 `sm_100a`，37 个显式 `sm_103a`，9 个采用专用/历史合同结构，没有统一 `semantics.target` 字段；不能把字段缺失解释成可在任意设备运行 |

共有 **72 个注册 operator**；上表分别说明统一入口、pack、其余 factory 和
专用路径的检查范围。46 + 26 去掉 17 项重叠后是 55 项，再加 9 个额外工厂
和 8 个源码审查的 operator，覆盖该注册集合。所有 GPU 测试均为 `not_run`。

这里的“生成 lowering”只表示当前 Compiler 的 assess + lower 接受了该
目标上的 Schedule，并生成目标源代码；没有运行 Triton 二进制编译，也没有
产生设备正确性、吞吐或延迟结果。B300 / DCU 的结果不能转算为 MetaX 通过。

## 当前阻塞点

1. `tasks/devices.py` 的 7 个 backend 分别对应 Apple、NVIDIA、Hygon 和 AMD，
   没有 MetaX。CLI 的 `--backend` choices 来自该表。直接调用任务工厂也拒绝，
   不是只修改 CLI 参数就能绕过的问题。
2. `compiler/targets/` 没有 C550 的精确 Target。查询 `metax_c550` 或
   `xcore1002` 都得到 `no Target document declares ...`。这两个名字只是探测
   请求，不构成新的 Target 声明。
3. `CodeObject` / Triton compile route / Evaluation 平台目前处理 cubin、
   HSACO 和 Metal binary archive，没有 MetaX 的产物与运行时。
4. `resolve_executor(..., target='metax_c550')` 返回
   `no host capture is published for exact target 'metax_c550'`。
   尚无 MetaX host kind、二进制 launcher、timer、profiler 及对应任务设备分配。
5. `c550-1` 当前系统 Python 没有 torch / triton / numpy；containerd 只列出
   `k8s.io` namespace，尚无上一轮选定的 MetaX 开发镜像。再次观察到 8 张
   C550、KMD 3.6.11、MACA 3.5.3.18，镜像存储所在文件系统约 853 GiB 可用。
   本轮没有下载镜像、改动集群组件或启动 GPU kernel。

可获取的 MACA 3.5.3.x / PyTorch 2.8.0 / FlagTree 开发镜像及验证记录见
[接入调查](metax-c550-bringup.md#可获取的开发镜像)。环境准备能解决第 5 项，
不能替代前四项的软件实现与验收。

## 哪些任务适合先移植

下面是根据现有 Schedule / Workload 源码得到的实施优先级，**不是已在
C550 验证通过的名单**。

| 分组 | 任务 | 已有基础与剩余工作 |
| --- | --- | --- |
| 第一批：27 项 FP32 任务 | 29 项通用矩阵中除 `gelu_tanh`、`gelu_tanh_backward` 外的任务 | Schedule 仅使用 load / elementwise / reduce / store；已有跨 NVIDIA / Hygon 的 authoring。优先用 RMSNorm、Softmax、SiLU、bias reduction 验证 MetaX 的 64-lane reduction、数学函数和完整输出 |
| 单独验证 tanh：2 项 | `gelu_tanh`、`gelu_tanh_backward` | NVIDIA source 声明 `libdevice.tanh.f32`，DCU 使用自己的 OCML contract；MetaX 需要测量自己的实现并声明 contract，不能直接继承任意一方 |
| 第二批：9 项 BF16 norm | FlashInfer RMSNorm 6 项、fused add-RMSNorm 3 项 | 有独立 oracle、显式 FP32 accumulation 和 cast；需要 MetaX BF16 读写及舍入证据。1536 / 7168 宽度的现有 starter 已用多个静态 span 表达，并非因为非二次幂就没有路径 |
| 第二批：8 项 FP16 GEMM | FlashInfer pack 004–011 | 当前 starter 使用 FP16 load → FP32 multiply/reduce → FP16 cast，未依赖 MMA primitive；先验证语义可运行，再单独研究矩阵单元与性能 |
| 后续：9 项 launch plan | 8 项 GQA / MLA，加 1 项 MoE | 合同及验证器当前精确绑定 B300 `sm_103a`；需要新的 MetaX 执行绑定和相应 verifier/plan 支持，保留原始合同。还要验证多阶段 scratch 生命周期、数据相关索引、FP8/路由语义和完整输出 |
| 另外 17 个注册 operator | AKA v3 6 项、V4 2 项、add-RMSNorm 1 项、tile 2 项及六类专用路径 | 部分显式锁定 B200/B300，部分没有统一任务入口；逐个迁移自己的 workload / oracle / launch owner，不能批量改 Target 字符串 |

27 项第一批任务的完整名称：

```text
rmsnorm layernorm residual_rmsnorm softmax
silu swiglu softsign selu softplus_gradient prelu
softmax_backward layernorm_backward_input rmsnorm_input_gradient
absmax_rescale cosine_similarity layernorm_gamma_beta_backward
bias_gradient_reduction per_channel_moments channel_absmax_scale
momentum_sgd adamw adadelta
gemm gemm_silu pairwise_sqdist attention_decode gemm_bias
```

`attention_decode` 是这里的 FP32 contraction fixture，不代表 pack 中的
GQA/MLA 或完整 serving attention。GEMM starter 可表达数学运算，也不代表
能发挥矩阵单元性能。

对 46 个统一入口的默认 shape（contraction 的 K=256）计算 tensor ABI 的
输入输出字节总量，最大为 `fib_gemm_n28672_k4096` 的 **234946560 bytes**。
这没有显示默认输入输出本身逼近 64 GiB 显存；它不是峰值显存测量，未计入
运行时、临时区、oracle、多 case 并存，也没有覆盖 pack 的所有 batch 或
launch plan 的中间张量。

## 下一条完整验证链

先准备独立 MACA 开发环境，读取镜像里固定版本的 `GPUTarget`、产物和 metadata；
补齐 MetaX Target、Triton compile route、ExecutionPlatform 和 Executor。
从 `rmsnorm` 的五个既有输入分布开始，沿相同 Workload 和完整输出 oracle
验证。静态 Corpus Gate 及适用 admission 通过后，再在分配到的 C550 上执行。
正确性通过后才接原生 timer / profiler，最后扩展到上述任务组。

当前分支只加入调查与本次可运行性结论，没有把未实现的 backend 注册为可用，
没有改 frozen workload、原始 Corpus expectations 或历史设备结果。

## 可复核记录

记录根目录（checkout 外）：
`/Users/haiyan-infiniai/Agent4Kernel/open-cake-ir-evidence/metax-task-readiness-20260920/`

* `task-readiness.json`：实际 CLI choices、46 项 factory / assess / lower 结果、
  26 项 pack 检查、57 份 workload 清单、Target / Executor 拒绝。
* `supplement.json`：72 个 operator 的剩余集合，以及递归读取
  `instruction.contract` 的结果。初版记录的 `contracts` 字段没有遍历该嵌套
  结构，不能作为“没有指令契约”的依据；以此补充记录为准。
* `additional-factories.json`：AKA v3、V4 和 add-RMSNorm 的 9 次实际 factory 拒绝。
* `audit-v2.py`：调用现有检查器和任务工厂的调查脚本，不注册 backend，不修改
  Target，不更改 workload acceptance，不运行 GPU。

保留的 `audit.py` 是第一次脚本版本，在读取不存在的 `Assessment.schedule`
属性时中止，没有形成完整结果；修正版读取 frontend 的原始 Schedule 并完成
上述检查。不存在将中止运行标记为通过的情况。
