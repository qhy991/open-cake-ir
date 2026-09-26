# MetaX C550：当前可用路径

MetaX 的精确 Target 是 [`xcore1002`](../compiler/targets/xcore1002.json)，任务入口为
`triton-metax`，维护分支为 `metax`。它复用现有 Python authoring、Triton emitter、
Workload oracle 和 common Evaluation；MACA 编译产物、加载器和 host admission
由各自的平台实现负责。

当前范围是 **FP32 / FP16 / BF16 / INT32 缓冲区、load / cast / elementwise / reduce / store、
完整输出正确性验证**，以及下述受限的 E4M3FN 存储和解码。转换复用现有 typed cast 规则：三种浮点格式之间，以及有向的
INT32→FP32；浮点转整数仍被拒绝。FP32 tanh 使用独立的 `maca.tanh.f32` 契约。
FP32 单次舍入乘加使用 `maca.fma.f32`：三个同形寄存器操作数在 MetaX Triton
路径发射 `tl.fma`，不借用 NVIDIA 的 PTX inline assembly。
矩阵路径复用现有 `mma` 与 `triton.dot.fp16_fp32`、`triton.dot.bf16_fp32`、
`triton.dot.fp32_ieee` 三条契约；FP16、FP32 已有下述正式任务结果，BF16 尚限于
单 tile 原生诊断。TF32 和 FP8 矩阵尚未准入；FP8 的范围限于下述存储和解码。原生 MCPTI 成对计时和独立 profiler 已接入；
测量质量不通过时明确返回 `measurement_quality_failed`，不作为有效性能结果。
历史的无计时策略仍可回放。没有 CUDA/HIP fallback，没有借用其他设备的校准或性能结论。

## 三种身份分别检查

物理设备、工具链 family 与兼容 API 分别是 `xcore1002`、`xcore1000` 与
`GPUTarget("maca", 80, 64)`。硬件常量及来源只在 Target 文档声明。
`80` 是 MACA Triton API 的兼容值，不是 NVIDIA SM80；设备 admission 还会从
原生 MACA 属性查询实际物理架构。

已验证环境使用 MACA PyTorch `2.8.0+metax3.5.3.9`、FlagTree
`0.5.1+metax3.1`，其提供的 Triton **API 版本为 3.1.0**。系统 Python 不等于
该环境：实际解释器是 `/opt/conda/bin/python3`。
[`runtime/hosts/xcore1002.json`](../runtime/hosts/xcore1002.json) 由 canonical capture
命令生成并绑定解释器、包、构建工具、MACA runtime、MCPTI 库及其 API 版本。
每个 worker 都重新 admission；实际已映射的库必须是捕获的绝对路径。

## 编译与执行

编译通过现有 bubblewrap 路径运行，不挂载 GPU，也不暴露作者工作目录。
当前 MACA backend 实际提供 TTIR、TTGIR 和 mcfatbin；没有 LLVM/PTX artifact
时不会生成占位文件。mcfatbin 是 Clang offload bundle，包含原生设备 ELF 和
bitcode。检查器验证声明的 family、成员边界和 MXC ELF 类型，loader 仅提交
原生 ELF，避免模块加载回退到 bitcode 编译。

当前 vendor launcher 只传非 constexpr 参数。封存 manifest 读取实际 TTGIR
signature 并验证 tensor 参数数量，隐藏指针数为 0。

### FP32 FMA 指令 lowering

源码提交 `28982965` 的 `[8,128]` Cake Schedule 通过 assessment 和 kernel-only
投影；C550-1 已捕获的 Triton 3.1 与 C550-2 Triton 3.6 均把投影源码离线编译成
`mcfatbin`，TTIR/TTGIR 各保留一个 `math.fma`。C550-2 的单卡 broker job
`maca-f697166c27ba` 对全部 1024 个输出逐 bit 比较独立精确有理数 RNE 参考：
128 个区分融合与分步舍入的点、892 个有限常规点和 4 个次正规边界点均为
0 mismatch，输入 bits 未改变。先行的独立 `tl.fma` 探针 job
`maca-feef2fc7be92` 也在 644 点上逐 bit 通过。原始输入/输出、参考脚本、
编译产物与收据位于 checkout 外的
`open-cake-ir-evidence/metax-fma-probe-20260926/`。

设备作业使用 C550-2 的 Triton 3.6 对 Cake 生成源码 JIT；并未加载上述离线
`mcfatbin`。C550-1 Triton 3.1 只有离线编译证据，没有对应设备数值结果。
这些点支持本次有界 FP32 指令准入，不覆盖所有异常值、注册 Workload 的
Evaluation、延迟或性能收益；也没有物理独占声明。

### 固定循环的 MetaX 专属 full-unroll lowering

在源码提交 `667c8c93`，`loop_unroll_factor > 1` 只有在循环边界来自静态
Buffer 维度、`num_stages=1`、且因子恰好等于完整迭代次数时才生成 MACA
`tl.static_range`。部分展开、查询决定的边界和带流水阶段的展开由
`MACA_LOOP_UNROLL_UNSUPPORTED` 拒绝；NVIDIA 等目标仍发射原有 `tl.range`
选项。该规则改变生成源码，并没有引入第二套 Triton emitter。

以 BF16 GEMM-bias `512×256×256`、K tile 64、4 次 K 累积为固定切片：
已捕获的 C550-1 FlagTree/Triton 3.1 环境离线编译得到 TTIR、TTGIR 和
mcfatbin；两层 IR 各有 4 个 dot、没有 `scf.for`，产物位于
`c550-1:/root/.local/share/open-cake-ir/metax-c550-20260920/full-unroll-667c8c93/`。
C550-2 现有 Triton 3.6
容器也完成相同源码的离线编译。在 C550-2 上，新建短时容器只暴露 1 张卡，
使用与宿主同 inode 的 `maca` 锁执行生成源码的 JIT 路径；随机与交替符号
两组各 131072 个输出元素，对独立 CPU FP32 dot+bias 参考在
`atol=rtol=0.001` 下均为 0 mismatch，最大绝对误差分别为
`3.0517578125e-05` 和 `1.7881393432617188e-07`，输入未改变。
原始源码、编译产物和设备诊断收据保存在 checkout 外的
`open-cake-ir-evidence/metax-full-unroll-20260926/full-unroll-667c8c93/`，
对应 broker job 为 `maca-8970a53c0f29`。设备诊断由生成源码重新 JIT，
没有加载前述离线 mcfatbin；C550-2 运行时也不是当前已捕获的 C550-1
3.1 Host。它不构成正式 Workload Evaluation、精度普适证明或性能资格，
没有计时样本，也不声明整机 GPU 独占。

Evaluation 验证全部 Workload input cases、每个输出元素及输入不变性，并保留
PCI 标识和 runtime 路径。无计时收据使用 JSON `null` 的 `timing_samples`，
不会给出零延迟。CUDA/NCU 的 profile child 仍保留自己的单次校验与独立 profiler 路径。

MACA 通过已有 local broker 的 `maca` kind 执行。容器必须与宿主共享同一个
`/tmp/open-cake-ir-maca-<uid>.lock` inode，并只暴露一个选定设备；不能在每个
容器里各自建一个私有锁。这个分配模式只声明本用户任务串行，外部活动没有被排除，
计时收据保留 `local_serialized` 和 `external_gpu_activity=not_excluded`，不声称物理设备独占。

## 现有 C550 开发环境

调查选用的公开镜像地址及下载来源在[初次调查](metax-c550-bringup.md)中保留。
当前节点 `c550-1` 的 `open-cake-metax` namespace 使用准备后的本地镜像
`open-cake-metax-dev:20260920`：保留原 SDK，增加已捕获的 bubblewrap。
首轮冻结源码在容器内 `/work/source`；数值扩展分别保留为
`/work/source-f1cadbdd` 和 `/work/source-5aef189b`。这些目录保持各自已验证的提交，
构建与证据目录位于 checkout 外；日常开发从维护分支 `metax` 创建独立 checkout。

原编译容器发生嵌套 `/proc` mount 拒绝，失败记录保持原样。经 owner 明确授权，
后继 CPU 容器 `open-cake-metax-compile-v2` 在原配置上增加
`systempaths=unconfined`，保持无 GPU、无网络，随后完成隔离构建。
这不是在构建器中增加无隔离 fallback。相同镜像不保证任意宿主拥有相同隔离权限；
新的环境必须在自己的 capture 和 admission 下验证。

在这个已准备环境中，封存一个新的 RMSNorm baseline：

```sh
nerdctl --namespace open-cake-metax exec \
  -w /work/source-5aef189b --env PYTHONPATH=/work/source-5aef189b/src \
  open-cake-metax-compile-v2 /opt/conda/bin/python3 tools/launch_task.py \
  --task rmsnorm --backend triton-metax --rows 128 --columns 1024 \
  --harness codex --model baseline-only --effort low \
  --workspace /work/rmsnorm-new-baseline --baseline-only
```

workspace 必须不存在。`--baseline-only` 构建、封存后退出，不调用 provider，
也不执行 GPU kernel。构建后的固定 bundle 由 `load_prepared_baseline` 读取，再交给
`CommandBrokerSubmitter` / `BoundedBrokerEvaluator`。当前本地 Triton TaskLab 使用
`tasks.evaluate --local-kind maca`：先计算 CPU 输入和原 oracle，再通过
`evaluation.local_broker.admit_local_job` 获取真实 allocation 并执行。
上例所用历史源码仍保留原先 broker exec worker 的入口。
原验证 driver 及它调用的确切输入保存在下述外部证据目录。

## 验收范围与后续

设备验证在源码提交 `b7179ede6e7f1591d0aa856cb87b08712922df11`、上述固定软件环境完成。
27 项 FP32 任务在列出的固定形状上通过了全部既有输入分布：RMSNorm 为
`128×1024`，其他一般任务为 `8×128`，contraction / GEMM-bias 为 `R=8,C=8,K=32`。
这覆盖 135 个 case，原 oracle 和容差未修改；它不外推到所有形状、性能、模型或 serving。
原始报告及 per-attempt 收据保留在 checkout 外：

* `c550-1:/root/.local/share/open-cake-ir/metax-c550-20260920/fp32-qualification-summary.json`
* 同目录 `gpu-rmsnorm-b7179ede-v1/` 与 `gpu-bounded-fp32-b7179ede-v1/`
* 本地镜像：`open-cake-ir-evidence/metax-implementation-20260920/c550-fp32-qualification-b7179ede.tar.gz`

静态验证与设备结果分开：该提交通过完整合同测试（2115 tests、7768 subtests，
19 个环境 skip）和现有 Corpus Gate（164/164）。现有 corpus 没有 C550 case，
其报告明确列为 unexamined；不能把 164/164 当作 C550 的设备或 corpus 覆盖。
本轮没有 kernel 优化机制、性能测量或经验 pass promotion。

后续数值能力验证在 `open-cake-ir-evidence/metax-numerics-20260920/` 保留独立记录：
`f1cadbdd` 的七条转换路径共 717114 个输入与标准库 RNE 参考逐 bit 一致；tanh 的
82097 个有限样本相对 `math.tanh`→FP32 参考最大 2 ULP，保留正负零。源输入 bit pattern
先在设备内核调用前核对，输出用 NaN 预填充以拒绝漏写。这是独立 primitive diagnostic，
不等同于 Workload 收据；实际任务通过既有 broker / worker / oracle 另行验证。
这些观测不构成全域 tanh 误差上界，也不把原始记录重标为后来的源码提交。

任务级补验保留在同一外部证据目录的 `final-device-results/`：`f1cadbdd` 的 20 项
混合精度／整数任务通过 100 个 case；`5aef189b` 的 GELU forward／backward 通过
10 个 case。FP16 GEMM 保留原始固定 N/K、最小声明 M；BF16 FlashInfer normalization
保留原始 hidden size、最小声明 batch；其余新任务为 `8×128`。所有输出按各自原有
oracle 和容差逐元素比较，输入不变，零 fallback、零 timing sample。最大的 FP16
GEMM 最大绝对误差为 `0.5`，在原容差内通过，不能描述为 bit exact。

加上前一阶段 29 项／145 cases（包含单独补验的两项 AKA FP32），统一 CLI 的 51 项
任务累计有 255 个 case 的 C550 结果。这是上述固定形状的验收，不是全部 batch/shape
或完整模型的资格。各批原始源码身份分别保留；逐项源码兼容性检查不替代新设备运行。

## 原生计时与归因

MCPTI API 18 的并发 kernel record 提供原生 start/end 时间戳；样本是单次目标
dispatch 的 end−start，不包含 host launch 或前置 reset。每个样本前，同一 stream
写入一个四倍于 Target 所声明 L2 的 FP32 缓冲区（C550 为 32 MiB）。这是明确的
reset 操作，不是“所有缓存已清空”的证明。收据保留独立 reset 校准、每次原生活动、
API/kernel correlation 和 sealed manifest；丢失／额外 dispatch、跨 cohort 换校准、
错误设备或样本与原始时间戳不一致都会被拒绝。

正式成对流程仍检查全部 Workload cases，并验证每次 warmup／timed 调用的新输出。
独立 profile 另做预检和 instrumented 输出检查，反馈 dispatch、寄存器、共享内存及
函数 local-memory 需求。`function_local_bytes_per_thread` 来自已绑定的函数属性；
`mcpti_reported_local_bytes_per_thread` 仅保留 MCPTI 原值。后者曾在函数报告 492 bytes/thread
时报告 0，故 `local_memory_reservation` 明确列为 `not_qualified`，不推断 padding、
实际零用量或两个 API 量相等。occupancy、bandwidth 和 instruction counters 未采集。

在冻结执行源码 `66dcc472`，GEMM-bias `M=8,N=8,K=4096` 的显式四执行组版本
通过全部五 case、完整 fresh-output 检查及 500 个计时样本；20 个 cohort 的 CV 为
0.006988–0.015165，低于原 0.05 门槛。相同 artifact 的两 arm 中位数均为 26.368 us，
判定 `close_null`；这是该形状上的测量控制，不是优化收益。该 artifact 编译于
`becb3bb1`，只把原 Schedule 的 `execution_groups=[0]` 改为 `[0,1,2,3]`。
默认一组的 K4096 artifact 曾发生 native launch 拒绝，不能宣称它已被这个结果验证。

RMSNorm `128×1024`、LayerNorm `8×128`、GEMM-bias `8×8×32` 的正式正确性和
独立 profile 均通过，但它们在原 0.05 CV 门槛下的成对测量均失败。记录全部保留，
没有重试取最好值或放宽门槛；以上较长 kernel 的通过也不外推到这些短 kernel。
这批收据与失败诊断位于 `c550-1:/root/.local/share/open-cake-ir/metax-c550-20260920/`，
本地镜像及审查在 `open-cake-ir-evidence/metax-parity-20260920/RESULTS.md`。

BF16 正式矩阵任务、TF32、FP8、更多形状及完整 provider 优化闭环仍需后续验收。
FlashInfer GQA/MLA/MoE 和其他精确绑定 B200/B300 的合同仍需各自的后继验证，
不能只改 target 字符串后宣称可用。

## 矩阵路径的正式任务证据

以下结果执行于 `8c0cad53`，沿用各自原 Workload 的五个输入分布、完整输出 oracle、
输入不变性和容差。统一 Compiler 生成 Triton 源码，经 CPU 隔离编译封存为 mcfatbin，
再由原 MACA broker、worker 和 receipt reader 执行。每项成对检查均有 740 次候选路径
调用、500 个样本和零 fallback；全部 20 个 cohort 通过原 CV 0.05 门槛。
三个比较都是同一产物的自比较，结论均为 `close_null`，不构成优化收益。

| 原任务 / 精度 | M / N / K | 覆盖 | 最大绝对误差（全部 cases） | cohort CV 范围 | 成对 job |
| --- | --- | --- | ---: | --- | --- |
| `fib_gemm_n128_k2048` / FP16 输入与输出、FP32 累积 | 17 / 128 / 2048 | 32 次 K 累积、M 尾部 | 0.03125 | 0.006690–0.010692 | `maca-59e41335edd4` |
| 同一任务 / 同一精度 | 64 / 128 / 2048 | 32 次 K 累积、完整 M tile | 0.25 | 0.005623–0.010624 | `maca-c0e219b81422` |
| `aka_gemm_nt_bias` / FP32 IEEE | 72 / 128 / 128 | 两次 K 累积、一次 bias、M 尾部 | 0.000019073486328125 | 0.011054–0.018379 | `maca-38b2c5bbb9d2` |

原 FP16 任务使用 `atol=rtol=0.01`，AKA 任务使用 `atol=rtol=0.00002`。
表中是容差内通过，不能描述为逐 bit 一致。所有候选显式使用 `64×64×64` tile 和
四个执行组，未扩展 NVIDIA 专用的 Compiler width pass。它们增加的是既有任务的
矩阵路径与形状证据，不是新增三项注册任务。

三个独立 attribution job 分别为 `maca-02169aa5f6fd`、`maca-eacd79b7ee49`、
`maca-d14c8ca7fec4`。每个都检查预检和 instrumented dispatch 的实际 primary 输出。
FP16 两个形状均报告 224 registers/thread、16384 bytes 动态共享内存和 0 function-local
bytes/thread；FP32 为 256、32768 和 396。MCPTI local-memory reservation 仍未合格，
profile 时间仍只用于归因；occupancy、带宽和指令计数尚未采集。

产物编译、prepared 上下文与执行身份分别保留：M17 编译于 `5ed6420f`、准备于
`2c06833e`；AKA 编译于 `2c06833e`、准备于 `8c0cad53`；M64 两者均为 `8c0cad53`。
后继 worker 在申请原 broker allocation 前计算各 case 的 CPU 输入和 oracle，
并在同一进程复用；锁仍保持到 worker 退出，包括失败路径，不在构造失败时提前释放。
原始 bundle、控制器、成对与归因收据均在外部证据根
`open-cake-ir-evidence/metax-parity-20260920/matrix-*/`，远端同名目录位于
`c550-1:/root/.local/share/open-cake-ir/metax-c550-20260920/`。

BF16 的范围仍是 `64×64×64`、五个分布的原生独立诊断，尚未完成注册 Workload 的
整条路径。原 TF32 RNE 假设有两个失败分布；事后 RTZ 分析不替换失败，也不授权准入。
FP8 直接 dot 保留 LLVM lowering 失败。AKA 的 `N=80,K=130` 请求被现有任务工厂的
row-span 限制拒绝，本表不声称 N/K 尾部或任意矩阵形状已验收。

2026-09-26 在 C550-2 的 Triton 3.6.0 容器中，无 GPU 的 `64×64×64` E4M3FN
直接 `tl.dot` 探针仍未产生 native artifact：`ConvertTritonGPUToLLVM` 走到
`GenericFMAVectorMultiplier::multiplyVectors` 时触发
`aElem.getType() == tgtTy` 断言。重试只为保留完整编译 stderr，源码、请求、
失败结果和原始日志位于 checkout 外的
`open-cake-ir-evidence/metax-fp8-dot-3p6-probe-20260926/`。这说明该固定源码在
这套工具链上编译失败，不是 C550 物理 FP8 能力、设备数值或性能的结论；
`xcore1002` 仍不声明 FP8 dot 合同。其他软件栈的显式解码矩阵路径属于不同机制，
需要自己的精度合同与完整输出验证，不能拿来替代直接 FP8 dot 的失败记录。

MACA 预检只准入固定、单阶段且完整展开的非默认 `loop_unroll_factor`，其余展开
请求会明确拒绝。非默认 `flatten`、`disallow_acc_multi_buffer`、`disable_licm`
仍被拒绝；`num_stages` 保持可表达。
partial-K 仍由 `TRITON_MMA_K_RANGES_UNSUPPORTED` 拒绝，不会落入 NVIDIA inline assembly。
新增三个正例和两个反例使完整 Corpus 成为 169 项，其中五项检查 xcore1002；
静态 Corpus、上述设备证据和完整后端能力仍分别报告。

## M17 的一次显式 M tile 优化

在已验收的 FIB `M17/N128/K2048` 固定 baseline 上，仅将 M tile 从 64 改为 32；
N/K tile 仍为 64，四个执行组和 32 次 K 累积不变。完整解析后的 Schedule 比较
只改变 M tile、MMA M extent 和三个派生寄存器 buffer 的 M extent，原 Workload、
输入 ABI、oracle 和测量策略保持。候选编译、执行于 `8c0cad53`，baseline 仍是原
`5ed6420f` 产物。

一次 search（`maca-4785c930f50c`）通过后，按事先声明的流程进行了独立 confirmatory
（`maca-f166d55e4231`）。两次均通过全部五 case、所有 fresh-output 检查、原 CV 0.05
和 materiality 1.05 门槛，候选均赢得 10/10 对；每次保留 500 样本、零 fallback。
确认阶段 baseline/candidate 中位数为 **72.448 / 58.368 us，1.241228×**，
20 个 cohort 的 CV 为 0.007220–0.012788，最大绝对误差为 0.03125，输入未改变。

独立 profile `maca-51d9d428f693` 的实际输出同样通过，报告 170 registers/thread 和
12288 bytes 动态共享内存，原 baseline 为 224 和 16384。函数 local bytes/thread
均为 0；这没有补齐 MCPTI local reservation、occupancy、带宽或指令计数的资格。
减少 tile extent、资源变化和速度改善是同一次受控改动的观测；没有计数器证据可以
把全部收益进一步归因到某种硬件瓶颈。

收益限于这一个固定形状、固定 baseline 和 `local_serialized` 测量范围。
这是显式 authoring 对照，尚不是完整 provider/Ralph 优化 Campaign；promotion disposition
为 **No promotion**。原始结果在上述外部证据根的
`matrix-fib-m17-tile32-{search,confirmatory,attribution}-8c0cad53-v1/`。

## FP8 原生存储与数值诊断

MACA native loader 接受封存 ABI 中的 `fp8_e4m3`，要求真实的
`torch.float8_e4m3fn` 和既有 DType 声明的一字节存储宽度。uint8、FP16、E4M3FNUZ、
E5M2 或错误宽度均在 dispatch 前拒绝。Compiler 允许 E4M3FN 的 load/store，以及非标量 tile 向 FP16、FP32 的显式 cast。
该解码路径供后继 scaled-reduction/MoE 使用；其他算子需求仍须各自验收。

`MACA_FP8_CAST_UNQUALIFIED` 拒绝逆向编码和 FP8→BF16 等未验收转换；
`MACA_FP8_SCALAR_CAST_UNSUPPORTED` 在外部编译前拒绝当前 SDK 会断言的
单值转换。直接 FP8 算术由 `MACA_FP8_OPERATION_UNQUALIFIED` 拒绝，
直接 FP8 dot 契约仍未在 Target 声明。唯一新增的 FP8 矩阵路线是下述
`maca.simt.fp8e4m3_compensated_fp32`，先解码再做 FP32 SIMT 补偿累加；
存储准入本身不改变算术边界。

冻结执行源码 `6d997c3d` 的独立诊断直接传输原始 bytes，再 view 为真实 FN tensor，
launch 前后先 view uint8 再复制到 CPU，避免数值转换改变 NaN 编码或负零。
它们使用真实 MACA broker job，但不是注册 Workload 的 EvaluationReceipt，也没有计时。

- `maca-bd7ef50c9f98`：全部 256 个原始 FP8 编码在两个输出 poison（0、126）下
  直接复制，两次调用的 512 个输出 byte 均相等，包含正负零和两个 NaN 编码，输入
  bytes 不变。这是纯 load/store，无转换或算术，2 registers/thread、shared/local 为 0。
- `maca-584187da1812`：FP8→FP16 和 FP8→FP32 各检查全部 254 个有限编码的输出 word，
  包含正负零，均逐 bit 相等；两个 NaN 编码的分类也分别通过。
- `maca-ff7880c5bf02`：FP32→FP8 在 `[-448,448]` 内的 1014 个特定有限输入模式和
  10 个 padding 上验证 RNE。四批各用两个不同输出 poison，共八次 native call、
  2048 个输出 byte 全部精确匹配，输入 bits 未改变。NaN、Inf、overflow/saturation
  尚未取得编码资格，不能从这些点外推一般转换行为。
- 显式 FP8→FP16 后的 K64 dot 在原严格诊断中仍有一个超差输出；K16 分组后继修正了
  那个点，但在 mixed_magnitude `[43,28]` 出现新的超差。因此两次完整诊断均保留失败，
  不按 case 选择两个实现中较好的结果。原参考和 `.001/.0001` 容差没有改动。
- `maca-59a4486282a2` 使用不同的 SIMT 算术：FP8 解码到 FP32、FP32 products 和
  Neumaier 补偿累加。原五个 case 的 20480 个输出 word 全部与参考相等，输入 bytes
  未改变；五次 native call，21 registers/thread，shared/local 为 0。
  这是精度 control，不是 FP16 dot、native FP8 MMA 或性能替代，也不改判前两次失败。

后继 Cake 以 `2×64×64` staged tile 明示这条 SIMT 补偿机制，Target 仅准入固定的
`64×64` 两个 E4M3FN 输入、FP32 输出、两行 program map、四执行组和无跨循环累积。
源码 `a0f16157` 的 kernel-only 投影在 C550-1 Triton 3.1 与 C550-2 Triton 3.6
都生成 `mcfatbin`，TTIR/TTGIR 均无 `tt.dot`；后继 `878afb61` 收窄准入并把
MetaX 发射体移入平台模块，投影 kernel 的字节与 `a0f16157` 相同。
C550-2 单卡 broker job `maca-8b89488ad3be` 对旧五组冻结输入的全部 20480 个输出
逐 bit 匹配原参考，输入字节未改变。源码、两版编译产物、完整输入/输出和收据保存在
checkout 外的 `open-cake-ir-evidence/metax-fp8-resident-simt-m2-20260926/`。
设备作业 JIT 了 Cake 生成源码，并未加载离线 bundle；Triton 3.1 只有离线编译，
没有该实现的设备数值结果。这是有界软件 lowering 的正确性诊断，尚不是注册
Workload Evaluation、原生 FP8 矩阵指令或性能收益。

### 一行与两行的诊断性 MCPTI 对照

在 C550-2 已捕获的 Triton 3.6 Host、同一冻结 `64×64` 输入下，broker job
`maca-55df39af5db2` 先让一行、两行和 Cake 两行三份源码各通过五组完整输出的
逐 bit 检查，再按同 stream 4×L2 reset 采集一行/两行各 250 个 MCPTI dispatch
样本。10 个反向顺序 pair 中一行赢 10/10；pooled median 为一行 **31.232 µs**、
两行 **42.496 µs**。四个前后 A/A pair 的两 arm pooled median 均为
**31.488 µs**，cohort CV 门槛通过。Cake 两行源码另有一个 25 样本 cohort，
median **39.168 µs**，不能拿它与前两个 pooled median 作配对速度比。
29 份原生活动已由仓库 `dispatch_samples` 和纯配对派生独立重放；原始输出与
活动在 checkout 外的 `open-cake-ir-evidence/metax-fp8-paired-screen-20260926-v4/attempt4/`。
先前 v3 在第一个主 cohort 的第 20 样本因 256 ns 的 MCPTI 记录重叠被拒绝，
失败记录保留，未用后继通过结果改判旧尝试。

这项 `local_serialized` 对照不排除外部 GPU 活动，也没有注册 Workload、封存
候选和正式 EvaluationReceipt；它只能指向下一条优化调查，**不构成已合格的
1 行加速结论**。Cake 当前一行程序的最早结构性拒绝与所需 IR/Verifier 审查见
[F-2026-09-26-001](../findings/2026-09-26-001-metax-fp8-one-row-capacity.json)。

调查还发现当前 SDK 对标量 FP8→FP32 的最小程序触发 `RankedTensorType` 内部断言。
补偿 control 先按 tensor 转换，再在 FP32 上选择元素，才通过编译；这没有修复或
取得标量 FP8 转换的资格。全部原始源码、编译失败、封存参考、NPZ 观察与结果保留于
上述外部证据根的 `fp8-*` 目录。后继算子需按自己的原 Workload、范围和 oracle 验收；
这批诊断不构成 FP8 MoE、scaled matrix 或完整后端能力的资格。

三条新增 Corpus 程序还完成了 Compiler 生成路径的设备复验：`e21aeaad` 的
`xcore1002-fp8-copy`、`xcore1002-fp8-decode-fp16`、`xcore1002-fp8-decode-fp32`
经正式 assessment、lowering、kernel projection 和 CPU 隔离编译，封存为 `[1,256]` ABI；
参考仍是上面原封存的全部 256 个编码与解码 word，仅张量形状增加一行维度。
执行源码 `7e86e8ad` 的 `maca-b1424247b525` 完成四次 native call：复制两次共
512 byte 精确匹配，两个 decoder 各 254 个有限 word 精确匹配、两个 NaN 分类正确；
全部输入 bytes 未改变，零 timing samples。复制报告 2 registers/thread，两个 decoder
各 10；三者 shared/local 均为 0。原始观察在外部证据根的
`fp8-corpus-gpu-7e86e8ad-v1/`。这验证了生成路径的原语正确性，尚不构成
完整 MoE、scaled matrix、其他形状或性能资格。

## 路由操作与四阶段原生验证

Target 现声明 `coordinate`、`compare`、`select` 和 `top_k`。前端、类型检查及
Triton emitter 复用共享实现；MACA 当前仅准入 resident FP32 tile 的 top-k，
整数 score 与跨循环累积选择由 `MACA_TOP_K_UNQUALIFIED` 拒绝。其他目标的
选择实现不变，未声明的新操作仍由 Target 拒绝。

执行源码 `49f707f5`、job `maca-f2c4ba1f66a2` 对原 MoE 的四个路由阶段进行了
原生组合诊断：group scores、group selection、expert selection、route weights。
保留原始 shape 与六个 case 的输入生成和 CPU `routing_reference`；专家权重
只在 CPU 输入生成时创建，GPU 未执行专家投影、激活或最终组合。

六个 case 共 24 次 native call。48 个 expert ID 全部相等；48 个 FP32 权重在
预先声明的 `atol=rtol=2e-6` 内通过，最大绝对误差为 `5.960464477539063e-08`，
其中 21 个权重 word 与 gold 不同。所有外部输入和 54 份逐阶段输入张量的 bytes
保持，24 份中间输出均通过初始化、有限值或整数域检查。独立复核还检查了
选组与实际 group scores 的稳定排序一致，选出的 expert 均属于所选组。

四个模块的 registers/thread 依次为 20、22、30、13，动态共享内存为
192、48、256、0 bytes，local 均为 0；没有 timing samples。原始数据及
独立复核在外部证据根的 `moe-routing-gpu-49f707f5-v1/` 和
`reviews/moe-routing-device-49f707f5.md`。这些是固定 shape 的路由证据，
不代表任意 top-k 形式、专家 GEMM、完整 MoE、正式多阶段 Workload 收据或性能资格。

操作准入后的静态检查最初只覆盖阶段结构。随后新增了各自精确绑定 C550 的
revision-2 Workload，保留原 B300 合同的数学定义、形状、case、输入、oracle 和容差。
其完整程序验收范围见下一节；旧路由诊断仍保持原来的源码身份和范围。

## 完整 GQA、MLA 和 FP8 MoE 的正确性

执行源码 `315bbcb4920397ac3b5b75223fdc05976526b1e9` 完成了六项 GQA、两项 MLA
和一项完整 MoE 的设备正确性。八项 attention 各有 captured、boundary 两个固定
Workload，每个五类原始输入；MoE 为原 `T=1,E=256,local_E=32,H=7168,I=2048`
Workload 的六类输入。共 **17 个 Workload 变体、86/86 case、368 次原生 kernel 调用**。

GQA 包括 KV heads 为 4/8 的 paged decode、paged causal prefill 和 ragged causal
prefill；MLA 包括 paged decode 与 paged causal prefill。attention 每次调用执行
metadata、scores、normalize、values 四阶段；MoE 每次调用执行选组、选专家、路由权重、
两次专家投影、SwiGLU 与最终组合的原八阶段。结果保留了原 oracle、逐输出容差和 IEEE
规则，没有把中间阶段通过计作完整输出通过。

独立复核重放所有 86 份普通 EvaluationReceipt，检查了 **3,376,608 个输出元素**，
零不匹配，其中包括原合同要求的 65,536 个对应 NaN 和 6,368 个同号 infinity。
MoE 的 43,008 个 BF16 输出全部满足原 `atol=rtol=0.01`；七个 word 与参考不同，
最大绝对误差 `0.00048828125`。专家计算使用原生成程序的 FP8 解码和 FP32 缩放归约，
这不等于原生 FP8 MMA 已验收。

所有 case 的 590 个公开输入字节检查 verdict 均为 true，调用数与封存阶段数一致，
零 fallback、模块正常关闭。归档保存完整输出与参考 bytes，以及输入检查 verdict；
没有把不存在的原始输入快照描述为已重新核对。首项 GQA 的两个程序保留编译源码
`a40d9f06`，其余十五个保留 `315bbcb4`；所有执行身份仍为 `315bbcb4`。

完整原始记录位于外部证据根 `metax-parity-20260920/complete-program-cases-315bbcb4-v1/`
与首例 `gqa-first-captured-primary-315bbcb4-v1/`，独立逐 case 重算及形状表在
`reviews/native-program-device-315bbcb4.md` 和同名 replay JSON。实际能力范围是上述
固定 Workload 的完整正确性，计时均为 null。多阶段计时、自动优化流程、其他形状
和完整模型推理仍分别需要验收。

## 多阶段 profiler

`tools/qualify_tensor_program.py profile --built <sealed-build> --case primary --output <new-path>`
复用上述 CPU 准备与 MACA allocation，先检查完整原始输出，再单独采集一次完整
Program，并再次验证实际输出。新的 `maca_program_activity_v1` 通过普通收据和
attribution feedback 返回逐阶段资源、设备执行时间及阶段间隔。reader 核对全部
stage、原生 API、顺序、同一 stream、封存启动参数和实际资源；不完整采集保留原始行并拒绝。

执行源码 `c5a7499b822b7d7d091db457d5fcc9a0681cf598` 分别完成以下两次原生独立采集。
表中的数值来自单次 instrumented Program 的完整 MCPTI 记录，均为 **attribution only**，
没有冷缓存成对测量、性能样本、稳定性结论或加速比。

| 固定 primary Workload | 实际 job | instrumented 阶段数 / 全部 native calls | 阶段设备时间之和 | 首阶段开始至末阶段结束 |
| --- | --- | ---: | ---: | ---: |
| GQA paged decode、KV heads 4、captured | `maca-73a77add6591` | 4 / 8 | 13.056 us | 194.816 us |
| 完整 FP8 block-scaled MoE、captured | `maca-176780b3f007` | 8 / 16 | 2238.208 us | 2473.472 us |

每次均先执行完整 preflight，再采集并验证一次独立 Program。GQA 两次输出的最大绝对
误差为 `2.384185791015625e-7`；MoE 为 `0.00048828125`，均在原逐元素容差内通过，
所有输入检查为 true，模块已卸载，零 fallback、零额外 dispatch。完整原始 API/kernel
correlation、每阶段启动参数、资源、两份完整输出和参考 bytes 保留在外部证据根的
`gqa-program-profile-c5a7499b-v1/` 与 `moe-program-profile-c5a7499b-v1/`。
GQA 编译身份仍为 `a40d9f06`，MoE 为 `315bbcb4`，两次执行身份均为 `c5a7499b`。

GQA 记录的三个阶段间隔为 70.4、52.224、59.136 us，不能把其 13.056 us 的阶段时间
之和描述为完整程序延迟。MoE 的两个专家投影阶段分别报告 1502.464、711.424 us，
可用于选择后续优化方向；采集本身不证明具体瓶颈或某项改写会加速，promotion disposition
为 **No promotion**。现有单 dispatch timer 继续拒绝 Program；多阶段的正式成对计时与
自动优化仍需后继验收。occupancy、带宽、指令计数和 local-memory reservation 的限制
与前述单 kernel 路径相同。

## Indexed gather 的完整原始 case 正确性

源码 `35c562e9b1255ac323fd05ce51bb9ce80a85bd76` 增加了明确绑定 `xcore1002` 的
`indexed-gather-bf16-v3`。它保留 B300 v2 的数学定义、四个 case、输入生成、独立
oracle、正零越界行为和 BF16 bitwise 判定。四个 case 的全局 tensor shape 不同，
所以编译阶段从同一规范 Schedule 分别生成并密封四个 ABI 对应的 `mcfatbin`，不把
primary 产物当作其他 case 的可变形状实现。

四次普通 confirmatory Evaluation 均由已有 `maca` 本地 broker 串行持锁，且在分配前
完成 task-owned CPU 输入与 oracle 准备：

| 原始 case | 实际 job | module / preflight / native call | 结果 |
| --- | --- | ---: | --- |
| `primary` | `maca-5dbf0d3a4f98` | 1 / 1 / 1 | 0 mismatch，输入不变 |
| `tiny` | `maca-0fff377b0ec4` | 1 / 1 / 1 | 0 mismatch，输入不变 |
| `index_boundaries` | `maca-868d5dd4c710` | 1 / 1 / 1 | 0 mismatch，输入不变 |
| `repeated_indices` | `maca-d287dc5fac80` | 1 / 1 / 1 | 0 mismatch，输入不变 |

四个收据均被公共 Evaluation 接受，最大绝对误差为 0，零 fallback。编译与设备原始记录
保留在 `c550-1:/root/.local/share/open-cake-ir/metax-c550-20260920/` 的
`indexed-gather-build-35c562e9-v1/`、`indexed-gather-device-35c562e9-v1/`，本地外部
证据根另存 `metax-parity-20260920/indexed-gather-35c562e9-evidence.tar.gz`。

这建立的是上述四个固定 Workload case 的 C550 编译与原始 oracle 正确性。所有收据的
timing 均为 null；尚未完成成对计时、profiler、目标框架端到端验收、其他 shape 或性能资格。
