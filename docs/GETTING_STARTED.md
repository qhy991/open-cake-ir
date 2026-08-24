# 从一个真实 B200 GPU 例子理解 open-cake-ir

这是一条面向编译器初学者的 10–15 分钟路径（不含 GPU 排队时间）。你会把一份描述 GPU 执行计划的
Schedule JSON 交给 Compiler，确定性 Lowering 为 Triton 源码；固定的 Triton toolchain 再把源码编译成
B200 CUBIN。随后通过 GPUQ 在独占 B200 上加载候选 CUBIN，并恰好启动一次其中的 kernel；独立 oracle 会
检查 16,384 个输出。

这个例子不调用 Agent、不比较 Open Cake 与 CUDA、不测性能，也不产生论文或科学结论。它只回答：
“这份固定 Schedule 能否经过真实 Compiler 和 GPU 正确跑通？”

## 1. 先建立心智模型

```text
数学任务
  -> Schedule（结构化 GPU 执行计划）
  -> Assessment / Finding（Compiler 检查）
  -> Lowering（确定性生成 Triton 源码）
  -> Triton toolchain compile
  -> CUBIN（B200 加载的二进制）
  -> GPUQ 独占 B200 worker 加载候选 CUBIN，并从中 launch 一次 kernel
  -> Oracle 正确性检查
```

- **GPU kernel** 是在 GPU 上由大量线程并行执行的函数。
- **Schedule** 不写最终机器指令；它声明数据、切块、线程分工、操作依赖和循环策略。
- **Assessment** 告诉你 Schedule 是否被接受；**Finding** 精确指出不满足规则的 JSON 路径。
- **Lowering** 把被接受的 Schedule 翻译成目标源码，但此时还没有运行 GPU。
- **CUBIN** 是固定 Triton toolchain 编译、CUDA Driver 最终加载到 B200 的二进制；它不由 Compiler
  Lowering 直接生成。
- **Oracle** 是独立于候选 kernel 的可信参考计算。

## 2. 示例在计算什么

Workload 是 Flash-KMeans assignment。对每个 token，计算它与 1,024 个 centroid（聚类中心向量）的平方
欧氏距离，并输出距离最小的 centroid 编号。MMA 表示矩阵乘加。

本例固定形状：

```text
B=32 batches, N=512 tokens, K=1024 centroids, D=128 dimensions
```

完整 Schedule 位于
[`examples/gpu/flash-kmeans-b32-smoke.json`](../examples/gpu/flash-kmeans-b32-smoke.json)。下面只是帮助阅读的
摘录，不是另一份配置：

```json
{
  "buffers": [
    {"name": "tokens", "shape": [32, 512, 128], "mode": "input"},
    {"name": "centroids", "shape": [32, 1024, 128], "mode": "input"},
    {"name": "assignments", "shape": [32, 512], "mode": "output"}
  ],
  "operations": [
    {"id": "load_tokens", "kind": "load"},
    {"id": "load_centroids", "kind": "load"},
    {"id": "distance_mma", "kind": "mma", "depends_on": ["load_tokens", "load_centroids"]},
    {"id": "argmin", "kind": "reduce_argmin", "depends_on": ["distance_mma"]},
    {"id": "store_assignment", "kind": "store", "depends_on": ["argmin"]}
  ]
}
```

直觉上就是：读取 token 和 centroid → 计算距离 → 找最小编号 → 写回结果。

## 3. 本地只运行 Compiler（无需 GPU）

从私有仓库安装：

```bash
git clone https://github.com/qhy991/open-cake-ir.git
cd open-cake-ir
python3.11 --version
python3.11 -m venv .venv
./.venv/bin/pip install -e .
```

先运行不会提交 GPU 的准备步骤：

```bash
./.venv/bin/python examples/gpu/flash_kmeans_quickstart.py \
  --project-root . \
  --prepare-only
```

关注这些字段：

```json
{
  "status": "prepared",
  "assessment": {
    "accepted": true,
    "lowering_eligible": true,
    "findings": [],
    "target": "sm_100a"
  },
  "evaluation": {"gpu_submitted": false}
}
```

含义是：Compiler 接受了 Schedule，也能 Lowering；还没有申请或运行 GPU。

## 4. 在 verda-b200x4 上运行真实 GPU

服务器已经有受控的 CUDA/Triton 环境：

```bash
ssh verda-b200x4
cd /home/qinhaiyan/open-cake-ir
id -nG
```

`id -nG` 必须包含 `gpuq-users`。不要直接设置 `CUDA_VISIBLE_DEVICES` 或绕过 GPUQ 占卡。运行：

```bash
/home/qinhaiyan/agent-gpu-broker/bin/gpu-run \
  --label open-cake-ir-getting-started \
  --mode exclusive \
  --gpu-count 1 \
  --cwd /home/qinhaiyan/open-cake-ir \
  --estimate 5m \
  --queue-timeout 30m \
  --run-timeout 15m \
  --env GPUQ_JOB_ID=gpuq-000000000000 \
  /home/qinhaiyan/megakernel-exp/.venv-flashinfer/bin/python \
  /home/qinhaiyan/open-cake-ir/examples/gpu/flash_kmeans_quickstart.py \
  --project-root /home/qinhaiyan/open-cake-ir
```

`GPUQ_JOB_ID` 这里是 worker admission 的固定占位符；真实 job ID 由 `gpu-run` 在父进程 stderr 中报告。
概念图按产物关系排列；真实命令先由 GPUQ 分配受控 worker，然后在该 worker 中 compile 并 launch。

## 5. 真实验证结果

2026-08-23 在 `verda-b200x4`（hostname `warm-sun-begins-fin-03`）上实际运行得到 GPUQ job
`gpuq-5209de9fd52b`。关键结果为：

```json
{
  "status": "passed",
  "executor_revision": {
    "executor_id": "open-cake-ir-b200-v6",
    "canonical_sha256": "28ae0cd7f4c09f0b2f6d0335e674f56447d33ecef29a41a6a74835fce9d89f4a"
  },
  "candidate": {
    "candidate_record_sha256": "79c4fa920d02895157126e70661e466513ccd99e835ccad5286252e083963e1f"
  },
  "build": {
    "target": "sm_100a",
    "cubin_size_bytes": 149792,
    "cubin_sha256": "e1e65c7335f022c9cc7f9450caa8ad9838e9524df04627f6907df9cc66fab647"
  },
  "gpu": {
    "name": "NVIDIA B200",
    "compute_capability": [10, 0],
    "mode": "exclusive",
    "exclusive_b200_preflight_passed": true
  },
  "evaluation": {
    "correctness_passed": true,
    "evaluation_protocol_sha256": "5129ef0766a4b2558cb0c8f3e3b4931f2ff1415c36fa976b80f43b69f81af01d",
    "metrics": {
      "exact_match": true,
      "mismatch_count": 0,
      "total_assignments": 16384
    },
    "kernel_calls": 1,
    "fallback_calls": 0,
    "module_unloaded": true,
    "performance_measured": false,
    "scientific_claim_authorized": false
  }
}
```

验收权威是 `correctness_passed`，它由 Workload Contract 的 tie-aware 规则派生；`exact_match` 只是本次观察，
不是额外发明的正确性规则。资格验证摘要及其绑定的原始结果/父进程日志位于
[`inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json`](../inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json)。
该摘要绑定 create-only worker result、完整 stdout 和包含真实 job ID 的 `gpu-run` stderr。

## 5b. 另外三个算子

教程例子是 Flash-KMeans，但语料里有四个可发射的 Profile。另外三个都是行内归约，都只声明 Schedule、不改后端：

| Schedule | Profile | 后端 | 需要的新词汇 |
| --- | --- | --- | --- |
| `corpus/schedules/rmsnorm-b8-smoke.json` | `rmsnorm_b8_smoke` | Triton | — |
| `corpus/schedules/softmax-b8-smoke.json` | `softmax_b8_smoke` | Triton | `exp`、`div`、`max` 折叠 |
| `corpus/schedules/layernorm-b8-smoke.json` | `layernorm_b8_smoke` | Triton | 无 |

不需要 GPU 就能看它降出什么：

```bash
./.venv/bin/python -c "
from pathlib import Path
from open_cake_ir.compiler import Compiler
c = Compiler.load(Path('.'), Path('compiler/revision.lock.json'))
print(c.lower(c.assess_file(Path('corpus/schedules/layernorm-b8-smoke.json'))).source)
"
```

在 B200 上核对正确性，用的是仓库里的仪器而不是临时脚本——记录会写成一个新文件，它拒绝覆盖已有的：

```bash
python tools/observe_lowered_kernel.py \
  --schedule corpus/schedules/layernorm-b8-smoke.json \
  --observed-at 2026-01-01T00:00:00Z \
  --out /new/path/LAYERNORM_OBSERVATION.json
```

`inventory/*_OBSERVATION_*.json` 是已经留下的记录。每一条都钉住它运行过的那个产物的 `source_sha256`，所以改了 Lowering，证据就与代码脱钩——正确的做法是重新观测，永远不是改记录。

## 6. 三条路径不要混淆

| 路径 | 回答的问题 | 能否形成科学结论 |
| --- | --- | --- |
| Teaching smoke | 一个固定 Schedule 能否在一张 B200 上正确运行 | 否 |
| System qualification | Compiler、Lab、Evaluation、Evidence 能否完整组合 | 否 |
| Scientific Study | 两种 Authoring Environment 是否存在可估计差异 | 仅在冻结计划和足量独立 Runs 后 |

## 7. 常见失败

- `gpu_admission_differs`：没有通过 GPUQ 获得独占 B200、缺少占位 `GPUQ_JOB_ID`，或卡上已有进程。
- `Executor Revision file ... differs`：服务器源码与当前 Executor Revision 不一致，先同步并重新验证。
- `Schedule Workload binding differs`：Schedule 不是为当前 Workload v2 和 `b32_smoke` case 冻结的。
- `correctness_rejected`：候选 CUBIN 已运行，但输出未通过 Workload Contract；不能继续解释性能。
- GPUQ 排队超时：这是基础设施状态，不是 Compiler Finding，也不是候选结果。

## 8. 下一步读什么

- 想理解 Compiler 的正式术语：[`contexts/compiler/CONTEXT.md`](contexts/compiler/CONTEXT.md)
- 想编写完整 Schedule：[`compiler/AUTHORING_CONTRACT.md`](../compiler/AUTHORING_CONTRACT.md)
- 想运行 Agent 优化或科学实验：[`RUNBOOK.md`](RUNBOOK.md)
- 想审计结论边界：[`ACCEPTANCE_GATES.md`](ACCEPTANCE_GATES.md)

### 缩写

- **IR**：介于算法意图与目标代码之间的结构化表示。
- **PTX**：CUDA 的虚拟指令表示。
- **CUBIN**：GPU 加载的 CUDA 二进制。
- **SASS**：具体 GPU 架构执行的机器指令。
- **CAS**：以内容哈希寻址的不可变对象存储。
- **CUPTI**：CUDA 的性能分析接口；本教学 smoke 不使用它。
- **GPUQ**：为任务分配独占 GPU 的 broker。
- **Estimand**：科学 Study 在执行前声明、准备估计的目标量。
