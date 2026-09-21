# 用 Agent 管理优秀算子的 Cake 复现

任务规范只有一个来源：[AGENTS.md](../contracts/scaffolds/kernel-reproduction/AGENTS.md)。
它要求 Agent 自主拆解参考实现、建立结构对应、选择候选、诊断缺口归属，并提出或执行
其权限范围内的 Compiler 演进。研究者不需要逐次决定失败属于 IR、后端还是候选。

## 多架构实验入口

`tools/kernel_experiment.py prepare --config /absolute/experiment-input.json --workspace /absolute/new-experiment`
准备管理 Agent 的 `TASK.md`、规范、参考材料快照及目标列表。管理 Agent 随后调用
`tools/kernel_experiment.py run --workspace /absolute/new-experiment --cell CELL_ID`。
每个 cell 独立选择本机或 SSH 节点、精确 backend、工具链 Python 和已运行的 GPU Infra
socket；在节点创建固定 commit 的独立 worktree，然后调用现有 `launch_task.py`。
该入口不会启动生产 daemon、修改驱动或自动安装工具链。

每个 cell 直接冻结独立的 `run.json`，结果保存在 `run-evidence/` 和经审计的 `report.json`；
普通复现无需 Study。已声明的参考访问、固定基线、验证用例和预算都由同一 Run 约束。
使用原 source checkout 的 `report_task_efficiency.py --workspace ...` 可以重建描述性性能报告。

输入示例（替换模型及所有节点路径；每个 cell 的预算独立计数）：

```json
{
  "schema_version": 1,
  "objective": "复现参考 RMSNorm 的执行结构，并解释性能差距",
  "provider": {"harness": "codex", "model": "YOUR_MODEL", "effort": "high"},
  "budget": {"turns": 8, "token_budget": 300000, "wall_seconds": 7200},
  "references": [{"path": "/absolute/reference/kernel.py", "source": "repository commit and original path"}],
  "cells": [{
    "id": "rmsnorm-b300", "task": "rmsnorm", "backend": "triton-b300",
    "rows": 128, "columns": 4096,
    "node": {"transport": "ssh", "host": "B300-M2",
      "project_root": "/absolute/open-cake-ir", "python": "/absolute/venv/bin/python",
      "kernelctl": "/absolute/gpu-infra/bin/kernelctl", "socket": "/absolute/kernel-infra.sock",
      "workspace": "/absolute/experiments/rmsnorm-b300"}
  }]
}
```

可添加 `metal-m1-pro`、`triton-dcu`、`triton-gfx1151` 等已声明且该任务支持的 cell。
本机节点使用 `transport: local` 并省略 `host`。源 commit 必须已存在于节点仓库，节点
Python 和 host capture 必须满足对应 Executor；不自动把任务改投另一架构。
`node.provider_executable` 可指定 provider 的绝对路径，避免非交互 SSH 的 PATH 差异。
Codex 可直接绑定已安装的原生二进制及其同目录 code-mode host，无需调用 Node.js wrapper。
节点访问模型服务需要代理时，可显式设置 `node.http_proxy`（例如
`http://127.0.0.1:17990`）。入口只对该 cell 的启动进程设置 HTTP/HTTPS 代理，记录在
实验输入中，不修改用户全局环境；不接受含凭证的代理 URL。代理转发必须在运行期间可用。
如果已有外部参考的 sealed baseline bundle，可在 cell 中提供节点上的绝对路径
`fixed_baseline_bundle`。否则仍使用注册任务的 starter，不能据此宣称外部实现性能复现。

管理 Agent 获得源码分析、实验推进和 Compiler 演进的规范；冻结的 Run 作者仍执行已有
候选协议。入口不另外调用一个管理模型，也不创建第二个 Campaign 数据库。
`launches/<cell>/request.json` 和日志保留启动输入；再次启动同一 cell 会被拒绝。
SSH 失败意味着状态可能未知，不能从退出码推断远程实验未启动，更不能自动重提。

## GPU Infra 执行边界

单任务和 `launch_task_matrix.py` 均接受 `--kernelctl /absolute/kernelctl`
与 `--infra-socket /absolute/socket`，不能同时提供旧 `--gpu-run` / `--broker-socket`。
每次 Evaluation 都通过 GPU Infra 快照、异步提交和固定 run-id 观察，实际 GPU 作业由
daemon 的 broker 独占分配。同步 Lab 调用等待这一作业，但不会持卡等待作者生成候选。
daemon 必须报告 `allocation_environment: gpuq_v1`；旧服务会在 provider 启动前被拒绝。

Cake 仍拥有 oracle、paired timing、profiler、confirmation 和最终接受语义。适配器只
投影 judge 正确性和保留原始 receipt，不在 GPU Infra frontier 中重新做性能晋升。
Metal 保留 cooperative 范围与外部占用未知；AMD 与 Hygon 必须匹配不同 broker 后端，
并通过实际 HIP 精确设备检查。规范和 CPU 测试不等于这些节点已经完成运行资格验证。

## 启动输入

在现有 `tools/launch_task.py` 命令的任务、硬件、模型、预算和外部 workspace 参数之外，加上：

```text
--agents-md contracts/scaffolds/kernel-reproduction/AGENTS.md
```

此参数选择该任务的 authoring scaffold，替换默认 scaffold；不是追加一份未绑定的提示。
也可以传入仓库外规范文件的绝对路径。相对路径从仓库根目录解析。
需要加入具体参考源码或分析材料时，在仓库外准备包含这份规范和已审阅参考材料的完整
scaffold，再传入它的绝对路径。保留来源和内容，不要仅给隔离作者一个无法读取的路径。

已有自定义 Study 可将 `arms.open_cake.scaffold` 绑定到同一文件，沿用现有 reference
绑定和 preflight 流程。双 arm 的 matched comparison 仍要求共享 scaffold；若使用这类
Study，规范应按各 arm 声明的 authoring environment 分别执行，不能让比较 arm 改写 Cake。
读取实现的 arm 必须声明 `known_kernel_reproduction`；本规范
不能绕过 clean-start 的参考访问检查。不指定参数时保持原来的 scaffold 选择。

`launch_task.py` 仍使用注册任务的 starter。本参数传入规范与材料，不会自动导入任意
CUDA 工程、注册新 Workload 或将外部源码编译为测量基线。管理 Agent 应先完成这些任务
输入的准备；只有 starter 而没有外部实现时，结果只能称为 starter 优化。

## 每轮如何获得规范

现有链路是：`--agents-md` → Study 的 `arm.scaffold` → CampaignLock → 任务包
`AGENTS.md` → 每轮 provider 请求及其保留的 evidence bundle。
任务包从已绑定的 scaffold 渲染规范，不从机器上任意位置查找 `AGENTS.md`。
`TASK.md` 保留任务与其他参考材料，指向 `AGENTS.md` 中的规范；规范正文只出现一次。
运行中修改原始文件会触发现有输入绑定检查，不会静默更新已冻结任务。

管理整个实验的 Agent 也应在初始任务输入中读取同一规范。该 Agent 可在运行之外准备
参考、整理 Finding、修改代码并验证；冻结 Run 内的作者仍只写 candidate envelope。
这份规范没有创建后台管理进程，也没有让受限作者获得修改 Compiler 的工具。

## 结果与验证

结构对应、最小复现、差距归因、性能比较与 promotion disposition 写入既有实验结果，
运行产物保持在仓库外。Compiler 缺口使用既有 Finding 生命周期；修改提交后在隔离
worktree 验证，再用后继 Campaign 测量。无需新建 Study kind 或第二套证据账本。

规范是 authoring treatment。修改规范后启动新任务并保留旧输入，不重标历史结果。
被传入请求的内容可核对，但传入本身不证明模型遵循了它；结构、正确性和性能仍由
实际候选及外部评测证据判断。


## 多节点 Codex 状态目录

可在 cell 的 node 中显式绑定 `codex_home`，指向该节点上已准备好的绝对目录。
入口仅为该 cell 的 launcher 设置 `CODEX_HOME`，initial/resume 及资格验证共享此绑定。
使用节点本地目录可隔离跨主机共享 HOME 下的临时 helper 和会话状态；它不会关闭 sandbox，
也不会自动复制凭据、修复旧运行或创建目录。身份与初始/恢复行为仍需新的两轮 qualification。
旧运行的失败记录不重分类；换绑定后必须准备新实验输入与运行目录。

## Native tensor Program correctness qualification

Composed FlashInfer attention and MoE factories accept an explicit `backend`, for example
`attention.workload_document(task, backend="triton-metax", variant="boundary")`. The default
B300 revision-1 documents remain unchanged; another admitted target receives a revision-2
identity while preserving the original mathematics, input distributions, oracle and tolerances.
The existing `owner.launch_plan(WorkloadContract(document))` returns a Compiler `Program`.
Write these generated Workload and Program documents outside the checkout.

`tools/qualify_tensor_program.py build --workload <workload.json> --program <program.json>
--output <new-build-directory>` uses the existing Compiler, source admission, isolated
Triton builder and sealed Program bundle. Run build in the captured CPU-only compilation
environment. No provider or GPU execution occurs in this command.

The local MACA correctness adapter is
`tools/qualify_tensor_program.py evaluate --built <build-directory> --case <case-id>
--output <new-result-directory>`. Invoke it once for every original Workload case, using a
new output directory each time. The process materializes one original CPU tensor case and
computes its original oracle before acquiring the existing MACA broker allocation. Keeping
one case per process avoids holding all MoE weight cases in memory at once. Inputs cross
the device boundary as raw bytes and retain their declared tensor dtype; they are never
expanded into Python float lists. The existing Program loader owns ordered native dispatch,
intermediate storage, alias checks, module lifetime and stream identity.

Each result retains a common `EvaluationReceipt`, full observed/expected output bytes,
input-effect verdicts, the true native stage count, and JSON-null timing samples. IEEE
NaN/Inf output rules remain the Workload's own. An execution or teardown error prevents a
passing receipt. The worker holds its real allocation through process exit. All source,
software, host and review gates must pass before any device invocation.

These commands qualify construction or correctness only. They do not create an optimization
Run endpoint or measured speedup. MACA multi-stage timing is explicitly refused by the
existing single-dispatch timer until its interval and profiler coverage are qualified.
