# 统一任务目录

[实验流程](wiki/experiments.md) · [执行手册](RUNBOOK.md) · [复现集合](../experiments/flashinfer_rewrites/README.md)

从这里选择任务，再通过现有 Run 入口执行。任务成员和单任务默认形状由
`src/open_cake_ir/tasks/catalog.py` 统一选择；算子语义、oracle、目标约束仍由各任务自己的
Workload factory 负责。单任务、默认批量、多节点复现和参考集合共用这个选择边界。
参考源码与接入状态继续由复现包的 `catalog.json` 维护，统一目录直接读取它。

## 查看任务

在项目根目录使用已安装的 `open-cake-ir`，或 `python -m open_cake_ir.cli`：

```sh
open-cake-ir tasks list
open-cake-ir tasks list --suite portable
open-cake-ir tasks list --family contraction
open-cake-ir tasks list --suite flashinfer-rewrites
open-cake-ir tasks show cake_tinygemm2
open-cake-ir tasks show 027_cake_kda_prefill
```

三个集合的关系：

| 集合 | 内容 | 选择与执行 |
| --- | --- | --- |
| `all` | 现有单任务 launcher 的完整 authoring 集合 | `tools/launch_task.py --task TASK` |
| `portable` | normalization、activation、rowwise、reductions、optimizers、contraction、gemm | `tools/launch_task_matrix.py` 的默认集合；每项仍须检查自己的目标和形状 |
| `flashinfer-rewrites` | 已保存参考的 FlashInfer/CAKE 集合，包括未接通项 | `tools/rewrite_collection.py` 默认只选择 `ready` 项 |

集合有重叠，数量不能相加。不同固定形状的 task id 不等于不同算子族。
清单中的“已注册”表示作者任务入口存在，不表示所有目标都可运行。
复现集合中的 `skipped`、`blocked` 会显示原有原因；参考齐全不代替 Workload、oracle
或完整 authoring 接线。不要用已注册的简化任务覆盖这些未接通项。

## 检查一个精确目标与形状

```sh
open-cake-ir tasks check --task silu --backend triton-metax
open-cake-ir tasks check --task gemm --backend triton-b300 --depth 256
open-cake-ir tasks check --task rmsnorm --backend metal-m4 --rows 128 --columns 1024
open-cake-ir tasks check --suite flashinfer-rewrites --backend triton-b300
```

`check` 使用干净提交，实际构造 Workload 和 starter、解析 Python、调用 Compiler
assessment，并尝试生成源码。它不编译设备二进制、不启动作者、不申请 GPU、不执行 kernel。
检查不改输入形状、不替换目标、不放宽规则。返回码：全部生成源码为 `0`；存在拒绝或未接通
项为 `1`；选择、输入或源码环境错误为 `2`。形状覆盖参数只允许用于一个选中的任务。

普通 contraction 和 `gemm_bias` 的单任务入口要求显式 K；目录不会替作者猜测它。
默认批量入口保留原有的 K=256 选择，JSON 目录的 `matrix_depth_argument` 单列这项参数；
`tasks check` 默认检查单任务形状，缺少必需的 K 时会明确拒绝。
FIB GEMM 的 N/K 仍是上游常量；其 batch 默认值与 BF16 normalization 的默认值来自各自
task owner。实际限制由 factory 和 Compiler 检查，目录不复制一份硬件能力表。

| 显示的状态 | 已检查的范围 | 仍缺什么 |
| --- | --- | --- |
| 已注册 / 作者入口已接通 | 任务选择、参考包接线 | 指定目标和形状的可生成性 |
| 源码已生成 | 本次形状的构造、类型、Verifier、lowering | 设备工具链、provider、资源准入、正确性、测量与确认 |
| 拒绝 | 具体构造阶段或 Compiler Finding | 根据原诊断修候选或整理系统缺口 |
| 未接通 | 参考集合自己的接入记录 | 补完整 Workload、oracle 或 authoring 路径 |
| host `present` / `missing` | 当前树中是否有该精确 Target 的采集文件 | 文件存在不证明当前节点环境、租约或测量资格 |

目录把未检查的 runtime 项明确列为 `not_examined`。完整 Corpus Gate 不会在这里自动运行；
正式实验仍通过 launcher 和 Lab 的既有准入。clean-start 的实际执行仍受 provider 读取隔离
资格限制；目录检查通过不能解除该限制。

## 导出与启动

```sh
open-cake-ir tasks list --format json
open-cake-ir tasks list --suite flashinfer-rewrites --format markdown
open-cake-ir tasks check --task silu --backend triton-metax --format json
```

导出是当前目录或固定形状源码检查的阅读投影，不是 GPU Evaluation Report。
需要保存时，把输出重定向到仓库外的新文件。正式实验输入、运行输出和证据继续放在仓库外。

选择后沿用现有入口：单项使用 `launch_task.py`；批量使用 `launch_task_matrix.py`；
明确的形状和节点配置使用 `kernel_experiment.py`；已有参考集合使用 `rewrite_collection.py`。
这些组织入口最终共用冻结的 Run、共同 Evaluation 和审计，没有新增运行模式或判分规则。
科学比较另由 Study 预分配处理条件，不能把目录导出或工程 Run 事后重标成研究分组。

## 添加与维护

1. 在任务 owner 完成 Workload、独立 oracle、starter 和实际支持域。
2. 同族任务加入该族原有 registry；新族在统一目录声明成员来源和分组。
3. 有特殊默认形状时在目录接入 owner 的参数；显式输入和 owner 拒绝保持有效。
4. 参考包只有在 authoring task 已注册后才能标记 `ready`；目录读取时拒绝陈旧声明。
5. 在固定提交的独立 worktree 验证选择、factory、拒绝反例和受影响入口，再提交 PR。

设备正确性、性能、晋升和 Compiler 缺口分别沿用 Evaluation、结果记录和 Finding 的现有
负责人。目录不另外维护手填的“已验收”成绩、通用支持矩阵或第二份缺口账本。
