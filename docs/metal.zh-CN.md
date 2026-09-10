# 通过 TaskLab 运行 Apple Metal 任务

Compiler 支持精确目标 `Apple M1 Pro` / `apple_gpu_family7`、`Apple M2` /
`apple_gpu_family8` 和 `Apple M4` / `apple_gpu_family9`，任务入口均已接入，参数分别为
`--backend metal-m1-pro`、`--backend metal-m2` 和 `--backend metal-m4`。一个 backend 只对应一个精确目标和一个已认定设备名；
为某台设备封存的 Workload 不会在另一台上通过校验。入口核对精确设备、OS、工具链和已发布
Executor，若已发布 Executor 绑定的是另一块 Apple GPU 会直接拒绝，不自动替换设备，
也不借用其他 Apple GPU 的成本校准。因此在 M2 或 M4 上运行都需要一份在对应主机上采集并发布的
Metal Executor，其他 Apple GPU 的那份不能复用。

内置任务为 `rmsnorm`、`layernorm`（带仿射参数、中心化总体方差）和 `residual_rmsnorm`
（先将残差加法舍入到 FP32，再归一化）。[任务自己的 Workload 与 oracle](../src/open_cake_ir/tasks/normalization/workload.py)
固定 rank-2 FP32 ABI、epsilon、输入范围、种子和容差。每个 Workload 只覆盖一个形状，
包含 primary、零值、近零值、交替符号和混合幅度五类必测输入；全部通过后才测量 primary。
不同形状属于不同 Workload，单形状结果不构成 portfolio 或框架集成结论。

## 启动一个任务

需要已有的 Apple Silicon/macOS 15+ 环境，以及经审查发布且通过完整 Corpus Gate 的 Compiler。
当前发布的 Executor 必须为 Metal Executor，其 Python、Swift、SDK、设备/OS 与原生 helper
必须和本机一致。入口会报告缺失或不匹配的条件，不自动安装、修复环境或发布版本。

在 checkout 中，使用 Executor 声明的 Python 执行：

```sh
python3 tools/launch_task.py \
  --task rmsnorm --backend metal-m4 \
  --harness codex --model "<exact-model-id>" --effort high \
  --workspace "$HOME/.local/share/open-cake-ir/runs/metal-rmsnorm-example" \
  --rows 128 --columns 1024 --turns 4 --token-budget 150000
```

将模型占位符替换为实际配置的精确模型。Claude Code 使用 `--harness claude-code`，
并填写它的精确模型标识和支持的 effort。harness、模型和 effort 均为必填实验条件，
不会静默使用别名或替代模型；Claude 原生事件报告的模型必须与请求一致。

`--workspace` 必须是所有 Git checkout（包括父仓库）之外的新绝对路径。
actor 工作区只创建一次，Ralph 各轮保留它并继续同一个 provider 会话。
重复使用已有任务根目录会拒绝，不会重置历史状态。可通过 `--provider-executable` 指定 CLI；
省略时按 harness 在 PATH 中查找 `codex` 或 `claude`。已知的 Codex npm wrapper 会解析到
它自己安装包中的原生可执行文件，不替换为另一份安装。

broker 和 worker 使用 Executor 声明的 Python、`-I` 和同一 checkout 中的绝对路径
bootstrap 启动，无需外部 `PYTHONPATH`。继承的 `PYTHONPATH`、`PYTHONHOME` 不能
将它们切换到另一份源码；broker 仍通过 exec 保留作业身份和继承的锁描述符。

[薄入口](../tools/launch_task.py) 在该目录写入 Workload、可读的 `starter.py`、Study 模板和
运行时绑定，然后通过公共 Open Cake 环境准备封存基线，取得或验证真实 provider 资格，
执行 `TaskLab.preflight`，保存 `campaign-lock.json`，再调用现有
[TaskLab composer](../src/open_cake_ir/tasks/compose.py)。候选过滤、反馈、确认、token/时间记账
和停止条件始终由 Ralph 管理。

已有封存基线和资格时，可传入外部路径参数 `--fixed-baseline-bundle`、`--qualification` 和
`--qualification-anchor`，由 preflight 核对绑定。`--preflight-only` 在保存 Campaign Lock 后停止；
它仍可能编译基线和调用 provider 资格验证，因此不是离线测试选项。

## 资格和评测边界

公共 [provider 资格入口](../tools/qualify_codex_provider.py) 使用同一份不可变 TASK.md/AGENTS.md，
实际观察两轮操作：新增包含 Python 源码的候选 envelope，再于同一工作区、同一会话更新。
它保留原生事件、实际 token 用量和受保护文件检查。可执行测试替身必须使用 `--fixture-only`，
其收据不能授权真实任务。

公共 Evaluation 只接收已封存的 Metal binary archive 和明确的 Workload 启动 ABI，
严格命中 archive 后加载，不编译候选源码。[本地 broker](../src/open_cake_ir/evaluation/local_broker.py)
串行化本项目的作业，但不声称其他应用没有使用 GPU。

[任务 Study 策略](../src/open_cake_ir/tasks/normalization/study.py) 明确声明十组交替的候选/基线配对，
每个 cohort 先 warmup 三次、再采集 25 个样本，materiality ratio 1.05、方向判定要求六组获胜。
这些是工程测量规则，不是目标校准，也不预设加速。
计时区间为完成后的 Metal command buffer，不是纯 kernel latency，不继承 CUPTI/L2 flush 语义。

`fixed_baseline_paired_metal_v2` 另外声明两个值。`dispatches_per_sample` 是每个计时
command buffer 连续编码的 dispatch 数量，一个样本等于该 buffer 的时间除以它。编码、提交
和完成的开销每个 buffer 只付一次，因此比这个开销更短的 kernel 否则主要是在测量该开销；
kernel 从未被修改的输入重写全部输出，重复执行不改变被校验的缓冲区。`maximum_relative_iqr`
在同一个数值上，把 cohort 离散度的判定从变异系数换成原始样本的相对四分位距。两者都描述
离散度；本测量明确声明未排除其他 GPU 客户端，而个别被干扰的样本不应否决一个主体稳定的
cohort。原有的 `fixed_baseline_paired_metal_v1` 保持其精确的单次 dispatch、CV 判定语义，
已冻结的 Study 照常重放。
Profiler 单独采集 compute-stage 时间戳；缺少原生能力仍会拒绝，不推导物理寄存器、spill、
occupancy、带宽或指令数量。

静态 lowering 使用 32-lane 条带映射、安全 MSL 2.3 数学和显式启动元数据。
CPU 语义、原生编译、设备正确性、稳定计时和框架验收是不同证据域。
目前没有经校准的 Apple 成本排名模型，GPU 前的过滤会保留未排名原因。

以下可移植检查既不调用真实 provider，也不执行 GPU：

```sh
PYTHONPATH=src python3 -m unittest \
  tests.contracts.test_normalization_tasks tests.contracts.test_task_launch \
  tests.contracts.test_metal_task_composition tests.contracts.test_harness_qualification
```

生成代码体通过已有 C++ 编译器执行 CPU 语义检查，资格测试使用明确的可执行替身。
真实 Campaign 资格与性能结论仍需经审查发布的版本和实际设备执行。
