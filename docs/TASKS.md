# 任务代码与通用实验引擎

[English](en/TASKS.md)

`lab/` 负责如何迭代，`evaluation/` 负责通用评测记录和执行边界。
具体算子的输入、参考实现、形状准备和结果解析都由 `tasks/` 管理。

| 目录 | 内容 |
| --- | --- |
| `src/open_cake_ir/lab/` | Ralph、预算、候选提交、反馈路由、共同证据回放 |
| `src/open_cake_ir/evaluation/` | 工作负载数据类型、张量 ABI、收据、CUDA 生命周期、CUPTI 和 profiler |
| `src/open_cake_ir/tasks/qsa/` | QSA 合同校验、参考实现、Program、编译/运行入口和反馈投影 |
| `src/open_cake_ir/tasks/flash_kmeans/` | K-means 合同、输入/参考、固定 ABI、候选编译、校准和 portfolio |
| `src/open_cake_ir/tasks/tinygemm/` | TinyGEMM 合同、输入/参考和评测 |
| `src/open_cake_ir/tasks/tiles/` | RMSNorm、GEMM+bias、indexed gather 共用的任务实现 |
| `src/open_cake_ir/tasks/dsa/`, `tasks/kda/` | 各自的严格语义合同校验 |

任务组合入口是 `tasks/runtime.py` 的 `TaskLab` 和 `tasks/compose.py`。
CLI 的 `lab` 命令使用这份组合。通用 `Lab` 不导入任务；它接收明确的工作负载加载、
Schedule 准备、任务准入和启动描述解析函数。没有动态插件发现、导入字符串或旧路径别名。

```python
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import load_workload

workload = load_workload("contracts/workloads/rmsnorm-fp32-v2.json")
lab = TaskLab(".")
```

`tasks/workloads.py` 是内置任务的唯一静态选择表。通用 `WorkloadContract` 保存和检查
共同结构，任务模块仍执行原有的严格语义校验。K-means 的固定 ABI 由它自己的工作负载
适配类型提供；通用 authoring 环境只从同一工作负载读取形状和目标，不接受另一套值。

新增算子时，把语义实现放入对应任务目录，并引用唯一的 Workload Contract。
已有的纯张量任务可以复用 `tasks/tiles/`，无需为每个算子复制通用机制。
`contracts/workloads/` 等冻结合同保留原始身份和引用；其中历史 oracle 名称不用于动态导入。
QSA 的参考实现原始字节保持不变。历史发布与证据需在其原始 Git 版本下读取。

QSA 反馈入口现在是 `python -m open_cake_ir.tasks.qsa.project_feedback`，只提供
`compiler` 和 `evaluation`。旧的 `qsa_next_turn_request` 和 `turn` 子命令已移除；
Ralph 控制器是下一轮 StateCard 和 TurnRequest 的唯一生成路径。

K-means 的 portfolio 是该任务的验证流程，其固定 B/N/K/D 形状、seed、dispatcher、
旧结果解析和运行组合都在该任务目录中。通用层没有为它保留特例实现。
