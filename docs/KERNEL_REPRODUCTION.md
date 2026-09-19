# 用 Agent 管理优秀算子的 Cake 复现

任务规范只有一个来源：[AGENTS.md](../contracts/scaffolds/kernel-reproduction/AGENTS.md)。
它要求 Agent 自主拆解参考实现、建立结构对应、选择候选、诊断缺口归属，并提出或执行
其权限范围内的 Compiler 演进。研究者不需要逐次决定失败属于 IR、后端还是候选。

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
