# 用 Agent 管理优秀算子的 Cake 复现

任务规范只有一个来源：[AGENTS.md](../contracts/scaffolds/kernel-reproduction/AGENTS.md)。
它要求 Agent 自主拆解参考实现、建立结构对应、选择候选、诊断缺口归属，并提出或执行
其权限范围内的 Compiler 演进。Cake 探索与 lowering 假设要求也由该规范维护：
作者需结合已交付 API、候选结构和可见低层证据判断下一步，不能把 starter 当作表达能力上限。
研究者不需要逐次决定失败属于 IR、后端还是候选。

可移植任务集合与 CAKE 原框架能力评估见
[改写任务包](../experiments/flashinfer_rewrites/README.md)。其中 027–030 对应论文的
KDA prefill/decode、TinyGEMM2 和 Alpha-MoE；`rewrite_collection.py assess` 生成
源码绑定的组件探针结果和管理任务包，完整 GPU 改写仍需逐项完成其 Workload 与验收接入。

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
  "schema_version": 2,
  "objective": "复现参考 RMSNorm 的执行结构，并解释性能差距",
  "provider": {"harness": "codex", "model": "gpt-6.1-sol", "effort": "xhigh"},
  "budget": {"turns": 8, "token_budget": 150000, "wall_seconds": 3600,
    "max_candidates": 3, "searches_per_turn": 3,
    "max_compilations": 24, "confirmation_seconds": 600},
  "cells": [{
    "id": "rmsnorm-b300", "task": "rmsnorm", "backend": "triton-b300",
    "rows": 128, "columns": 4096,
    "references": [{"path": "/absolute/reference/kernel.py", "source": "repository commit and original path"}],
    "node": {"transport": "ssh", "host": "B300-M2",
      "project_root": "/absolute/open-cake-ir", "python": "/absolute/venv/bin/python",
      "kernelctl": "/absolute/gpu-infra/bin/kernelctl", "socket": "/absolute/kernel-infra.sock",
      "workspace": "/absolute/experiments/rmsnorm-b300"}
  }]
}
```

模型与 effort 是显式输入；示例按当前实验选择 `gpt-6.1-sol` / `xhigh`，不是 launcher
全局默认值，也不证明该 CLI/模型组合已经通过资格。初始与续轮均使用同一绑定。

`budget` 为每个 cell 单独计数，不从 scaffold 或参考材料读取。`turns` 与 `wall_seconds`
必填；`token_budget` 可省略或为 `null`。schema v2 另外允许以下四项独立覆盖，均可省略，
不要求同时填写：

| v2 可选字段 | 传入现有 launcher 的参数 | 含义 |
| --- | --- | --- |
| `max_candidates` | `--max-candidates` | 每个 Turn 最多提交的候选数，正整数 |
| `searches_per_turn` | `--searches-per-turn` | 每个 Turn 的搜索评测配额，正整数，不超过最终解析的候选数 |
| `max_compilations` | `--max-compilations` | Run 内 native source-to-artifact 编译入口调用上限，正整数，包含失败调用和内部 variants |
| `confirmation_seconds` | `--confirmation-seconds` | 总 wall budget 内预留给确认阶段的秒数，正有限数，可带小数，严格小于 `wall_seconds` |

其余预算计数必须为正整数，布尔值、字符串与自动数值转换不被接受。
`token_budget` 是 provider token 停止阈值，在 Turn 边界检查，不是单次响应的硬截断；
省略或 `null` 仍记录用量，但不启用 token 停止阈值。
例如上述配置向既有 Run budget 投影为最多 8 Turns、每 Turn 3 候选和 3 次搜索、
全 Run 24 次搜索配额、24 次 attribution 配额和 24 次编译调用；600 秒确认预留包含在
3600 秒总预算内。配额不是实际执行次数，既有 `task_run_inputs` 继续推导其余 Run 限制。

管理器保留省略项，不填补默认值。固定 source commit 的 `launch_task.py` 解析默认值，
`task_run_inputs` 校验完整预算；这两个入口仍是实际执行参数与 Run 预算的 owner。
当两个计数字段都显式填写时，管理器先检查 `searches_per_turn <= max_candidates`；
只填写其中一项时，它与 CLI 默认值是否相容由真实运行入口检查。
`prepare` 成功只表示输入可准备，不表示 Run 已冻结、通过资格或一定能启动。
完整预算校验先于 stack admission、native baseline 构建、provider qualification 和 GPU
评测；此前节点可能已经创建输入与 source worktree、解析 provider 可执行文件并查询
GPU Infra `node-status`。失败仍应检查保留的启动记录，不自动重试同一 cell。
schema v1 继续只接受原有的三个预算字段；新字段不会改变旧版配置或其 CLI 默认行为。

schema v2 为每个 cell 创建 `cells/<id>/TASK.md`、`AGENTS.md`、`references/` 和
`scaffold.md`。每项必须声明自己的 `references`；没有顶层资料继承或缺失文件回退。
共同材料也要显式出现在各自列表中。`run --cell` 只传该 cell 的准备快照，后续修改原始
资料不会悄悄进入它。旧 schema v1 继续使用顶层 `references` 与同一 scaffold，不能同时
混入 v2 字段；旧实验仍从其固定提交执行。

cell 可选 `agents_md: "/absolute/reviewed-task-AGENTS.md"`，完整替换该 cell 的默认复现
规范，其快照写入该 cell 的 `AGENTS.md`。这可承载任务专属的 Cake/Metal 技能说明，但
不会安装 CLI skill、运行依赖脚本或扩大工具权限。`references` 正文仍是数据，不是指令。
实际 Run 的 TASK/AGENTS 仍由 Lab 根据冻结权限生成；准备目录里的 TASK 只指导管理 Agent。
模板与脚本依赖若需作为原生 skill 自动发现，须另行声明和验证作者环境策略。

重复实验使用不同 cell id 和外部 `node.workspace`，例如
`/absolute/experiments/rmsnorm/C0/r01/run` 与 `.../r02/run`。对应 `run-inputs/` 和
`run-source/` 由既有入口创建在同一 replica 目录。每个 Run 保留自己的作者目录、预算、
证据与报告；同一任务的相同工具链可以复用固定安装，无需重复安装一套 Compiler。

Claude-compatible 网关若有已核对的响应模型别名，可在 `provider` 中明确声明
`response_model_aliases` 列表；入口将其逐个传给单任务 launcher，沿用已有 qualification
与回放校验。请求模型仍必须精确匹配，别名不从名称猜测，也不绕过模型或工具权限检查。

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

普通复现的链路是：`--agents-md` → Run 的 authoring scaffold → 任务包 `AGENTS.md`
→ 每轮 provider 请求及其保留的 evidence bundle。独立 Run 无需 Study 或 CampaignLock；
旧配对 Study 由适配边界绑定 scaffold 后投影到同一 Run。
任务包从已绑定的 scaffold 渲染规范，不从机器上任意位置查找 `AGENTS.md`。
`TASK.md` 保留任务与其他参考材料，指向 `AGENTS.md` 中的规范；规范正文只出现一次。
运行中修改原始文件会触发现有输入绑定检查，不会静默更新已冻结任务。

管理整个实验的 Agent 也应在初始任务输入中读取同一规范。该 Agent 可在运行之外准备
参考、整理 Finding、修改代码并验证；冻结 Run 内的作者仍只写 candidate envelope。
这份规范没有创建后台管理进程，也没有让受限作者获得修改 Compiler 的工具。

## 低层代码怎样进入任务

规范要求分别判断 Cake 表达、lowering 实现与设备收益。冻结输入时，管理 Agent 应列明
作者实际能看到的材料层级、来源、目标和候选归属。生成源码反馈默认关闭；显式向
`tools/launch_task.py` 传入 `--generated-source-feedback`，或在管理输入 schema v2 的
相应 cell 设置 `"generated_source_feedback": true`，才会把 `generated_source_v1`
绑定到新 Run 的 authoring feedback。schema v1 和未开启的旧 Run 保持原行为。

此权限只允许 `open_cake` + `known_kernel_reproduction` 的作者检查自己该轮已封存且
完成搜索评测的候选 `lowered_source`。反馈逐候选保留原声明 stage 身份（独立 Schedule
为 null）、精确 target、lowering route 和 Compiler 声明的源码语言；route 的入口属于
源码，不证明 native binary 符号。它不交付 baseline/其他 Run 的实现，也不授予低层
authoring、读取任意 artifact 或额外工具的权限。独立回放从既有封存件及原作者程序
重新 lowering 验证对应关系，再重建下一轮反馈，无第二套源码存储或 history 副本。

原文按提交顺序、Program 声明 stage 顺序有界交付：UTF-8 正文每候选最多 32 KiB、
每轮最多 64 KiB；每候选最多 32 个 stage，含 metadata 的 JSON 视图最多 64 KiB。
总视图上界为该限制乘以 Run 冻结的最大候选数。保留的源码完整不截断，行号从 1
开始，已有 CAKE_OP 标记原样保留；缺失、未搜索或超界省略都有明确原因与计数。
只有下一次实际 provider 请求及其保留 bundle 才能证明投递；末轮结果或 provider fault
本身不证明作者收到了源码，更不证明模型使用了它。Run-local optimization history
仍只汇总观察；provider 自身会话历史可能保留先前请求。Metal 多 stage Program 仍不准入，
源码反馈不扩大任何目标的 executor 能力。

Metal 的几个层级不可混称：

| 层级 | Metal 对应物 | 任务中的用途与限制 |
| --- | --- | --- |
| 生成源码 | Metal Shading Language（MSL，`.metal`） | 对照 Cake 的 work mapping、访存、归约与算术；属于 CUDA C++ 一类的源码层 |
| 编译器中间表示 | Metal IR；离线流程可使用 `.air` 文件 | 在编译链位置上可类比 PTX，但不能据此假定有同等的文本 ISA、手写或检查接口 |
| 目标机器指令 | GPU-specific binary 中的 Apple GPU 指令 | 只有精确工具链实际提供且能解释的证据才能支持指令层结论；二进制容器不是可读反汇编 |

上述是层级类比，不是格式或能力等价。Apple 描述了
[MSL → Metal IR → GPU-specific binary](https://developer.apple.com/documentation/metal/metal-libraries)
的编译过程；其[离线工具说明](https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Dev-Technique/Dev-Technique.html)
说明 `.air` 保存 IR。NVIDIA 将 [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/)
定义为虚拟 ISA。不要将 MSL 命名为“Metal PTX”。

当前 Cake Metal builder 留存 `lowered_source` 和 `metal_binary_archive`，通过运行时
`MTLDevice.makeLibrary` 构建，没有向作者提供 AIR 或机器指令检查通道。
`Compiler.lower` 的 operation source map 仍是区域对应的 owner；当前交付保留原文行号和
CAKE_OP 标记，复用既有 candidate/artifact 身份并独立回放。缺失证据保持 unknown，
不能从 logical slots 或 timestamp profile 补出寄存器、spill、occupancy 或指令事实。
修改源码可见性属于 authoring treatment 变更，须绑定后继 Run；参考访问与工具权限仍适用。

## 结果与验证

结构对应、最小复现、差距归因、性能比较与 promotion disposition 写入既有实验结果，
运行产物保持在仓库外。Compiler 缺口使用既有 Finding 生命周期；修改提交后在隔离
worktree 验证，再用后继 Campaign 测量。无需新建 Study kind 或第二套证据账本。

规范是 authoring treatment。修改规范后启动新任务并保留旧输入，不重标历史结果。
被传入请求的内容可核对，但传入本身不证明模型遵循了它；结构、正确性和性能仍由
实际候选及外部评测证据判断。


## 多节点 Codex 状态目录

可在 cell 的 node 中显式绑定 `codex_home`，指向该节点上已准备好的绝对目录。
入口仅为该 cell 的 launcher 设置 `CODEX_HOME`；该目录影响 launcher 的默认凭据来源与
CLI 资源查找。采用 `isolated_auth_only_v1` 的新 Codex Run 另建私有
`actors/.codex-homes/<run-id>/`，按既有私密性规则只复制凭据。同一 Run 的 initial/resume
共享其私有 home，其他 Run 与资格验证使用各自的新 home；不复用 launcher 的个人技能或
会话目录。该策略的原实现与边界见 [ADR 0081](adr/0081-isolate-codex-author-home-per-run.md)。

这不是完整 skill 发现隔离的证明。auth-only 策略保留宿主 `HOME`，检查范围集中在私有
`CODEX_HOME`；Codex 还会从用户 `HOME/.agents/skills`、工作目录祖先、admin 与 system
位置发现技能，见[官方加载规则](https://learn.chatgpt.com/docs/build-skills#where-codex-loads-local-skills)。
任务文本快照也不等于原生 skill runtime。受控技能包需要另验：允许材料确实可见，宿主与
邻任务材料未进入实际 catalog，初始与续轮策略一致，并且脚本依赖及工具范围明确。
不向现有 auth-only home 填入 user skills/plugins 绕过其拒绝规则。

这里没有关闭 sandbox、修复历史环境或资格。新模型、effort、工具或作者环境需要相应的
两轮 qualification。旧行为及失败记录仍按固定提交保留；换绑定必须创建新输入和运行目录。

## 按任务准备完整原生技能包

`isolated_skill_package_v1` 为 Codex 的 known-kernel 作者准备私有 `HOME` 与完整技能材料。
当前只开放材料准备和可执行替身的 CPU qualification；原生技能发现、实际请求投递及
initial/resume 等价仍未获资格。真实 qualification、正式 Run/Campaign 和正式回放均拒绝
该策略，直接 Lab 入口或旧收据也不能放行。软件准备成功不代表已经能运行带技能的实验。

包是一个未压缩 tar，包含原生技能目录，例如：

```text
skills/
  cake-exploration/
    SKILL.md
    scripts/check.py
    references/metal.md
    assets/example.bin
```

每个顶层条目需要非空 UTF-8 `SKILL.md`；其 front matter 由原生 loader 解释，包检查不
另造 YAML 或依赖格式，也不证明 Codex 已接受这些技能。脚本、引用资料、二进制资源和
可执行意图完整保留。包只接纳普通文件与目录，拒绝链接、路径穿越、重复项、文件/目录
冲突及特殊条目。上限为 64 MiB tar、32 MiB 文件内容、8 MiB 单文件、2048 个条目及
32 个技能目录。源码与包放在各自授权位置，实验产物继续在 checkout 外。

schema v2 的 cell 可显式增加：

```json
{"author_skill_package": "/absolute/reviewed/author-skills.tar"}
```

`prepare` 先读取并检查所有所选材料，再把原包保存为该 cell 的 `author-skills.tar`。
本地和 SSH 传输均读取这个准备快照，在节点的 `run-inputs/` 中重建同一包，传给
`launch_task.py --author-skill-package`。源文件后续修改不进入已准备 cell；缺少自己的
快照就拒绝，不向管理根目录或其他 cell 回退。schema v1 不接纳此字段。当前传输接线可做
CPU 验证，真正启动仍在 launcher 的原生资格门前拒绝；`--preflight-only` 不是豁免。

Run 的 provider 声明持有唯一原包引用；TaskPackage、技能投影和保留证据使用同一份已读取
快照。正文只投递位置与“已准备、原生投递未验证”的元数据，不把脚本或二进制塞进提示。
软件资格保存原始 tar 到既有 evidence role `native_skill_package`，可据此重建完整投影；
这不开放正式 Run 的回放域。Provider 配置从已有包引用派生内容身份：相同包字节放在不同
Run 路径时仍是同一材料条件，内容变化则不能复用原资格；每个 Run 的可写状态始终独立。

私有 `HOME/.agents/skills` 接收只读投影，另一个私有 `CODEX_HOME` 保留原来的凭据与
会话生命周期。初次调用与 resume 核对同一绑定，内容、文件集合或权限漂移会拒绝继续。
依赖与工具说明仍由原生技能文档持有；投递文件不会安装依赖、启用插件或增加工具权限。
依赖是否可用、祖先/admin/system/plugin 来源是否受控、文件读取范围和模型是否真正使用
技能，仍须独立验收。私有 HOME 本身不能证明全文件系统的读取隔离。

开发者可在现有**可执行替身**资格命令上使用 `--fixture-only --author-skill-package
/absolute/reviewed/author-skills.tar`，同时提供既有 fixture 凭据与输出参数；包参数选择上述
新策略。省略 `--fixture-only` 会在读取凭据、启动 provider 或创建输出前拒绝。生成的
fixture 收据只证明其实际检查的软件行为，不能授权真实模型或 GPU 实验。

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

The local HIP/MACA correctness adapter is
`tools/qualify_tensor_program.py evaluate --built <build-directory> --case <case-id>
--output <new-result-directory>`. Invoke it once for every original Workload case, using a
new output directory each time. The process materializes one original CPU tensor case and
computes its original oracle before acquiring its Target's existing local broker allocation. Keeping
one case per process avoids holding all MoE weight cases in memory at once. Inputs cross
the device boundary as raw bytes and retain their declared tensor dtype; they are never
expanded into Python float lists. The existing Program loader owns ordered native dispatch,
intermediate storage, alias checks, module lifetime and stream identity.

Each result retains a common `EvaluationReceipt`, full observed/expected output bytes,
input-effect verdicts, the true native stage count, and JSON-null timing samples. IEEE
NaN/Inf output rules remain the Workload's own. An execution or teardown error prevents a
passing receipt. The worker holds its real allocation through process exit. All source,
software, host and review gates must pass before any device invocation.

The build/evaluate commands qualify construction or correctness only. A separate MACA
`profile` command retains every stage's native activity and checks both preflight and
instrumented outputs; its intervals and gaps are attribution observations, not latency
samples. HIP Program profiling and ordinary HIP/MACA optimization Run measurement remain
refused. No command here establishes a measured speedup.
The build command uses the shared `build_program_candidate` source/ABI handoff; it does
not construct an optimization environment whose measurement loop it cannot satisfy.
