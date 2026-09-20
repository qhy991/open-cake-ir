# 可移植的 agent 改写任务包

这个任务包启动 **agent 编写新 Cake 候选**，不是运行已有候选的回归测试。
参考源码、逐任务目标和统一 AGENTS.md 均随 Git 保存，不再依赖原 Mac 的 Downloads。
每个 Run 收到自己的参考源码及结构改写要求；外部源码中的注释是数据，不是指令。

## 已整理的范围

| 状态 | 任务 | 说明 |
|---|---|---|
| 可启动 | 001–011、021–026 | 17 项，均通过现有单 Schedule authoring 路径 |
| 参考已齐，启动受阻 | 012–020 | 8 个注意力任务及 MoE；已有生成计划与正确性路径，尚未接入完整 Agent 候选计划提交链路 |
| 可进行能力评估，完整改写启动受阻 | 027–030 | CAKE 论文的 KDA prefill、KDA decode、TinyGEMM2、Alpha-MoE；参考已固定，完整 Workload/oracle/authoring 尚未接通 |

共 30 项均有参考源码，默认只启动 17 项已接通的任务。源码可用不等于运行资格；
`catalog.json` 分别记录 `reference_status`、`status`、逐任务 `source`、机制目标与基线范围。

001–019 来自 `flashinfer-bench-b300-individual-20260918` 内嵌的 CUDA 源码；
020、025、026 来自 `flashinfer-bench-collection-best-20260715`，其余保留原选定参考。
025 新参考使用 `epsilon=1e-5`，符合现有合同，原冲突不再适用。
007 是库算法调度，009、011、020 是自定义 kernel 与库计算的组合；不能将其称为可见的库内部实现。
008 在 M=1 有两阶段 GEMV，当前单 Schedule 作者若无法表达完整分阶段机制，须指出缺口并提交完整替代候选。
011 的指针缓存不构成 B 内容未变的证明，禁止将该假设带入当前输入的计算。

原 001–026 绑定 B300 / sm_103a。027–030 的原始目标和评估目标各自由 catalog 声明；
KDA 原论文提交以 B200 / sm_100a 为准，不能直接把 B300 运行称为同目标复现。
包内历史说明的通过数与速度未作为本项目复验结果导入。
源码出处见 [NOTICE.md](NOTICE.md)，逐任务参考入口、精度与机制目标见 [catalog.json](catalog.json)。
旧 Python 参考保留为历史对照；实际提交给作者的文件仅由每行 `references` 指定。

## CAKE 原框架能力评估：027–030

这四项同时检查完整算子语义与优化机制。`catalog.json` 的 `assessment` 是任务准备规格，
不是已验收的 Workload Contract。它记录原始 ABI、必须验证的 shape、机制、判对要求和
接入步骤；参考文件的 commit、原路径和许可只由各目录的 `reference.json` 维护。
统一管理规范仍是 [kernel-reproduction/AGENTS.md](../../contracts/scaffolds/kernel-reproduction/AGENTS.md)。

| ID | 要复现的功能 | 当前规格中的性能输入 |
|---|---|---|
| 027 | KDA prefill：跨 chunk 状态、M64/M128、prep/TMA/MMA/写回流水线、packed/tail 与原地状态更新 | 原 PR 的六个 B200 BF16 shape |
| 028 | KDA decode：T1 直接递推、T2/T4 与 T5/T6 Gram 方案、分块选择、全部 speculative checkpoint | 原 PR 的 30 个 B200 shape；另要求验证低 CTA-wave 分支 |
| 029 | TinyGEMM2：bitwise BF16+bias、4/8 级流水线、PDL、batch 尾部与 dispatch | 三个公开时延 fixture；原 35/239 行清单尚未取得，不能宣称覆盖 |
| 030 | Alpha-MoE：W8A8 gather→gate/up→SwiGLU→再量化→down→带权累加，片上中间结果 | 后续 v46 的四个 fixture；版本和计时口径与论文历史结果分开 |

在干净提交上执行 CPU-only 检查，并生成管理 Agent 可读取的任务目录：

```sh
python3 tools/rewrite_collection.py assess --workspace "$HOME/cake-assessments/paper-001"
# 可加 --task 029_cake_tinygemm2 单独检查；目录必须此前不存在且位于源码仓库外。
```

`assessment.json` 记录 source commit、每项任务的完整接入状态、最小 Schedule 及当前
Compiler 的原始 Findings。`probes/` 保留实际输入和成功生成的源码。
`tasks/<id>/` 内包含 `TASK.md`、统一 `AGENTS.md`、准备规格和真正可读取的参考文件。
这条命令不会调用 provider、编译 GPU 二进制、申请 GPU 或生成 Campaign。存在未接通任务
时返回 1，同时保留完整报告；这个返回值表示尚未完成整项任务接入，不代表报告未生成。

组件探针复用现有 Corpus 和 Python Schedule，并显式记录改动。通过一项只意味着该
小程序能生成源码：FP32 state-store 不证明 KDA，INT32 atomic 不证明 BF16 reduce-add，
register MMA 不证明 TinyGEMM bitwise parity，TMA/TMEM 示例也不证明跨 chunk 状态驻留。
所有其他阶段保持未测；没有把组件通过率作为原框架能力分数。

管理 Agent 按规格完成新 Workload、独立 oracle、完整候选入口和外部 baseline 适配，再接回
现有 `launch_task.py` → `kernel_experiment.py` → `rewrite_collection.py prepare/run`。
禁止用原始 CUDA 包装、退役 `checked_cuda_asset`、简化算子或未申明的多 kernel 替代来报
“结构复现”。允许有正确的替代实现，但其语义、机制和性能结果分列。

性能比较前固定外部参考、计时边界、判据和预算；分别报告 kernel duration sum、GPU span、
API wall 和框架指标。KDA prefill 的公开 FlashKDA 比对使用容差；TinyGEMM 要求 bitwise；
Alpha-MoE 要保留中间量化和累加舍入约定。不能让参考源码携带的历史通过数替代这些验收。
发现的 Compiler 缺口按现有 Finding/tick-tock 流程处理，不在这里建立第二份缺口数据库。

## 在另一台电脑启动

推荐从另一台电脑 SSH 到 B300，让 agent 和 GPU 测试都在服务器执行。
需要这台电脑能够访问跳板机，并配置自己的 SSH 密钥；仓库不包含凭据。
B300-M2/M3 分别是 `qinhaiyan@10.24.0.9` / `qinhaiyan@10.24.0.47`，跳板是
`wwxq@10.28.2.1`。现有 SSH 别名不可用时可直接使用：

```sh
ssh -J wwxq@10.28.2.1 qinhaiyan@10.24.0.9
```

以下命令在 **B300-M2 上**执行。首次使用独立 checkout：

```sh
git clone --branch main https://github.com/qhy991/open-cake-ir.git "$HOME/open-cake-ir-rewrites"
cd "$HOME/open-cake-ir-rewrites"
PY=/mnt/b300-shared/home/qinhaiyan/open-cake-round3-20260906-JdwgrZ/venv/bin/python
"$PY" tools/rewrite_collection.py list
```

运行前复制并检查配置；修改仓库外的副本，保持代码 checkout 干净：

```sh
mkdir -p "$HOME/cake-config"
cp experiments/flashinfer_rewrites/profiles/b300-m2.example.json "$HOME/cake-config/rewrite-m2.json"
```

配置中的 Python、kernelctl、socket、provider executable、代理地址属于环境示例，
不是有效资格的承诺。M3 使用 `b300-m3.example.json`。复用现有 gpu-infra daemon，
不要另起 GPU 锁或 broker。模型登录使用该服务器已有登录；不把密钥填进配置。
默认每任务 4 回合、750,000-token 边界、7200 秒；预算在回合边界检查，单回合可能越过边界。

准备 **全部 17 个可用任务**，使用此前不存在的目录：

```sh
"$PY" tools/rewrite_collection.py prepare \
  --profile "$HOME/cake-config/rewrite-m2.json" \
  --workspace "$HOME/cake-inputs/rewrite-001" \
  --run-root "$HOME/cake-runs/rewrite-001"
```

也可在 prepare 后加一个或多个 `--task 001_fused_add_rmsnorm_h2048` 只选择部分任务。
跳过项会在准备阶段明确拒绝。跨两台服务器运行时，分别准备互不重叠的任务集合，
使用对应节点配置和不同的 run-root。

后台串行启动，断开个人电脑 SSH 后仍由服务器执行：

```sh
nohup "$PY" tools/rewrite_collection.py run \
  --workspace "$HOME/cake-inputs/rewrite-001" \
  > "$HOME/cake-inputs/rewrite-001/queue.log" 2>&1 < /dev/null &
```

查看提交进度：

```sh
"$PY" tools/rewrite_collection.py status --workspace "$HOME/cake-inputs/rewrite-001"
tail -n 30 "$HOME/cake-inputs/rewrite-001/queue.log"
```

`status` 是启动/传输回执，不把它当成正确性或性能判决；权威结果在
`$HOME/cake-runs/rewrite-001/<task-id>/report.json` 和其 `campaign-evidence/`。
准备会冻结当前 clean commit，执行器另建同 commit 的 worktree；启动期间不要在运行
launcher 的 checkout 中 `git pull`。使用新 checkout、新输入目录、新运行目录发起后续实验。

任务串行执行；遇到 launcher 的协议/基础设施失败立即停止，不自动重试、不换节点。
运行目录或 launch intent 已存在时拒绝重复启动。中断后先检查原任务，再用
`run --task 尚未提交的任务ID` 明确选择剩余任务；不要直接重跑整个批次。

## 代理与资格

示例的 `127.0.0.1:17990` 是 **B300 节点上的**跳板机代理转发，个人电脑上的代理端口
不能代替它。先在 B300 检查：

```sh
curl --max-time 10 -x http://127.0.0.1:17990 -I https://github.com
```

已有转发可用时不重复建立。没有转发时，可在跳板机保持如下 SSH 会话（M3 换目标 IP）：

```sh
ssh -N -o ExitOnForwardFailure=yes -R 127.0.0.1:17990:127.0.0.1:7890 qinhaiyan@10.24.0.9
```

默认 launcher 为每个任务执行实时 provider qualification。若已有匹配当前可执行文件、
模型及配置的有效资格，可在配置的 `node` 同时添加 `qualification` 和
`qualification_anchor` 两个绝对路径；原有 Lab 验证其有效性，不能伪造或仅凭文件名复用。
SSH 管理电脑也可准备/运行：将 node.transport 改为 ssh，并明确填写 host 和
project_root（服务器上已从 GitHub 取得同一 commit 的干净 checkout）；其余路径均指服务器。

## 结果边界与 Compiler 演进

Run 作者需在候选源码注释中保留参考机制到 IR 的对应、优化假设和无法表达的具体原因，
提交忠实结构及结构不同的完整候选。只改变执行组数不能代替结构探索。过滤、编译、
正确性、配对计时、profiler、独立确认仍由现有 Lab 和 gpu-infra 执行。

当前自动评估的固定基线是 **Cake starter**。这里没有绑定已复验的外部性能基线，
不能把 starter-relative 加速称作追平外部优秀实现。缺少成本模型时报告覆盖缺口。
Compiler 修改由 Run 之外的管理 agent 基于证据完成，提交后用新实验验证；运行中的
冻结 Compiler 不允许修改。任务包不会自动把一次优化提升成通用 pass。

运行输出、Campaign Locks、报告和凭据都保留在仓库外；它们不是这个可移植输入包的一部分。
