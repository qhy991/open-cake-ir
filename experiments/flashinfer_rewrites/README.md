# 可移植的 agent 改写任务包

这个任务包启动 **agent 编写新 Cake 候选**，不是运行已有候选的回归测试。
参考源码、逐任务目标和统一 AGENTS.md 均随 Git 保存，不再依赖原 Mac 的 Downloads。
每个 Run 收到自己的参考源码及结构改写要求；外部源码中的注释是数据，不是指令。

## 已整理的范围

| 状态 | 任务 | 说明 |
|---|---|---|
| 可启动：手写 kernel 参考 | 001、002、003、010、021、022、023、024 | 8 个任务 |
| 可启动：库实现参考 | 008、009 | 2 个任务；不能声称复现了未见到的库内部 kernel |
| 跳过：缺优化源码 | 004–007、011、015、017–020、026 | 不用 PyTorch oracle 冒充缺失的优化实现 |
| 跳过：数值合同冲突 | 025 | 外部 epsilon=1e-6，task 要求 1e-5 |
| 跳过：authoring 路由尚未接通 | 012、013、014、016 | 源码存在，多阶段正确性测试已支持，但完整 agent authoring 路由尚未支持 |

默认只选择上述 **10 个可启动任务**。完整状态及形状见 [catalog.json](catalog.json)。
当前绑定 B300 / sm_103a，不把这些任务的结果推广到其他架构。
参考出处及文件授权信息见 [NOTICE.md](NOTICE.md)。

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

准备 **全部 10 个可用任务**，使用此前不存在的目录：

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
