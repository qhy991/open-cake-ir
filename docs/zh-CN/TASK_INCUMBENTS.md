# 让下一轮使用每个任务已验证的最佳实现

Task incumbent 让下一轮工程 Run 与同一精确实验单元中已经验证的
最佳实现竞争。它不替换科学实验长期固定的 reference baseline。

## 晋升结果

晋升命令读取原始、具备 Filesystem Custody 的工程 Run，完成审计后，在外部
incumbent registry 中追加封存的晋升记录：

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python tools/promote_task_incumbent.py \
  --project-root /绝对路径/open-cake-ir \
  --registry-root /绝对外部路径/task-incumbents \
  --run /绝对路径/task/run.json \
  --evidence-root /绝对路径/task/run-evidence
```

旧 artifact-optimization Campaign 使用 `--campaign-lock` 和原证据目录。两种输入共用
同一个收益检查和追加式 registry；原记录保持不变。Study 分配的 Run 不能借工程入口绕过
其研究晋升策略。

候选必须通过外部 oracle 正确性、确认测量质量和冻结 Evaluation Protocol 的实质收益要求。
Git 复制品没有 Filesystem Custody，不能晋升。第一次晋升后，新 Run 的 fixed
baseline 必须与当前 incumbent 完全相同，旁支实验不能覆盖更强结果。
历史 Campaign 使用它自己绑定的 Executor Python 与源码完成语义重放，不拿今天的
TaskPackage renderer 去冒充当时的投影。

## 下一轮或矩阵使用 incumbent

单任务和矩阵分别增加：

```sh
python tools/launch_task.py ... --incumbent-registry /绝对外部路径/task-incumbents
python tools/launch_task_matrix.py ... --incumbent-registry /绝对外部路径/task-incumbents
```

每个任务用 Workload identity、case、精确 target、backend 和完整 Evaluation
Protocol 解析自己的 cell。完全匹配时，incumbent 成为 Run 的固定黑盒
baseline；registry 尚不存在或没有匹配项时，明确回退到 starter reference，
第一次 material confirmation 随后可以通过晋升命令创建 generation 0。工作区的
`baseline-selection.json` 记录解析来源，`run.json` 固定真正执行的 candidate identity 与
bundle 路径。当前 registry 的 Program 或原生 kernel 使用自身已审计的执行合同；
starter/reference 基线继续接受原来的源码与 launch 比对。

## 不会发生的事情

- 只有 search 优势、测量不稳定、变慢或 close-null 的结果不会晋升。
- 一个 shape、target 或测量协议的冠军不会被另一个单元借用。
- clean-start author 看不到 incumbent 的底层实现。
- scientific matched Study 不会静默采用滚动 baseline。
- registry 没有可修改的最佳分数表；current 由 sealed promotion Runs 重放得到。
