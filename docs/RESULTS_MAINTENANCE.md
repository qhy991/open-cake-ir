# 硬件分支维护数据，main 汇总展示

每个平台在自己的长期分支维护发布数据和平台页面。`main` 汇总已合入的四份数据，
不在线读取移动中的远端分支，也不另写一份最佳分数表。

分支仍然包含完整仓库；下面的目录划分表示维护归属，不要求从平台分支删除其他平台的数据。
通用分支与合入规则仍由[开发分支指南](DEVELOPMENT_BRANCHES.md)负责。

| 硬件 | 维护分支 | 发布数据 | 平台页面 |
|---|---|---|---|
| NVIDIA B200 / B300 | `nvidia` | [nvidia/records.json](results/nvidia/records.json) | [NVIDIA](results/nvidia/README.md) |
| Apple M1 Pro / M2 / M4 | `metal` | [metal/records.json](results/metal/records.json) | [Apple](results/metal/README.md) |
| Hygon DCU | `dcu` | [dcu/records.json](results/dcu/records.json) | [DCU](results/dcu/README.md) |
| AMD | `amd` | [amd/records.json](results/amd/records.json) | [AMD](results/amd/README.md) |

## 哪些文件可以改

`docs/results/<平台>/records.json` 是该平台的**发布投影**：记录要展示的观察、固定来源
和图表选项，不是新的实验账本或晋升 registry。原报告、Finding 与原始证据根仍决定结论。
历史的 `docs/results/b300-service-20260920.json` 保留为只读来源；新发布数据由 NVIDIA
目录维护，不再让共享生成器解析这份旧快照。

- `branch` 必须与目录同名；`platform`、`as_of`、说明和图表选项由该文件维护。
- 每条观察保留稳定的、以所属分支开头的 `id`。新增实验追加新观察；不要把旧观察改写成
  新提交的实验。旧错误的更正应说明原因并链接后继证据，不能静默替换测量。
- 保留设备、Workload、精度、基线、计时范围、正确性和测量状态。没有测量填 `null`，
  不填零，也不补出加速比；未合格结果仍可作为有标签的观察展示。
- `source` 指向仓库内的来源路径。文件级 `source_commit` 是历史来源的完整 Git 提交，
  新观察可使用自己的 `source_commit` 覆盖它。来源文件应先提交推送，再在发布行引用该提交。
  这与被测源码的 `version` 不同；两者都保留。
- 原产物、事件与收据位置写入 `locator`。新实验原始输出继续保存在仓库外。
- `figures` 只引用本文件内有有效加速比的观察 id。是否值得作为代表案例由平台维护者选择，
  不能把未合格结果、不同基线或不同形状拼成一条晋升曲线。

同目录的 `README.md`、`index.html` 和可选的 `overview.svg` 都是生成文件，不手工编辑。
共同的 schema、收集器、页面模板与生成器走 `task/core-* → main`；平台只改自己的数据目录。

## 平台更新

例如 NVIDIA：从最新 `origin/nvidia` 建立独立的 `task/nvidia-results-*` worktree，
收集或读取本次实验报告，更新 `docs/results/nvidia/records.json`。现有
`tools/read_task_results.py --runs /原实验目录` 可只读提取 TaskLab 报告；先导出到仓库外，
再核对并追加发布观察，它不提供新的验收或晋升。

```sh
python tools/render_hardware_results.py --platform nvidia --write --figures
python tools/render_hardware_results.py --platform nvidia --check --figures
```

`--platform` 只读取和生成对应目录，不更新其他平台页、根 README 或 main 总览。
将平台数据和生成文件一起提交，通过目标为 `nvidia` 的 PR 合入。其他平台将参数与分支
替换为 `metal`、`dcu`、`amd`。图表需要 Matplotlib 3.10.6；未指定 `--figures` 的读取和
Markdown/HTML 生成仅使用 Python 标准库。没有合格图表的 AMD 页面不会虚构一张图。

## main 汇总

平台更新验收后，从 `origin/main` 创建独立集成 worktree，合入要发布的平台提交，
检查完整差异，再生成汇总：

```sh
python tools/render_hardware_results.py --write --figures
python tools/render_hardware_results.py --check --figures
```

不传 `--platform` 时，从当前检出的四个 `records.json` 生成平台页、
[`docs/RESULTS.md`](RESULTS.md)、[`docs/results/index.html`](results/index.html)、总图和根 README。
生成器不修改输入数据，不申请 GPU、不重新审计或晋升，也不自动合入其他平台代码。
集成分支通过目标为 `main` 的 PR 合入；遵循保留祖先关系的 merge 流程。

平台分支若含尚未准备好的其他代码，先按开发分支指南处理其完整待合入差异，不能借数据
汇总把未验收代码带入 main。生成页面发生冲突时，先按归属合并输入记录，再重新生成；
不要选择某个平台的整份总览覆盖另一个平台的结果。

## 检查范围

`Hardware results` CI 在平台 PR / push 上检查对应平台目录，在 main PR / push 上检查
完整汇总。通用单元测试检查来源、缺失数据、跨平台输出隔离，以及读取报告时的配对基线。
数据新增后不会要求平台维护者提前更新 main 的总览；但 main 集成必须重建它。

初始拆分保留 80 条观察：NVIDIA 43、Apple 4、DCU 32、AMD 1。此处记录迁移范围，
不是滚动总数；当前数量由各平台发布文件生成。
