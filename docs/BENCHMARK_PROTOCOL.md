# Compiler 开发与独立硬件 Bench

[共同研究主题](RESEARCH_AGENDA.md) · [分支流程](DEVELOPMENT_BRANCHES.md) · [结果发布](RESULTS_MAINTENANCE.md)

本页是 main、metax、metal、dcu 的共同工作标准；其他硬件分支采用同一职责边界。
Compiler 开发回答“Cake 需要什么能力”，独立 Bench 回答“固定版本的 Cake 能把任务
完成到什么程度”。Bench 可以包含公开开发题；独立仓库本身不授予 held-out 身份。

## 任务和代码的归属

| 内容 | 归属 | 验证要求 |
|---|---|---|
| IR、typing、Verifier、改写与 lowering | open-cake-ir Compiler | P1–P8、反例、完整 Corpus Gate、独立评审 |
| 能力探索、最小复现、开发 Workload | open-cake-ir 任务与 Findings | 每题独立外部目录，保留精确语义和失败 |
| 题目、输入生成、oracle、容差、强参考、计分与计时 | 各硬件独立 Bench | 不依赖 Cake 的 Verifier、oracle、Lab 或 provider 才能判定结果 |
| 作者模型、材料、权限、停止规则与研究分组 | 现有 Run/Study | 固定作者环境，保持已有执行与证据所有者 |
| README 的进步记录 | 平台发布投影 | 引用封存报告，绑定 Compiler 和 Bench 两个版本 |

| 维护分支 | 独立 Bench | 当前范围与进步入口 |
|---|---|---|
| metax | [c550-bench](https://github.com/qhy991/c550-bench) | [C550 进步记录](results/metax/BENCHMARK_PROGRESS.md) |
| metal | [metal-bench](https://github.com/qhy991/metal-bench) | [Metal 进步记录](results/metal/BENCHMARK_PROGRESS.md) |
| dcu | [bw1100-bench](https://github.com/qhy991/bw1100-bench) | [DCU 进步记录](results/dcu/BENCHMARK_PROGRESS.md) |

硬件独立 Bench 共享能适用的题目与契约，同时保留精确设备、工具链、ABI 和计时差异。
不要求同名任务强行使用同一 dtype、缩小原始形状或退到其他设备。unsupported 是结果。
数据来源和许可沿用各 Bench 的锁文件；本仓库不复制受限数据或 reference。

## 一次演进周期

1. 从 main 的任务选开发集，固定目标和 Compiler commit，为每题建立独立工作目录。
   Metal 可使用 [任务准备入口](METAL_DEVELOPMENT_TASKS.md)。移植保留任务身份、数学语义、
   dtype 和 oracle；显式改变开发形状时保留原形状的拒绝。
2. 冻结 Compiler 后执行 kernel 搜索：产生结构不同的 Cake 候选，先过 construction、
   verifier 和适用的 cost 排序，再做独立正确性、无 profiler 计时和 profiler 诊断。
   cost 未校准时报告覆盖缺口，不能借其他硬件的估计或跳过硬门。
3. 每题保留最小复现、原始诊断、自己的生成源码、负结果、资源/指令观察及建议。
   将问题归到 candidate、Verifier、cost model 或 IR vocabulary。作者不必制造 Compiler
   缺陷；“现有能力足够、候选选择不好”是有效结论。重复证据才支持通用规则。
4. 结束该冻结搜索，在独立开发任务中审查建议。原语与分析一起修改，运行完整 Corpus
   Gate，接受独立评审，再发布后继 commit；Bench 和旧 Run 的字节及判定保持原版本。
5. 在固定 Bench commit 上，以发布后的 Compiler 构建候选产物，交给 Bench 的统一接口。
   正确性先于计时；仅通过部分 case 就只报告部分覆盖。完整框架端到端性能与组件
   microbenchmark 分别记录。缺少计时资格的 Bench 先补独立测量验收，再报告性能。
6. 追加平台进步记录，经平台 PR 进入 main；把 main 合回平台后开始下一周期。

作者应尽量使用 Cake 探索机制，并结合自己的低层产物检查 lowering。NVIDIA 的 PTX/SASS、
Hygon/AMD 的 LLVM IR/HSACO/ISA、MetaX 的目标产物和 Apple 的 MSL/AIR/metallib 分别注明
层次与可获得性。MSL 是生成源码，不能当作 Apple 机器指令统计。clean-start 不得读取
目标原生参考；已知实现复现才按声明权限读取。独立目录或 HOME 不构成读取隔离。

## 分开检验两个问题

- **Compiler 演进收益**：固定 Bench、强参考和作者设置，在新旧 Compiler 上做全新搜索，
  并用固定候选 replay 辅助区分诊断、生成与作者选择的变化。修复用例通过不足以证明
  后续搜索改善；已经参与能力或参数选择的题记录为开发暴露。
- **Cake 的作者环境优势**：固定 Compiler，比 Cake 与原生作者环境；匹配模型、材料、
  参考权限、时间/工具预算和共同评测。仅一个 Cake 候选较快不能回答这个问题。

每种比较固定任务集合和停止规则，保留全部已分配 Run 与失败。新工程 Run 默认三小时
（含确认），token 只记账，不设总量/每轮限额，不因 token 用量排除结果。历史冻结 Run
保留原预算和终点；预注册研究的改变需要新的 Study/Run，不能事后改旧判定。

## README 怎样展示进步

每个平台维护 `docs/results/<platform>/BENCHMARK_PROGRESS.md`，首页链接对应平台记录。
它是封存证据的发布说明，不替代 Bench 报告、Run 审计或已有 records.json 的事实所有者。
三个分支同步同一标准和共享实现；更新平台自己的记录，然后通过 main 共享，避免分别
维护同一公共改动。现有生成结果区继续由其生成器维护。

每条新性能记录至少声明：日期、精确设备/工具链、Compiler commit、Bench commit 与
任务/workload、作者模型/设置与参考权限、固定起点/强参考的版本、正确性覆盖、计时器、
测量区间和 device-state reset、原始样本/失败报告位置、候选与参考的同口径时间。
用“前版 → 后版”表展示有效逐任务结果，同时保留变慢、未完成和不支持；未知值用
`未测`，不能填零。强参考差距和相对起点收益分开，不把组件 replay 写成版本演进收益。

任务集合、Bench 版本、参考或测量协议改变时开启新的比较段；只有同口径才能画连续
进步曲线。汇总先声明任务分母、正确性与可测覆盖、独立 Run 数和统计方法，不能删掉
失败任务后只展示成功几何平均。CPU 检查、生成源码、原生编译各自不授予设备性能。

各目标保留独立计时资格；目前 Metal 工程口径是 30 个样本、每样本 64 dispatch 后
归一化取算术平均。它不替换 MetaX/DCU 的现有协议，也不回写历史计时结果。计时重复
衡量测量波动，独立作者 Run 衡量搜索波动。跨硬件的速度比不组成统一排名。
