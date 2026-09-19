---
name: add-hardware-target
description: 为 open-cake-ir 评估、规划或实现新硬件接入，区分现有厂商新型号（如 NVIDIA RTX 5090、Jetson Thor）、同厂商主机平台变化，以及 Ascend 等新增厂商；定位 Target、IR、后端、工具链、运行时和验证缺口。用于新增硬件支持，不用于单纯优化已支持芯片上的算子。
---

# 新硬件接入

目标是把一块**明确型号与主机环境**的硬件接到已有 Compiler–Lab–Evaluation 路径，并准确说明做到哪一级。先读项目当前 `AGENTS.md`、`CONTEXT-MAP.md` 和 [代码定位表](references/code-map.md)。从 `git rev-parse --show-toplevel` 确定工程根目录；下文工程路径均相对该目录。只做用户请求的评估或实现；调用 skill 不代表授权驱动安装、刷机或正式 GPU/provider Campaign。

## 先分流，再承诺工作量

记录五个独立维度：**Vendor、精确 Target、源码生成 Backend、CodeObject、Host/runtime**。`tasks/devices.py::BACKENDS` 是任务别名表，不是 Compiler 的 emitter 表。

| 场景 | 分流依据 | 阅读 |
|---|---|---|
| 现有厂商的新型号 | 已有 Vendor；已有编程模型与产物格式可能复用 | [现有厂商](references/existing-vendor.md) |
| 现有厂商但主机/部署不同 | 同上，但 CPU 架构、OS/SDK、驱动、分配器或计时环境不同；Thor 是典型检查案例 | 同一参考中的主机迁移部分 |
| 新增厂商 | 当前 Vendor 中没有；可能需要新 Backend、CodeObject 和运行平台 | [新增厂商](references/new-vendor.md) |

**5090 与 Thor 都是 NVIDIA 型号，不因型号不同增加 Vendor。Ascend 当前走新增厂商路径。** 同一厂商不保证所有层都可直接复用；新厂商也不必然增加 Backend——例如现有 Triton 已服务多个厂商。以当前枚举、注册项和真实工具链为准。

不要把市场名称、Compute Capability、Target ID、代码对象格式混为一件事。两个 SKU 即使共享 ISA，也可能有不同资源与测量环境；先检查当前精确目标识别是否能区分它们，再决定文档身份。仅缺少会改变实施路线的信息时询问用户：具体型号/变体、可访问主机、期望支持到哪一级。可以先完成静态接入分析。

## 支持分级

逐级报告 `已验证 / 未验证 / 不适用 / 阻塞` 和证据位置，不用一个“支持”覆盖所有阶段：

1. **声明与类型**：Target 能解析，字段有依据，未知能力按名拒绝。
2. **静态 lowering**：代表性 Schedule 通过验证并生成源码；负例由预期规则拒绝。
3. **离线构建**：实际工具链产物、目标标记、ABI、资源元数据经过核对。
4. **目标机执行**：本机准入、加载与正确性通过，外部 oracle 覆盖形状/分布/边界。
5. **可信测量**：声明 timer、测量区间、样本前 reset、质量检查和 profiler 归因；不能声明则报告测量覆盖不足。
6. **Lab/框架接入**：任务别名、Workload/Study 绑定、封存候选、回放与目标框架评测通过。

合成 Target 或第三 Adapter 的 CPU 测试只证明扩展机制，不能填入实际设备正确性或性能结果。现有新目标测试中 `sm_120a` 是合成 fixture，不是已验证的 RTX 5090 支持。

## 形成接入方案

- 先核对当前可复用部分：Target schema、指令合同、Backend 的 `CODE_OBJECTS`、平台/host/工具链注册项与任务接线。
- 给出最小的逐层差异表：**所需能力 → 已有负责者 → 修改位置 → 依据 → 正例/负例 → 当前支持级别**。区分正常登记、新实现、共享层扩展缺口，不声称“一份 JSON 即完成接入”。
- 硬件事实从对应版本的官方资料与目标机只读查询取得，记录来源和观测条件；不要复制相邻目标的限制、峰值、计时或指令集。按所选分流参考检查不同层的隐含假设。
- 如果模型不匹配，先指出缺失的执行/存储/同步语义及其验证义务。不能为满足解析器而编造 `warp_size`、CTA、共享内存或代码对象。
- 若用户要求实现，从最小端到端能力切片开始；新语法同时带类型、效果、合法性与分析。新增算子能组合现有原语时归 Workload，不扩大名字路由。

## 验证与交付

按现有命令和运行时操作；先从工程配置确认 Python，不安装一套测试环境来修复环境失败。

在实现提交的**独立 worktree** 中运行相关定向测试、完整 contract suite、Corpus Gate、源码快照与 STATUS 检查。参考命令（替换已核定的提交和测试目录）：

```sh
git worktree add --detach /chosen/test-worktree COMMIT
cd /chosen/test-worktree
PYTHONPATH=src python -m pytest tests/contracts -q -p no:cacheprovider
PYTHONPATH=src python -m open_cake_ir.cli compiler check-corpus --revision compiler/revision.json
PYTHONPATH=src python tools/check_corpus_sources.py
PYTHONPATH=src python tools/render_current_status.py --check
```

这些命令不授权正式设备实验。对新 Target，测试其可用能力、合法但不可 lowering 的输入、错误目标/产物、缺失能力与未建模事实。保持老目标的验收和发射结果；新 case 的期望和受影响旧 case 的期望须有独立审查依据后单独采纳，不能 refresh 制造通过。没有病例的目标继续报告未检查。

Host capture 在确认真实主机与已声明 Target 后，通过 `tools/capture_executor_host.py --help` 选择该版本支持的参数，按精确 Target 提交一次。不能从别的主机复制 capture，也不为了过测重采或修复历史 Evidence 的权限。环境失败按项目规则报告，不修补环境后把重跑解释成原记录有效。

保留当前语义：Schedule v2、精确目标、不降级、无布局代数、显式后端；Compiler 无 Lab/任务依赖；没有显式经验模型时保留候选顺序。不要恢复退役的 `Compiler.rank`、`calibration_coverage` 或 Portfolio Study 生命周期。历史数据在原固定提交回放；新 Campaign/Evidence/报告放 checkout 外。源码身份使用干净 commit；外部输入绑定仍走已有合同，不添加例行 hash 清单。

交付时简述分流、复用内容、实际改动/缺口、各支持级别、验证提交和日志。仅规划时明确哪些事实尚待目标机核实；仅静态通过时不要声称“设备已支持”。
