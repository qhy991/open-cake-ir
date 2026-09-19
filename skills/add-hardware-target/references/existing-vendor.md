# 现有厂商的新型号，以及主机迁移

## 编译器侧：从文档扩展开始

1. 核对 `Vendor` 已有该厂商，明确精确设备变体、ISA 与工具链目标字符串。保留既有 Vendor；不要按产品型号增加 backend。
2. 在 `compiler/targets/` 准备有来源的 Target：组宽度、内存空间、操作集合、指令/同步合同、资源限制。`occupancy` 与 `peak` 有依据才声明；没有 tensor 空间就不写 tensor 上限。未知的必填事实是阻塞，不能填猜测值。
3. 复用指令合同必须是相同语义与数值承诺；新合同在 `compiler/ir/instruction_contracts.py` 定义其类型/效果，Target 仅引用名字。后端拥有发射拼写和 capability Findings。
4. 检查 emitter 的 `CODE_OBJECTS` 和输入限制。Python 后端的保留名称归自己的 `PythonNamespace`；额外 raw-input 限制通过注册项 `validate_input`，不要把型号判断写回 `core.py`。
5. 路由事实随 Emission requirements 进入离线编译，不让 jail 读取 Target 文档。核对实际编译器支持目标标记与所用指令。
6. 明确更新声明目标 pin、合同快照和实际覆盖用例。`test_target_documents.py::SameVendorTargetIsADocumentTest` 里的 fixture 必须保持合成性质；若真实 Target 使用同一 ID，调整 fixture 隔离方式或使用另一个合成 ID，不能把它的假数据直接提升成真机文档。

这些步骤可复用已有 emitter 的前提是新设备确实实现相同能力。CuTe 的 `_TMEM_ROUTE_EVIDENCE`、`REGISTER_ROUTE_EVIDENCE` 以及 warp-specialization evidence 是资格集合；不能因“同属 Blackwell”就加入一个名字。

## 型号与 Target ID 不等价

当前 CUDA 路由通过 compute capability 传递编译架构，运行时还存在从 capability 形成 `sm_…a` 身份的约定（`evaluation/cuda_driver.py` 的设备目标属性）。必须核对工具链支持的准确后缀和设备识别，而不是从产品名拼字符串。

同一 CC 的不同 SKU 若有不同资源事实，不能只扩大一个 Target 的 `device_names` 让它们共享不真实的 occupancy。若需要 SKU 级 Target ID，先检查编译架构与逻辑身份、host capture、加载时设备识别是否能分别表达；当前路径需要调整时，报告为身份/路由缺口并配负例。不要静默用 ISA 名代替精确设备合同。

## RTX 5090：现有 NVIDIA 路径

候选复用路径是已有 CUDA 代码对象、Triton/适用的 CUDA emitter、CUDA 加载器。需要实际确认：

- CUDA/驱动/Triton 版本是否支持该 ISA，产物的目标声明是否匹配；不能借 B200 的 tcgen05、tensor memory 或 warpgroup 能力。
- 工作站是否通过 `local_broker` 使用 `local_serialized`，而不是继承 B200 的集群 allocation；选择有现场依据的分配方式。
- 所需 dtype/数值合同、内核参数、设备内存与资源上限。
- 对应平台的计时、缓存 reset、Profiler 与实际主机包是否可用。已存在 CUDA 平台行不代表这台主机已经准入。

不要在本 skill 内固定一份硬件参数表。官方 [CUDA GPU 表](https://developer.nvidia.com/cuda/gpus) 当前将 RTX 5090 列为 CC 12.0；执行接入时仍须核对选定产品和 SDK 文档，不据此直接填写 Target ID 或其他字段。

## Thor：同厂商，也需要核对主机环境

先明确用户指 Jetson AGX Thor 的哪个模块/变体。它属于 NVIDIA；重点增加 **JetPack/ARM 主机与部署适配检查**，不是新增厂商。

- 查实际 CPU 架构、OS、JetPack/BSP、CUDA 和 Python/框架 wheel；独立核对 ARM 原生或交叉编译路径。
- 检查 jail 的 Python、runtime roots、动态库路径与 SDK 架构；不沿用 x86 主机 capture。
- 当前 CUDA 准入使用 `/usr/bin/nvidia-smi` 与 `torch.cuda`，CUDA host schema 还有 CUPTI/FlashInfer helper 等既有要求。确认真实可用性，无法满足时报具体依赖缺口；不要删除检查或填假 capture。
- 不假定“Jetson 没有 nvidia-smi”：NVIDIA 的当前文档已列出 Thor 支持；具体工具、版本和字段仍需现场确认。
- 核对集成设备的内存、一致性、reset、功耗/频率状态和计时语义，不继承离散 B200 的测量结论。
- 官方 CC 表目前将 Jetson T5000/T4000 列为 11.0，与 RTX 5090 不同。SDK 对架构标识的历史变化必须按选定版本核实。

依据：[Jetson Thor JetPack 指南](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/0.1.0/setup_jetpack.html)、[CUDA for Tegra](https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/)、[nvidia-smi 文档](https://docs.nvidia.com/deploy/nvidia-smi/)。这些是查询入口，不是安装授权；使用与目标机版本匹配的资料。

## 从可生成到可运行

- 已有 CodeObject 通常复用 `evaluation/platforms.py` 的平台类型，不复制一个每型号平台。新增主机/分配机制的实际差异归平台或 host 负责者。
- `tasks/devices.py::BACKENDS` 仍需要任务别名登记：target、route、device_name、allocation、timing_source 和任务所用合同。名称叫 BACKENDS，但它不是 emitter 清单。
- 在真机使用 capture 工具生成 `runtime/hosts/<target>.json`；先确认当前 schema 与工具支持的 kind/参数，历史 CUDA 无 kind 形式是显式兼容形式，不要手写一个未经支持的 kind。
- 选择少量代表性原语与外部 oracle，做端到端正确性、边界/尾部、计时质量和误归因负例；CPU 编译测试不能替代这一级。
