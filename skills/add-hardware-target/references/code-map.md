# 接入点与现状核对

以下是仓库路径和负责符号，不是另一份配置。每次先用 `rg` 确认它们在当前提交仍成立；缺失时查改名后的负责者，不重新创建旧模块。本表按后端解耦重构后的代码组织。

| 事项 | 首先查看 |
|---|---|
| 精确型号与事实 schema | `compiler/targets/`；`src/open_cake_ir/compiler/target.py` 的 Vendor、CodeObject、Target、ResourceLimits |
| Target 加载与合同闭合 | `compiler/revision.py` 的 load_revision、_load_target、_declared_contracts（下列简写路径相对 src/open_cake_ir） |
| 指令含义 | `compiler/ir/instruction_contracts.py` 的 CONTRACTS、InstructionContract |
| 操作/存储词汇与解析 | `compiler/ir/vocabulary.py`、`schedule.py`、`resources.py`、`operations.py` |
| 四类验证 | `compiler/verifier/`；不要把新设备差异硬塞进 Target-less 数据一致性规则 |
| emitter 协议与登记 | `compiler/backends/__init__.py` 的 BackendModule、Backend、BACKENDS；各 backend 的 CODE_OBJECTS |
| 命名与 raw-input 合同 | `compiler/backends/common.py::PythonNamespace`；各后端声明；Backend.validate_input |
| 离线路由与产物检查 | `compiler/backends/triton.py::target_route_facts`；`compiler/toolchain.py::CodeObjectRoute`、_CODE_OBJECT_ROUTES、triton_route；对应非 Triton 工具链 |
| 平台声明 | `evaluation/platforms.py::ExecutionPlatform`、PLATFORMS |
| 实际加载/启动 | `evaluation/core.py` 的 loader 分派与相应 driver；cuda_driver.py、hip_driver.py、metal_runtime.py；不止平台行 |
| 主机准入 | `lab/executor.py::_HOST_VALIDATORS`、HostKind、ExecutorRevision；`tools/capture_executor_host.py`；`runtime/hosts/` |
| 任务支持和设备分配 | `tasks/devices.py::BACKENDS`、allocation_mode、timing_source；`tasks/runtime.py` 的两项设备准入检查 |
| Lab 工具链绑定 | `lab/toolchains.py::TOOLCHAINS`、`lab/runtime_config.py`、各 build worker |
| 原生比较环境（需要时） | `lab/pairing.py::NativeBackend`、NativeAdapter、_NATIVE_BACKENDS；native_triton_adapter.py、native_cute_adapter.py |
| 任务语义和输入 | `tasks/` 的具体 workload/authoring/evaluation；`contracts/workloads/` 与当前 Study 模板 |
| 公共状态 | `reports/current/STATUS.md` 由工具生成，不能手写支持数字 |

## 可复用的验证入口

- `tests/contracts/test_target_documents.py`：同厂商合成 Target 的静态扩展、缺失/未知字段。
- `test_compiler_revision_architecture.py`、`test_instruction_contracts.py`：声明集合与指令 typing 的受审查快照。
- `test_backend_boundaries.py`、`test_backend_isolation.py`：发射能力、输入/命名策略隔离。
- `test_native_adapter_dispatch.py`：第三 native Adapter；仅说明这个接口可扩展，不代表第三厂商设备已支持。
- `test_vendor_neutrality.py`、`test_declared_warp_size.py`、`test_declared_code_object.py`：无厂商默认值、宽度/对象归属、离线路由。
- `test_platform_registry.py`、`test_execution_platform.py`、`test_study_admission_gates.py`：平台闭合与真实入口仍调用计时/分配检查。
- `test_corpus_target_coverage.py`：缺少 case 的目标必须被报告。
- `test_task_boundaries.py`：通用层不导入具体任务。

测试路径除首项外相对 `tests/contracts/`。结合实际改动选择正例和拒绝反例；不能只确认“被拒绝”，还要确认是哪条规则拒绝。

## 当前结论应怎样表述

同厂商已有代码对象：Compiler 的文档扩展已被合成测试验证，真实工具链/主机仍需核对。
同厂商新主机：复用编译路线不消除 host、SDK、分配和计时工作。
新增厂商：接口可扩展，但当前 CodeObject/Target 资源模型和运行平台不是任意硬件插件接口；先审查表达能力，再逐层实现。

不要把这些静态结论升级为 5090、Thor 或 Ascend 的实际支持记录。
