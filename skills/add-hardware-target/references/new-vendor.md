# 新增厂商：以 Ascend 为接入分析案例

当前 `Vendor` 没有 Ascend，`CodeObject` 只有 cubin、Metal archive、hsaco，当前 emitter 也没有已登记的 Ascend 实现。因此只能说**架构有扩展位置，尚未提供 Ascend 端到端支持**。代码新增完成与设备验证通过分开报告。

## 先审查模型能否忠实表达硬件

按具体型号、执行单元和 SDK 版本核对：任务分发单位、并发组、存储层级、同步、数据搬运、向量/矩阵指令、参数 ABI 与资源上限。对于 Ascend，要从其实际 AI Core 编程模型出发；不把 AI Core 的块调度直接等同于 CUDA warp/CTA。

当前 `Target.from_dict` 要求正的 `warp_size`；`ResourceLimits` 要求 CTA 线程上限、共享内存容量和三维 grid。`MemorySpace`、`OperationKind`、`DType` 等也有封闭词汇。逐项证明存在真实映射；若没有，先提出具体的 IR/Target 模型扩展及同一变化中的类型、效果、验证、资源分析和拒绝规则。不能填 32、64、0 或臆造的 shared/tensor 能力让 JSON 通过，也不能引入布局代数来隐藏差异。

从一个真实可表达的算子切片开始，明确不支持的能力；如果连最小切片都无法诚实映射，交付阻塞分析，而不是标记接入成功。

## 各层要增加什么

| 负责者 | 必须解决的问题 |
|---|---|
| Vendor / Target | 明确厂商枚举成员与准确设备合同；只声明有依据的事实 |
| CodeObject | 根据真实编译产物、ABI、加载器决定是否新增；不能因文件像 ELF 就冒称 hsaco/cubin |
| IR / 指令合同 | 若现有操作无法表达，补最小语义与分析；指令 typing 留在合同注册表 |
| Backend | 确认已有 emitter 是否真实支持；否则实现 requirements、preflight、emit、CODE_OBJECTS 并登记 LoweringBackend/BACKENDS |
| 工具链 | 可复现构建、版本与依赖、隔离、输出验证、真实 launch ABI；不默认用现有 Triton/CuTe worker |
| ExecutionPlatform | 新代码对象对应的 artifact roles、加载/运行、计时和归因声明 |
| Runtime / Host | 设备选择、上下文、流、内存、同步、错误报告、host validator 与 capture；不伪装 CUDA/HIP |
| Tasks / Lab | 任务别名、Workload 与外部 oracle、分配器、工具链绑定；需要原生比较 arm 时再新增 NativeAdapter |
| Evidence / 报告 | 保存实际产物和观察，定位 replay 拒绝；仅在确有新 artifact 语义时扩展合同 |

新厂商不自动等于新后端，也不自动等于新比较 arm。只有需要新的源码生成机制才加 Backend；只有用户的 Study 要比较原生作者环境才加对应 NativeAdapter。显式登记 source admission、构建工厂、block 与 baseline 投影，未知路径拒绝，禁止 `else` 默认为某个已有实现。

## Ascend 路线需查实的事实

[官方 CANN 架构说明](https://www.hiascend.com/doc_center/source/zh/canncommercial/81RC1/quickstart/quickstart/quickstart_18_0001.html) 区分了 Ascend C、编译器和 Runtime/AscendCL。它提供的是调研方向，不是本项目已有集成：

- 选定硬件和 CANN 版本支持的 kernel 编程/编译路径、实际代码对象及加载接口。
- host/device 编译关系、参数打包、workspace、内存传输、流同步与错误检查。
- 算子数值语义和参考实现；选择的框架扩展是否支持对应型号。
- 目标原生 timer/profiler 的区间、分辨率、reset、归因和质量门；缺失时记录未验证或覆盖不足，不替换成 CUPTI/HIP 时间。

不得凭 CANN 文档中的调用语法相似，就复用 CUDA launch manifest 或线程组含义。查阅与目标版本匹配的 [Kernel 加载与执行接口](https://www.hiascend.com/document/detail/zh/canncommercial/81RC1/apiref/appdevgapi/aclcppdevg_03_1830.html) 并以实物产物核对。不要在模板里预先指定未经核实的 CodeObject 枚举值、warp 宽度或机器参数。

## 当前扩展缺口要如实列出

平台注册表已经集中声明，但还不是“新厂商只改一行”的插件系统。新 CodeObject 的加载器分派、host validator/capture、工具链 worker 和 Target schema 仍有具体登记点（见代码定位表）；GPU 风格的资源模型也可能需要扩展。逐项判断哪些是正常接线、哪些是共享抽象的局限，并测试既有目标不受影响。

至少包含：第三厂商的 typed fixture、未知 vendor/object 拒绝、后端与产物不匹配、未声明事实、错误硬件/SDK/ABI、最小正确执行和数值反例。没有真实硬件时只交付能验证的静态层级与剩余条件，不写虚构测量。
