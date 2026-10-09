# C550 实验如何指导 Compiler 优化

审查依据：源码 `a50523e3`，以及 C550-1 已封存的候选、生成源码、编译资源和独立确认收据。最有价值的下一步是把已发现的调度机制变成带条件的可调用变换，并检验它们能否减少重复探索。目前没有证据要求新增 IR 原语。

## 已确认的机制

证据根：`c550-1:/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006/`。下表前三项位于 `dual-host-development-20261008/dispatch/runs/`；GEMM+Bias 位于 `fault-successor-development-20261008/dispatch-v2/runs/`。

| 任务 | 实际变化 | 独立确认 | 定位 |
|---|---|---|---|
| Pairwise Squared Distance | N 输出从64列分为每CTA4列，保留完整K256和FP32差→平方→求和 | 379.3664→33.9200 μs，11.184× | `pairwise_sqdist`，turn35，seal495，confirmation507 |
| RMSNorm H4096 | 执行组1→4；weight转FP32后移；weight load使用`evict_last` | 36.2752→18.4576 μs，1.965× | `fib_rmsnorm_h4096`，turn28，seal521，confirmation540 |
| Fused-add RMSNorm H4096 | 执行组1→4；调整load、widen和square的次序 | 37.1712→19.6608 μs，1.891× | `fib_fused_add_rmsnorm_h4096`，turn34，seal437，confirmation462 |
| GEMM+Bias | 每CTA1→128列；K256→K8循环；5幅值带→2带补偿合并；bias使用`.cg` | 943.1808→68.3776 μs，13.794× | `gemm_bias`，turn28，seal548，confirmation560 |

四项均通过确认前后五类输入，零不匹配、输入未改变；每臂10次事件计时取均值，运行协议审计通过。四项均有预算超限记录。收益相对固定起始基线，尚不能证明超过成熟实现，也不能证明Compiler整体收益。

Pairwise此次收益属于N分片，不是K分片或GEMM公式替换。Fused-add RMSNorm的原基线已经融合，1.891×不能归因于新增融合。GEMM+Bias最终确认的是同轮第二项`streamedbias`候选。

## 优先级与归属

| 优先级 | Compiler工作 | Lab负责 | 必须保留的边界 |
|---|---|---|---|
| 1 | 验证并扩大已有`specialize_triton_warps`的C550资格 | 选择何时尝试、执行组数 | 当前资格集合不含`xcore1002`；手写候选通过不能替代pass生成候选的后继验收；拒绝多角色、显式同步、存储或寄存器承诺冲突 |
| 2 | 提炼纯寄存器widen-cast后移变换 | 比较不同次序、缓存hint | 保留算术和舍入顺序，不跨副作用、同步或循环边界，不自动重排全部操作 |
| 3 | 复用已有`tile_squared_difference_outputs`及`specialize_squared_difference` | N/K/unroll参数、变换调用时机 | F-2026-10-03-003已内化N分片；此次补充形状和计时证据，不另造同义pass；保留差→平方及独立输出条件 |
| 4 | 为GEMM提供保留原数值算法的显式N/K分块 | 幅值带算法、缓存hint和参数搜索 | 先拆分13.794×的来源；两带及平方比较不直接成为通用隐式优化 |
| 5 | 将可解释的MACA原生分配字段接入共同反馈 | 使用经目标校准的模型辅助选择 | private字节是静态存储，不是动态spill流量；不恢复已退休的结构启发式排名 |

两项RMSNorm的编译资源均从256 registers、336 private bytes变为128 registers、0 private bytes。Pairwise从256 registers、1232 private bytes变为36 registers、0 private bytes。这支持调查资源压力，但没有单独证明每一项改动的因果贡献。MACA已输出资源报告；共同`CompiledResources`目前仅接受cubin/hsaco，扩展时必须保留各字段原生语义和未建模项。

## cast后移的安全范围

首版只处理单角色、直线、无显式同步的寄存器数据流：将BF16/FP16→FP32的widen操作移到它的最早消费者之前。源寄存器在移动区间内必须没有覆盖；移动后仍须支配全部消费者；不得跨循环、分支、barrier、pipeline、异步交接或具有状态副作用的操作。首版只移动cast，不移动global load；源值已经读取，因此不会改变读取时刻。

保持原cast种类、dtype和所有算术操作次序；不删除BF16/FP16窄化节点，不把cast推入算术。Fused-add合同要求先做FP32残差加法，不能替换为BF16加法再widen。多消费者时只能移到最早消费者之前；遇到区间内源覆盖、循环carry或显式寄存器复用就拒绝。反例应由该变换自己的依赖/效应guard拒绝。实际寄存器与性能变化仍由厂商编译结果和设备测量决定。

## 三组最小消融

以下是后继设计，尚未执行。原Run保持封存；每个性能配置只执行一次独立确认，每臂保留10次样本取均值。每组保留自己的完整正确性检查和独立profiler，不能把profiler延时混入事件延时。

### A：执行组与cast后移，10个配置

两个已见任务：R170/C4096的RMSNorm与Fused-add RMSNorm。每项交叉执行组`{1,4}`与weight cast位置`{原位置,最早消费者前}`：**2任务×2宽度×2位置=8个配置**。固定cache hint和其余操作次序，尤其不复制原fused候选额外的load/square重排。

另用既有RMSNorm H128开发任务比较执行组1与4，保留原cast位置：**2个配置**。合计**10个配置、10次固定基线配对确认**，其中三项原配置作为A/A。先验收pass输出，再比较收益，避免把4组固定成所有形状的默认值。

### B：已有N分片工具是否节省探索，6个Agent Run

三个明确声明的开发形状：`R128/K256/N32`、`R1024/K256/N64`、`R256/K512/N64`。第三个先通过独立Workload与基线准入；失败则保留拒绝，不临时换形状。每个形状一组手写Schedule、一组可调用已有N分片pass：**3形状×2方式=6个Run**，每个只运行一次，最多3小时。

两组拿到相同机制说明、参考访问、基线、工具反馈和模型；唯一差异是N分片pass调用权限。比较正确候选比例、达到共同性能门槛的候选数/编译数/时间、最终独立确认性能及拒绝原因。只有一重复，结论定位为机制和工程证据，不宣称统计稳定的总体胜率。c550-bench不参与这组开发。

### C：GEMM分块与数值算法，5个配置

固定原`M128/K256/N1024`。交叉分块`{N1/K256,N128/K8}`与累加`{原五带补偿,候选两带补偿}`：**2×2=4个配置**。四者使用相同默认cache hint；再给`N128/K8+两带`增加`.cg` bias load：**第5个配置**。总计**5次固定基线配对确认**，其中原五带配置为A/A。

另设**4组CPU数值反例**：幅值阈值两侧、强抵消、大值平方溢出、极小值平方下溢。它们用于审查“平方比较替代绝对值比较”和两带算法的适用边界，不回填原Workload成功。先看保持五带的分块能否改善，再决定哪些机制适合Compiler、哪些仅保留为受合同约束的Lab recipe。

## 完成标准

先在开发任务中验证每个pass的guard、反例、语义和设备实现；固定Compiler版本后进入c550-bench。报告分别列出固定基线、pass直接生成候选、Agent最佳候选，避免把已有Agent收益当成新Compiler收益。当前优先路线是执行组资格与cast消融，其次是验证已有N分片调用减少探索，最后是数值合同不变的GEMM分块。
