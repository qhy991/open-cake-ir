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

## 扩展审查后的优先级与归属

首轮审查集中在四个高收益候选。补充审查覆盖46个不同任务Run：44个已封存、2个运行中的部分记录；另对14个低收益/退化Run检查源码与确认结果。关键候选在最新`origin/main`的`1b5cca1b`上做了CPU解析、评估和lowering重放。下面的顺序取代首轮仅按赢家机制排列的顺序；这些新线索不构成额外性能收益。

| 优先级 | 工作 | 证据与边界 | 归属 |
|---|---|---|---|
| P0 | 检查归约读写存储；统一逻辑标量与lowering rank | LayerNorm直接归约全局指针、Cosine把shared结果生成为寄存器；两份保留源码在最新main仍accepted/lowering_eligible | Compiler backend admission、Verifier、lowering |
| P1 | 显式非二次幂填充/分片变换；调查MMA实现域 | arange现有门禁正确，应提供带掩码和中性元的合法改写；小M/N dot的SDK边界需独立探测 | Compiler变换、Triton/MACA route |
| P1 | 受控支持静态分区输出写入 | 三个尾部候选的写入区间互不重叠，但被整buffer单writer规则拒绝；须同步更新def-use、所有权与覆盖分析 | 共享Compiler分析与lowering |
| P1 | 纳入MACA实际资源；报告跨program轴重复工作 | 原生分配已给Agent，但共同CompiledResources不接收mcfatbin；Absmax每行重复4次整行归约 | Compiler/toolchain反馈 |
| P1 | 执行组资格、cast后移与数据复用 | 保留前述RMS来源证据；参数选择和cache hint选择仍归Lab | Compiler变换＋目标验收 |
| P2 | 验证pipeline深度与部分展开组合 | 单阶段unroll已实现；factor2和stages4/2等组合仍未验收 | 后端实现与资格 |
| P2 | 让完整Program进入统一优化评测 | IR和多阶段执行已有，common Run缺少完整Program计时/归因；限制split-reduction、partial histogram＋merge | Evaluation/Executor/Lab，不能称新IR需求 |
| P2 | 更清楚的合法表达与能力提示 | 广播轴、标量/向量、同步与资源选项反复被误用；错误应指出合法修法及责任层 | Frontend/diagnostics＋Lab材料 |
| P2 | 复用已有N分片；保数值合同的GEMM分块 | Pairwise相关pass已存在且该Run已获授权；GEMM两带算法先单独消融 | Compiler改写与Lab策略 |

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

先在开发任务中验证每个pass的guard、反例、语义和设备实现；固定Compiler版本后进入c550-bench。报告分别列出固定基线、pass直接生成候选、Agent最佳候选，避免把已有Agent收益当成新Compiler收益。扩展审查后，先修复存储/rank一致性和前置诊断，再验证分区写入、资源与重复工作反馈；原三组机制消融继续作为后续计划。


## 新发现的具体证据与验证边界

以下A指两机`/root/open-cake-runs-reviewed/c550-evolution-parallel-author-20261006`。这是冻结事件的只读审查，未重新执行GPU任务，未修改原Run。

### 归约存储承诺和标量表示应先修复

C550-2 `A/gated-event-task-readiness-20261006/runs/layernorm`的编译失败event403把全局张量`x`直接送入reduce，生成`tl.sum(x.to(tl.float32),axis=0)`，原生编译报指针不能转FP32。Cosine event360声明`lm.smem(...).view((1,),fp32)`为归约输出，生成`aa_slot = tl.sum(...)`，未实现shared存储；随后`tl.sum(aa_slot,axis=0)`对rank0标量失败。

两份保留的完整候选在最新主线`1b5cca1b`上仍通过静态验收并生成相同关键语句。应先由拥有实现的backend拒绝未实现的reduce输入/输出存储，不能默默换成register。独立的全register最小二次归约同样通过检查并发出无效的第二次`tl.sum(axis=0)`，说明scalar问题不依赖shared误用。应同时解决合法register singleton归约的逻辑`[1]`与物理rank0对应。首要目标是承诺一致性与明确诊断，不是宣称加速。

### 静态分区写入可以扩大尾部优化空间

- C550-1 `A/dual-host-development-20261008/dispatch/runs/fib_rmsnorm_h1536`，event5：同一row写`[0:1024]`和`[1024:1536]`，试图复用已加载输入。
- C550-1 `A/fault-successor-development-20261008/dispatch-v2/runs/add_rmsnorm_bf16`，event126：out与residual_out分别写`[0:2048]`和`[2048:2560]`，保留BF16舍入后归一化。
- C550-2 `A/path-successor-development-20261007-v2/runs/softmax_backward`，event141：同row写`[0:512]`和`[512:1024]`。

这些静态切片互不重叠并覆盖输出行；现有单writer规则是保守的支持子集，并非已证明存在冲突。第一版扩展可限定同role、无loop、普通global output、同一injective program map、无alias/atomic/readback。必须证明交集为空、并集覆盖、访问不越界，并更新producer/consumer分析。覆盖重叠、缺口、跨role、动态重复映射等反例；不能仅删除`BUFFER_MULTIPLE_WRITERS`。

### 可静态解释的重复计算比笼统资源建议更有用

C550-2 `A/path-successor-development-20261007-v2/runs/absmax_rescale`，确认候选seal454的grid为`[4,128,1]`。每个输出列tile都重新加载全1024宽行并归约max，再读取自身256列切片；每行有4份整行归约。独立确认仅0.974×。

Compiler可以报告“哪些操作不依赖该program轴却被重复、逻辑重复倍数是多少”。这只能是报告级提示：缓存可能复用数据，实际DRAM流量未知；重复计算也可能换取更多并行或更低寄存器需求，不应硬拒绝。

### 完整Program与目标资格要按责任层处理

`F-2026-10-05-005`的RMSNorm event57已能解析并降低为两个kernel，但common优化Run拒绝非CUBIN Program的计时和归因。已有独立Program执行/MCPTI证据不能替代配对端到端计时。补齐后才能公平比较单kernel与split-reduction/merge方案。

`F-2026-10-05-004`保留了maxnreg、16执行组、pipeline＋unroll、warp specialization的精确请求。它们是独立的实现/资格问题；单阶段unroll和最多8组路径已有支持。不得删除pipeline、降组数或复制NVIDIA限制来让检查通过。

### 还有研究机会，但证据弱于上面的具体缺口

Histogram起点按bin重读输入，存在分块局部统计＋merge的研究机会；需要同时满足精确bin归属和FP32输出合同。Gather可探索多行批处理，MaxPool可探索重叠窗口复用。这些来自起始实现结构，尚无确认收益；atomic/transpose等共享原语的存在不等于xcore1002已获资格。当前FP8固定形状SIMT和streaming路径已经实现，不能再描述成“FP8完全缺失”。

### 不应把失败和低收益直接当作Compiler上限

多个低收益候选的private memory已经为0，不能将剩余瓶颈统称spill。PReLU和RMSNorm输入梯度的搜索候选在独立确认中退化，应保留固定基线；这是Lab选择/评测问题。事件计时与profiler归因区间不同，不能将两者相减直接解释为Python wrapper成本或证明内存带宽饱和。

取得Pairwise11.18×的冻结Run明确授予`tile_squared_difference`、`tile_squared_difference_outputs`、`specialize_squared_difference`。手写成功不证明缺授权；应进一步检查工具使用、适用条件和可发现性。


## 审查规模、频率与重复探索

以下按每个Run内已有提交产物身份去重，再跨任务求和；不是结构独立的优化机制数量。活跃Run只统计已发生记录。频率用于挑选调查入口，不直接证明缺陷或收益。

| 保留的条件 | 任务数 | 提交产物去重数 | 解释 |
|---|---:|---:|---|
| maxnreg资格未完成 | 40 | 42 | 需验证厂商选项确实被执行，不能直接放开 |
| 大于8执行组路线未验收 | 15 | 17 | 与Target硬件线程上限分别判断 |
| 非二次幂执行组 | 14 | 15 | 正确拒绝的工具链选项；应缩小作者无效搜索 |
| pipeline与unroll组合未支持 | 12 | 14 | 基本单阶段unroll已存在 |
| 非二次幂arange | 9 | 11 | 门禁正确；可研究显式填充、mask和精确分片 |
| 完整Program归因不可用 | 20 | 21 | Evaluation支持边界，不能归为IR表达力不足 |
| 广播形状/轴错误 | 23 | 38 | 正确拒绝，需增强类型修复提示 |
| 字面量与广播语法限制 | 30 | 40 | 可研究无歧义的规范化表达，保持一种IR形式 |
| 跨role数据竞争 | 23 | 46 | 非法候选，不计作硬件并行能力缺失 |
| 多role纯lowering拒绝 | 3 | 3 | 与上述非法竞争分别审查 |

另有26个任务出现85组跨轮相同规范表示，共172次后续重复出现。Compiler已有等价标识；不能把它说成Compiler没有规范化，也不能把172次全算成GPU浪费。已核验一个实际重复案例：C550-1 FIB fused-add RMSNorm H4096的Turn12与Turn29拥有同一规范候选，却都发生原生编译、search与attribution。

这适合由Lab复用已绑定相同Compiler/Target/toolchain/Workload的编译和正确性证据，并显式区分有意性能复测。跨任务、不同版本或不同参考访问不能直接复用。固定Study的材料与变换权限继续优先。
