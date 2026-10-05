# FlashInfer-Bench 改写与外部实现：实验综述 / Experimental status

秦海岩（Haiyan Qin） · 2026-09-21 · NVIDIA B300 / `sm_103a`

本文是[仓库技术报告](../../README.md)的实验附录，是现有 Finding、节点报告和[发布记录](records.json)的阅读投影，**不建立第二份实验账本、冠军表或晋升规则**。引用时记录本文的 Git 提交与章节；实验版本仍以各行原始报告自身的源码和二进制为准。正文按截至本次核对的状态整理，旧记录及失败证据原样保留。

## 当前结论

26个FlashInfer任务中，**16个已有固定shape的三方数值比较，另10个还没有完整外部性能结论**（已追加026与008完整Program的新对齐比较）。已测任务均保留原始starter、改写产物与外部参考；“有结果”不等于“改写更快”，也不等于全部上游shape、框架端到端或模型性能已通过。

下面明确列出的代表产物/原starter，对外部的任务级结论为：

| 合格外部边的状态 | 数量 | 任务 |
|---|---:|---|
| Cake领先 | 2 | 002、022 |
| close_null：未检出满足原门槛的实质差异 | 6 | 001、006、009、010、023（仅原starter）、025 |
| Cake落后 | 4 | 003、005（保留较快原starter）、008、026（新实验中的generic control） |
| 暂无合格外部边 | 4 | 004、007、021、024 |
| 尚未闭合外部性能比较 | 10 | 011、012–020 |

这不是挑选最小延迟后建立的全局排名：005/023显式使用原starter的独立合格边，新候选的失败边仍保留；022没有证明4组候选优于原starter。003后续对齐候选虽然数值正确，其性能质量失败，不能继承整行generic的合格结论。不同实验的比值不相乘，也不求跨任务几何平均值。

截至2026-09-21本次节点核对，本任务已提交的009/010/026七项比较、四项新NCU均为completed；M3节点active_runs=0、queue=0，7/8GPU空闲。其他GPU占用不是本任务未释放。随后026新对齐实验已完成50/50guards和2550数值快照；新candidate/control边通过，candidate/external边仍CV失败。008新的完整split-K Program随后也已完成，2550份快照数值通过，但两条候选计时边CV失败；不能据此宣称收益或合格回退。排队、采集、CPU验证仍以原submissions和节点state为准。

## 与外部实现的逐任务结果

单位为微秒，数值来自**同一条候选/外部配对边**。失败行保留观测延迟便于诊断，但不计算合格加速比；close_null不写成更快或数学等价。正负比例均按本行固定shape定义。“组”指Schedule.execution_groups，每组lane宽度来自Target。

| 任务 | 本次固定输入 | 报告使用的Cake产物 | Cake µs | 外部 µs | 结论 | 证据入口 |
|---|---|---|---:|---:|---|---|
| 001 | R79/H2048 BF16 | 对齐变体 | 2.3040 | 2.3360 | close_null | [记录](README.md#nvidia-alignment-001-optimized_vs_external-20260920) |
| 002 | R170/H4096 BF16 | 对齐变体 | 3.0080 | 3.2960 | 领先1.096× | [记录](README.md#nvidia-alignment-002-optimized_vs_external-20260920) |
| 003 | R64/H7168 BF16 | 整行、16组、generic | 3.7440 | 3.0400 | 落后23.16% | [记录](README.md#nvidia-003-whole-row-structure-optimized_vs_external-20260921) |
| 004 | M1/N128/K2048 FP16 | 保留改写候选 | 2.6880 | 2.3680 | CV失败，描述值 | [记录](README.md#nvidia-fib-external-004-20260920) |
| 005 | M1/N256/K7168 FP16 | 原starter（改写回退） | 8.6720 | 6.9280 | 落后25.17% | [记录](README.md#nvidia-fib-external-005-starter-reviewed-20260921) |
| 006 | M1/N2048/K4096 FP16 | 4组候选 | 8.5440 | 8.5755 | close_null | [记录](README.md#nvidia-006-width-w4-optimized_vs_external-20260921) |
| 007 | M1/N4096/K4096 FP16 | 8组候选 | 12.3200 | 11.6480 | CV失败，描述值 | [记录](README.md#nvidia-007-width-w8-optimized_vs_external-20260921) |
| 008 | M1/N4096/K14336 FP16 | sliced-K、8组 | 41.0405 | 26.8155 | 落后53.05% | [记录](README.md#nvidia-008-width-w8-optimized_vs_external-20260921) |
| 009 | M1/N5120/K2048 FP16 | 4组候选 | 8.6725 | 8.7200 | close_null | [记录](README.md#nvidia-009-width-w4-optimized_vs_external-20260921) |
| 010 | M1/N6144/K4096 FP16 | 8组候选 | 17.5680 | 17.2805 | close_null | [记录](README.md#nvidia-010-width-w8-optimized_vs_external-20260921) |
| 021 | R2528/H128 BF16 | 保留候选/原结构 | 2.9120 | 2.5605 | CV失败，描述值 | [记录](README.md#nvidia-fib-external-021-20260920) |
| 022 | R539/H512 BF16 | 4组候选；原starter也领先 | 2.3360 | 2.7200 | 领先1.164× | [记录](README.md#nvidia-assignment-022-w4-optimized_vs_external-20260920) |
| 023 | R539/H1536 BF16 | 原starter（新候选未合格） | 4.0320 | 4.0640 | close_null（仅原starter） | [记录](README.md#nvidia-fib-external-023-starter-reviewed-20260921) |
| 024 | R79/H2048 BF16 | 保留改写候选 | 2.4640 | 2.2720 | CV失败，描述值 | [记录](README.md#nvidia-fib-external-024-20260920) |
| 025 | R170/H4096 BF16 | 对齐变体 | 2.7200 | 2.7200 | close_null | [记录](README.md#nvidia-alignment-025-optimized_vs_external-20260920) |
| 026 | R64/H7168 BF16 | 本轮generic sliced-w8对照 | 6.4005 | 2.6560 | 落后140.98%；仅generic对照 | [记录](README.md#nvidia-026-alignment-starter_vs_external-20260921) |

- **005：**改写9.152µs对原starter8.6405µs是合格回退（0.944×），因此保留starter。表中starter/external的8.672/6.928µs来自另一条自己的配对边，不能混用8.6405计算外部差距。
- **022：**原starter已经以2.336/2.688µs合格领先外部1.151×。4组候选对外部1.164×合格，但对原starter的比较未通过CV，所以不能据此替换starter。
- **023：**原starter4.032/4.064µs是close_null。新候选2.880/4.095µs仅为描述，其对starter与external两边均CV失败；不能宣称1.42×收益。
- **026：**旧6.464/2.720µs外部边仍CV失败。新的对齐候选对固定generic对照5.920/6.4005µs为合格1.081×，但candidate/external5.920/2.656µs再次CV失败。本轮独立generic-control/external6.4005/2.656µs通过质量门禁，表格仅据此标记generic产物落后；没有将该资格赋予更快的aligned候选，也不与旧1.787×相乘。

### 尚未完成的十个任务

| 任务 | 当前完成的工作 | 仍缺的验收 |
|---|---|---|
| 011 GEMM N28672/K4096 | Workload、原参考与完整Cake入口已存在；已查明外部按相同/邻近指针缓存B转置 | 同指针内容改变的真实验证；明确原始/派生参考边界；固定shape正式比较 |
| 012 GQA paged decode KV4 | captured、boundary各1个四阶段Program，Lab CPU封存/readback通过 | 正常task launcher、完整Program GPU数值与外部比较 |
| 013 GQA paged decode KV8 | 同上 | 同上 |
| 014 GQA paged causal prefill KV4 | 同上 | 同上 |
| 015 GQA paged causal prefill KV8 | 同上 | 同上 |
| 016 GQA ragged causal prefill KV4 | 同上 | 同上 |
| 017 GQA ragged causal prefill KV8 | 同上 | 同上 |
| 018 MLA paged decode | captured、boundary各1个四阶段Program，Lab CPU封存/readback通过 | 同上，并核对大输入CPU准备与快照预算 |
| 019 MLA paged causal prefill | 同上 | 同上 |
| 020 FP8 MoE routing top-k8 | captured的八阶段Program，Lab CPU封存/readback通过 | 混合dtype/标量/索引ABI与原oracle的完整共同GPU评测、外部比较 |

总计**17个Program、72个叶阶段**，全部经实际`load_baseline_bundle`、`program_components`及所有case公共ABI重读检查；不是仅看到配置文件便宣布成功。CPU封存已验证，不表示Agent正常启动、GPU数值或性能已通过。最初遗漏route.entry_point的调用方拒绝保留，补全已有route后通过，没有为此放宽Compiler。

attention公共输出为`output:bf16`和`lse:fp32`；外部调用还涉及索引/标量。当前比较工具的单`out`、BF16/FP16接口不能直接用于这些任务。已有Program adapter应继续复用；先让012沿正常任务入口闭合，再扩展。并行开发中的原生Program张量接口工作应先核对集成状态，避免另建一套运行器。

## Starter、compiler floor与候选收益

**原starter不变**。原始bundle只作固定对照，不因本次报告、CPU封存或profile而自动替换。最近手工结构/组数实验无自动晋升；此前历史Campaign自己的晋升记录保持其原意。下面每个比值都只属于自己的对照边。

| 任务/机制 | 对照身份 | 合格结果与处置 |
|---|---|---|
| 001 guarded alignment | 此前optimized二进制，不是最初starter | 1.083×；external为close_null |
| 002/025 guarded alignment | 此前optimized二进制 | 对旧control的CV失败；不能用external边倒推出compiler收益 |
| 003重新编译、1组 | 旧optimized binary | compiler-floor close_null |
| 003 whole-row、16组 | sliced-w16中间候选 | 3.713/7.040µs，1.896×；后续对齐比较CV失败 |
| 004改写 | 原starter | 2.688/3.712µs，1.381×；external仍CV失败 |
| 005改写 | 原starter | 0.944×，回退，保留原starter |
| 006同结构floor | 旧原starter | close_null；仅换Compiler未产生合格改善 |
| 006 4组 / 四列4组 | 旧原starter | 1.079× / 0.903×；保留4组候选与四列负结果 |
| 007 8组 | 本轮canonical baseline-only新冻结原模板 | 1.097×；不是旧Compiler floor |
| 008 sliced8组 / masked整K | 新冻结原模板 / 中间sliced8组 | 1.053× / 0.805×；整K回退 |
| 008两阶段split-K Program | 中间sliced8组 | 55.073/41.152µs，CV失败，仅描述；2550快照正确 |
| 009 4组 / 8组 | 新冻结原模板 | 1.096× / close_null |
| 010 4组 / 8组 | 新冻结原模板 | 1.081× / 1.116× |
| 021原结构候选 | 原starter | close_null，未证明改写改善 |
| 022 4组 | 原单行starter | 无合格改善，保留starter |
| 023新候选 | 原starter | CV失败，不晋升 |
| 024改写 | 原starter | 2.496/6.177µs，2.475×；external仍CV失败 |
| 026同结构floor / sliced8组 | 旧原starter | close_null / 1.787× |
| 026 whole-row8组 | 中间sliced8组 | 7.392/6.400µs，0.866×，明确回退 |
| 026 guarded alignment | 固定generic sliced8组 | 5.920/6.4005µs，1.081×合格；新候选external边仍失败 |

“starter”在旧比较工具的字段中有时只是**对照槽位名**。对齐消融和结构后继会把旧optimized产物放到该槽位；本报告按真实角色标明它，不将中间control称为最初starter。

## 独立Profiler：已经观测到什么

四项独立NCU（源码`34f04b40`）全部完成：每角色五个验证case加一次profiled primary，四实验共**72份完整调用观测**，原输出/输入不变检查通过；13个kernel实例metrics完整。NCU覆盖完整callable的所有launch，使用cold kernel replay。它不提供新的CUPTI速度结论，也不修复既有CV失败。

| 被采样产物 | Cake候选 / control | 外部观测 | 支持的分析与限度 |
|---|---|---|---|
| 003 whole-row / sliced-w16 | 同512线程、64CTA；寄存器51/39；DRAM读1.857/1.860MB；L2流量6.256/5.833MB | 448线程、64CTA；39寄存器；L2 4.877MB | 整行收益不能解释为HBM读取量下降；归约/依赖结构值得分离验证 |
| 008 masked整K / sliced-w8 | 同256线程、4096CTA；寄存器80/32；active warps 30.46%/76.62%；DRAM读都约117.49MB | stage1为1024CTA，stage2为16CTA；两次launch均捕获 | 回退与更高寄存器用量、较低活跃warp相容；没有证明唯一因果，更没有观测到spill |
| 009 w4 / 原starter | 128/32线程、均5120CTA；寄存器32/64；active warps 47.02%/20.96% | 此profile进程选择`sk_kernel<1,1>`，128线程、160CTA，shared约76KB | 可见工作划分差异；不能把此分支选择倒填给旧timing进程 |
| 026 whole-row / sliced-w8 | 同256线程、64CTA；寄存器48/28；L2 3.698/4.124MB | 256线程、64CTA；62寄存器；L2 3.478MB | 减少L2总量没有带来更快的候选；外部CV失败仍保留 |

以上13个kernel实例的local-memory load/store sectors均为0：**本次采样没有观察到local-memory流量**，不能把回退笼统归因于寄存器spill，也不能泛化到未测shape。理论occupancy limit不是实际瓶颈证明；stall百分比不是绝对等待时长。`dram__bytes_write=0`也不等于没有逻辑输出写入；完整输出已经单独验收。

003的静态生成代码还显示整行与切片的归约/同步结构不同。静态指令site数量不等于动态执行次数或流量；若下一步分离“归约合并”和“保留值避免重读”，必须先检查编译器确实生成了不同的消融产物。

### 008完整Program后继结果

新候选使用已有Program/IR表达完整split-K：第一阶段2048CTA，第二阶段256CTA，FP32partial经完整求和后输出FP16。原CPU oracle与全部输入不变检查通过，说明完整双阶段产物已实际执行。它并非外部算法的忠实复现，也没有新IR原语。

| 本轮配对边 | 左/右 µs | 最大cohort CV | 接受结论 |
|---|---:|---:|---|
| 新Program / 固定sliced-w8 | 55.073 / 41.152 | 0.068561 | 计时质量失败；名义0.747×仅描述 |
| 新Program / 外部 | 55.072 / 26.688 | 0.054530 | 计时质量失败；名义0.485×仅描述 |
| 固定sliced-w8 / 外部 | 41.1845 / 26.624 | 0.014607 | 外部更快，质量通过 |

三边分别对应[发布记录](README.md#nvidia-008-splitk-program-optimized_vs_starter-20260921)、[外部边](README.md#nvidia-008-splitk-program-optimized_vs_external-20260921)和[control边](README.md#nvidia-008-splitk-program-starter_vs_external-20260921)。不将“stage-result passed”当作计时质量通过；原始报告整体为measurement_quality_failed。保留此前sliced-w8的合格性能代表，本候选不晋升。旧NCU采样的是其他候选，不能直接用来解释这个新Program的因果机制。

### 008完整Program的stage对齐后继

共享封存与加载路径现已支持每个stage的generic/aligned变体，并依据实际公共、私有地址分派。新产物与旧完整Program保持相同源码、常量、grid、block和8组；第一阶段静态load site从136条b16变为17条v4.b32，不据此推导速度。

真实门禁[60/60通过](README.md#nvidia-008-program-alignment-guards-20260921)：五个case各含aligned、a/b/out各2/4/8B偏移及private FP32 partial的4/8B偏移。共120次stage launch、65次受限leaf提前拒绝，最终公共输出与输入不变全部通过。FP32的2B偏移不满足元素对齐，不列为合法输入；private内容不是新增独立公共oracle输出。

门禁最终验收后唯一提交正式比较，2,550份完整观测数值通过，但三条计时边均失败：

| 本轮配对角色 | 两者中位数 µs | 最大cohort CV | 接受结论 |
|---|---:|---:|---|
| 新aligned Program / 固定unaligned Program | 96.785 / 55.041 | 0.076709 | CV失败，仅描述 |
| 新aligned Program / 外部 | 95.552 / 26.721 | 0.097674 | CV失败，仅描述 |
| 固定unaligned Program / 外部 | 55.072 / 26.625 | 0.061629 | CV失败，仅描述 |

三条边见[新旧Program](README.md#nvidia-008-program-alignment-optimized_vs_starter-20260921)、[新Program/外部](README.md#nvidia-008-program-alignment-optimized_vs_external-20260921)、[control/外部](README.md#nvidia-008-program-alignment-starter_vs_external-20260921)。本轮control是旧完整Program，不能误称sliced-w8或最初starter。该能力已获得完整Program正确性证据，尚无接受的性能改善；观测较慢也不写成合格回退。此前sliced-w8的合格external代表与2/6/4/4统计均保留，不重复同候选刷CV。下一步分别检查Program调用开销和编译kernel；本轮没有新的NCU结论。

### 008执行器后继：正确性通过，候选计时仍未合格

CPU诊断定位了主机重复工作：同一个已绑定manifest在对齐分派及受限CUDA leaf中反复canonical序列化。修复只省略对象与自身的重复比较；不同对象仍检查原canonical值，每次实际指针、ABI、上下文、CUBIN和leaf对齐检查均保留。真实封存Program与FakeDriver的CPU探针中，aligned阶段间隔中位数从78.9765变为31.373µs；这不是GPU加速比。详见[F011的CPU诊断与修复](../../../findings/2026-09-20-011-triton-aot-pointer-alignment.json)。

新执行器`3bc753a4`使用完全相同的candidate `b5e1c34b`和完整unaligned Program control `65742cbb`，没有重编kernel。先完成新的[60/60设备guards](README.md#nvidia-008-program-dispatch-guards-20260921)，再唯一提交正式比较；全部2,550份公共输入／输出观测通过，计时结果为：

| 新执行器下的配对 | 两者中位数 µs | 最大cohort CV | 接受结论 |
|---|---:|---:|---|
| aligned Program / unaligned Program | 46.240 / 55.072 | 0.141390 | CV失败；10/10 pair wins仍不构成合格加速 |
| aligned Program / 外部 | 45.824 / 26.688 | 0.153932 | CV失败，仅描述 |
| unaligned Program / 外部 | 55.040 / 26.624 | 0.031967 | 质量通过；外部更快 |

对应[候选/control](README.md#nvidia-008-program-dispatch-optimized_vs_starter-20260921)、[候选/外部](README.md#nvidia-008-program-dispatch-optimized_vs_external-20260921)、[control/外部](README.md#nvidia-008-program-dispatch-starter_vs_external-20260921)。候选两边的CV失败不能由独立control边修复。旧执行器和新执行器是两次独立运行，不能把约96µs到约46µs的描述值直接换成合格的Executor GPU加速比。相同二进制的CPU运行路径已简化，设备正确性也通过；GPU候选收益与外部资格仍未验收，没有新NCU或唯一硬件因果结论。本节不改变旧sliced-w8代表、已有任务统计或任何旧记录。

## 差距的原因与应该修改的层

1. **AOT访存信息是已验证的Compiler/工具链问题。** 运行时检查对齐并保留通用路径，解决“为了vectorization而假设所有指针16B对齐”的错误边界。001/002/025已有正确性及部分合格性能证据；003对齐性能未合格。026 sliced8的新对齐产物已通过50guards和2550快照，对generic对照合格改善1.081×但新候选external边CV失败：相同源码/常量/grid，通用PTX为36条b16 load和4条b16 store，对齐leaf为3条v4.b32加15条v2.b32 load、7条v2.b32 store；这些静态变化本身不等于速度提升；本次性能结论来自独立的配对测量。
2. **工作划分与完整算法结构仍是主要候选差距。** 009从1组到4组接近外部，但8组反而无实质改善；008的一CTA一输出全K方案没有复现外部split-K的完整两阶段结构。现有IR能够表达合法masked ProgramMap，Program也已支持多阶段封存，所以下一步应先写完整候选并测量，不能仅据性能差距断言缺少IR原语。只有现有结构无法合法表达时，才给出具体Compiler/IR/Pass缺口与反例。
3. **公共运行路径仍有工程缺口。** 012–020需要正常task launcher、混合dtype/多输出外部适配和共同Evaluation。Compiler Program已存在，重复创建Lab私有图或临时GPU runner会造成第二套所有权。
4. **参考语义与测量质量也是未完成项。** 011的指针缓存不证明B内容不变；007及026新aligned候选的外部CV失败不是数值失败或已证实性能优劣；026的generic对照另有本轮独立合格边。需要记录输入刷新验证或具体测量波动原因，不能通过放宽门槛或重复运行直到绿色来补结论。

优先顺序：008完整Program及stage对齐后继均已取得数值正确性，但尚无合格性能收益；主机重复序列化已修复并经过设备正确性复验，但候选计时仍高波动；后续需独立区分阶段执行、主机间隙与测量波动，再选择不同的后继，不重复同候选刷CV。先闭合012的正常入口与完整公共输出验证；对011先完成参考有效性检查。026对齐已完成本轮受控比较，保留失败external边，不重测刷通过。上述未完成事项不作为已有性能结论。

## 测量、资源与历史边界

- 性能比较使用原oracle、CUPTI冷L2、原CV与5%materiality门槛；一次配对测量保留连续独占区间。外部多kernel callable按全部活动的`max_end-min_start`计时，不只取第一kernel。
- CPU准备/原生编译在租约外，GPU采集保留每次完整输出与输入观测，批量验证在worker退出后执行。lossless输入引用只在逐字节相等时引用首值，每个新literal保留并按原规则检查；每次输出仍完整保留。
- 026 floor的节点时间显示capture阶段约17.4秒、随后的CPU verify约746.9秒；这是stage wall time，不是kernel耗时，也不是旧1800秒timeout的因果重放。
- 最近009/010/026七实验：17,850份数值快照通过、21条计时边15条quality通过。四NCU是另72份观测，不能混作新的配对样本。
- 最新节点检查时共享状态盘只剩约3.52GB；没有因此清理历史证据或立即提交另一个大准备。018/019 captured的当前f64 CPU准备文件约22.803/22.825GB，020约67.651GB；另外还要计快照、解码缓存和外部参数组。资源规划中的42指**每cohort的调用数**，不是42个cohort；原规划用词错误已在Finding中澄清。不要无预算并发所有大shape。
- 旧006评测取消记录不变。旧026 M2 `cake-sm-103a-ec2f473ce604`是1800秒run_timeout/exit124、缺judge result、broker_fault、无endpoint/晋升。新026的completed数据是独立后继证据，不追认旧失败为成功。
- 003外部是明确标识的Graph主机参数修正派生参考；不是未改动原文。其余CUDA的C++20适配保留原源码及逐文件C++17失败/C++20成功probe。007/009是含库/运行时选择器的完整调用，不声称重建不可见库内部。

## 证据与可复查路径

表格链接到稳定publication record id；每条记录保存原始节点locator、实验版本和来源提交。数值、失败与curation由[F010](../../../findings/2026-09-20-010-rewrite-external-performance-gap.json)、[F011](../../../findings/2026-09-20-011-triton-aot-pointer-alignment.json)及其引用的节点报告负责。

节点共享根为`/mnt/b300-shared/home/qinhaiyan/cake-experiments/`：

| 证据集合 | 原始入口 | 性质 |
|---|---|---|
| 001/002/025 alignment | `nvidia-alignment-20260920/` | guards与配对结果，失败CV保留 |
| 003/022工作划分 | `nvidia-work-assignment-20260920/submissions.json` | 七项比较 |
| 003 whole-row与alignment | `nvidia-whole-row-20260921/submissions.json`、`nvidia-whole-row-alignment-20260921/comparison-submissions.json` | 结构合格、后续alignment失败边分别保留 |
| 006 | `nvidia-gemm-006-20260921/submissions.json` | 三项比较与负结果 |
| 007/008 | `nvidia-gemm-007-008-20260921/submissions.json` | 四项比较 |
| 009/010 | `nvidia-gemm-009-010-20260921/submissions.json` | 四项比较 |
| 026 | `nvidia-rmsnorm-026-20260921/submissions.json` | 三项比较 |
| 新NCU | `nvidia-phased-profiles-20260921/submissions.json` | 003/008/009/026四项，已完成 |
| 012–020 Program封存 | `flashinfer-program-sealing-20260921/complete-route/sealing-results.json` | 仅CPU封存与readback |
| 026新对齐 | `nvidia-rmsnorm-026-alignment-20260921/guard-submissions.json`、`comparison-submissions.json` | 50guards通过；2550快照通过；三边质量分别记录 |
| 008新split-K Program | `nvidia-gemm-008-splitk-program-20260921/submissions.json` | completed；2550数值通过；两candidate计时边CV失败 |
| 008 Program stage对齐 | `nvidia-program-alignment-008-20260921/guard-submissions.json`、`comparison-submissions.json` | 60guards与2550数值通过；三计时边均CV失败 |
| 008 Executor后继 | `nvidia-program-dispatch-008-20260921/guard-submissions.json`、`comparison-submissions.json` | 新60guards与2550数值通过；两candidate边CV失败，仅完整unaligned control/external边质量通过 |

发布顺序与source identity遵循[开发分支](../../DEVELOPMENT_BRANCHES.md)。结果PR与源码PR只在独立审查及对应CI通过后合入；报告中的版本不替换运行时冻结的提交。所有历史失败与baseline保留。

## English reading summary

Sixteen of the 26 tasks have fixed-shape three-way numerical comparisons. With each representative artifact or original starter explicitly identified above, two tasks have a qualified external lead, six are close_null, four remain slower, and four lack a qualified external edge. Task026 uses the retained generic control edge from the new alignment experiment; its faster aligned candidate still lacks a qualified external edge. Task 011 and tasks 012–020 have no complete external performance conclusion. This is a descriptive evidence projection, not a portfolio selector or cross-task leaderboard.

The qualified original starter matters: task 005's faster retained starter still loses to the external implementation; task 023's starter is close_null while its nominally faster rewrite remains unqualified. Task 022 already has a qualified starter lead, so the four-group candidate is not automatically promoted. Whole-row structure helps 003 but regresses 008 and 026. Newer failed quality edges inherit no earlier acceptance. The026alignment experiment passed50guards and2550numerical snapshots, with a qualified1.081x gain against its fixed generic control. Its own external timing failsCV; the independently paired generic-control/external edge passes and remains distinct.

Four independent NCU evaluations passed 72 complete input/output observations and captured 13 kernel instances. No local-memory traffic was observed in these instances. Task 008's whole-K version uses 80 versus 32 registers and has lower active-warps percentage than sliced-w8, which is consistent with register/residency pressure but does not establish unique causality or spilling. Task 009's profiler process selected `sk_kernel<1,1>`; that cannot identify its earlier timing process's runtime choice. Profiles do not repair CV failures or replace paired CUPTI timings.

All 17 complete Programs for 012–020 were CPU-built and independently reloaded, covering 72 leaf stages. Ordinary task entry points, complete common GPU evaluation and Workload-driven multi-output/mixed-dtype external bindings remain unfinished. The system already owns Program composition; add missing behavior at its existing owner rather than a second runner. A new complete two-stage008split-K Program now passes2,550 numerical observations, but both candidate/control and candidate/external timing edges failCV. Its nominally slower latency is descriptive, not a qualified regression; no performance improvement or promotion is accepted. No full upstream shape suite, model E2E, inherited upstream score or automatic promotion is claimed. Cite the report commit together with each experiment's own source revision, target, workload and original evidence locator.

The subsequent per-stage alignment treatment of the complete008 Program passed60 public/private pointer guards and2,550 full public observations. Its aligned/unaligned Program edge is96.785/55.041us and aligned/external edge95.552/26.721us; all three paired edges fail the original CV gate. These are descriptive latencies, not an accepted gain or qualified regression. The fixed control is the previous complete unaligned Program, not sliced-w8 or the original starter. No representative or promotion changes; static vector load-site reduction does not imply a performance improvement.

Executor3bc subsequently reuses the exact same aligned and unaligned Program binaries, passes60 fresh device guards and2,550 full observations, and retains the original timing gates. Aligned/unaligned46.240/55.072us and aligned/external45.824/26.688us failCV; only unaligned Program/external55.040/26.624us passesquality. The CPU dispatch simplification is implemented and device-correctness checked, but no qualified candidate GPU speedup, external qualification or promotion follows. Neither the descriptive cross-run latency change nor ten pair wins bypasses the CV requirement.
