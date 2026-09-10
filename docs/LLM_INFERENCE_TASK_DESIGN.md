# LLM 推理任务设计：Projection、MLP 与 Attention

状态：设计提案，2026-09-10。本文定义后续实现范围，不是 frozen Workload、已注册任务
或 GPU 资格记录。本轮不修改 Compiler、Executor、历史合同，也不启动实验。

目标是让真实推理中的 shape、数据复用、布局、舍入和融合问题驱动 Kernel–Compiler
协同演进。不同模型角色复用同一种数学实现；只有 ABI、输出或组合语义不同时才分任务。
每个优化 campaign 仍固定一个 shape 和一个 target，多形状矩阵用于选择任务与迁移验证。
首轮目标为 B200/sm_100a；B300、Metal 或其他设备另建独立执行绑定，不继承校准或成绩。

## 1. 来源及可继承的范围

本次读取的代码/资产版本：

- KernelEval：`qhy991/KernelEval@8cdf0b197ca0987f414ff1029b141727f9b4552a`，直接读取其当前
  `definitions/cases_common_103/` 定义。主要复用部署形状、packed-format 边界、
  M=1/8/512 三种工作量和固定 incumbent 的比较原则。定义中的 generated/verified
  标签不等于本任务的完整 GPU 资格，也不能继承旧设备的性能。
- AKA：`qhy991/AKA@a846b1c80a28939202a12cf64a8bf0aaf34ebefe`，
  `datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl`。选定的十条原记录
  均报告 qualified、compile/correctness/sanitize passed+valid，引用的本地源资产可读。
  本次没有重放 GPU qualification；qualified 仅适用于原记录的窄合同。
- 模型几何：官方 [Qwen2.5-7B-Instruct config](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json)
  与 [Qwen3-4B config](https://huggingface.co/Qwen/Qwen3-4B/blob/main/config.json)，本次观察对应
  模型仓库 HEAD 分别为 `a09a35458c702b33eeacc393d103063234e8bc28`、
  `1cfa9a7208912126459214e8b04321603b3df60c`。正式合同应读取并封存选定 revision 的配置，
  不在执行期跟随 main。这里只提取几何，不声称采用了真实 checkpoint 输入分布。

AKA 来源索引（行号为本版本定位辅助；case_id 和 derived_parent_id 才用于实现时的唯一解析）：

| 引用 | v7 行 / derived_parent_id | 用途与限制 |
|---|---|---|
| A1 | 441 / `dd_matmul_bias_f32_bt_oc_c_block16_v1` | X[M,K] × W[N,K]ᵀ，可选 bias；原范围仅 FP32 |
| A2 | 446 / `gemm_abt_rowmajor_f16_f32acc_i32_tile16_v1` | FP16 ABᵀ 索引/输出契约；原资格限制部分和及最终值精确可表示，不代表一般 FP16 数值域 |
| A3 | 910 / `qkv_split_fp32_int32_contiguous_block256_v1` | packed QKV→三份 head-major 输出；纯 FP32 置换，三部分 head 数相等 |
| A4 | 925 / `attention_head_unpermute_contiguous_fp32_int32_block256_v1` | head/sequence 轴交换，辅助 O projection 的输入布局设计 |
| A5 | 935 / `causal_attention_contiguous_fp32_int32_tile32_v1` | FP32 causal self-attention；含 N=37、D=7 的尾部正确性案例；没有 GQA、bias、dropout |
| A6 | 951 / `dense_self_attention_bhsd_fp16_i32_hd64_v1` | 非 causal FP16 dense attention；D=64、S 为64倍数，不支持 GQA/MQA |
| A7 | 917 / `single_decode_attention_nhd_fp32_i32_hd128_v1` | D=128、单 query、NHD KV 的 FP32 decode；没有 batch 或 grouped heads |
| A8 | 915 / `paged_decode_attention_fp32_int32_contiguous_d128_sm100_v1` | 独立请求的页表、尾页及两份 cache；原范围 FP32、D=128、同 head 数 |
| A9 | 963 / `paged_attention_contiguous_fp32_int32_h64_b16_block128_v1` | head_mapping 与 paged addressing 的机制来源；包含 ALiBi 且分母额外加1e-6，不能直接用作标准 SDPA oracle |
| A10 | 999 / `silu_and_mul_contiguous_fp32_uint32_float4_block256_v1` | packed [M,2,I] 的 SiLU-and-multiply；原范围 FP32、I%4=0，无量化或路由语义 |

所有 AKA 条目定位在 [v7 records](https://github.com/qhy991/AKA/blob/a846b1c80a28939202a12cf64a8bf0aaf34ebefe/datasets/curated/cuda_kernel_parent_completions_v7/records.jsonl)。
设计阶段读取原 record/bundle；known-kernel reproduction 才可向作者提供其中低层实现。
clean-start 作者只获得数学规格、受许可的 oracle 与高层说明，低层参考保持黑盒；
不同作者环境的参考权限在 Study 中分别固定，不因参考了 qualified parent 就放宽。
使用时不复制另一份可变案例库。BF16、新 shape、GQA、组合接口
或新 target 都创建新的任务合同与资格，不继承 parent 的通过结论。

发现的 KernelEval 来源问题必须保留：

- `QG_73` 的 QKV 定义为 N=7168、K=2048，同时要求 N%3=0，描述又写3×2560=7680。
  7168%3=1，定义内部不一致。因此不采用它作为 QKV 几何/分段权威，也不把模型名
  当作维度证据。新 QKV 几何由官方 Hq/Hkv/D 推导。
- `fp32_flash_attention_qwen2_5_7b_f16_cache4096` 的 Q/K/V 都用28 heads，且 K/V
  没有独立请求的 batch 维。它是 mixed FP32/F16、共享 KV 的已声明任务，不能直接
  改名为 Qwen GQA 或 B 个独立请求。Llama cache512 定义也有类似的共享 KV 边界。
- QG_61/62/63/64 与 QG_74/75/76 的 dense 形状可作模型角色锚点，但原执行是
  Q4_0×Q8_1 风格量化。新 BF16 任务只继承形状，不继承 packed bytes、oracle 或 timing。

[KernelEval 当前定义目录](https://github.com/qhy991/KernelEval/tree/8cdf0b197ca0987f414ff1029b141727f9b4552a/definitions/cases_common_103)
是上述 task id 的源，不以论文 CSV 替代机器定义。

## 2. 六个任务边界

首版主线为 BF16 storage、FP32 accumulation；FP16 和 FP32 conformance 各自是独立
合同实例。所有输入只读、输出完整覆盖且不别名；不允许以已有输出内容参与计算。
Dtype、bias 有无和 layout 在合同生成时固定，不让候选在运行时自行选择另一条语义。
`RN_D` 表示舍入到合同声明的 storage dtype；不默认允许 TF32、FP16 accumulator 或量化。

### T1 — `llm_linear`：复用的线性投影

ABI：X[M,K]、W[N,K]、仅在合同声明时存在的 bias[N] → Y[M,N]，均 contiguous。
语义：Y=RN_D(FP32-accumulate(X Wᵀ)+bias)。W 的物理布局固定为 [N,K]，不再为每个
角色提供另一种默认转置形式。输出 BF16；LM head 的 FP32 logits 为另一个明确实例。

角色是 shape/profile 标签，不是独立算法或 Compiler route：

- QKV packed：N=(Hq+2Hkv)D，按 Q、K、V 顺序分段；bias 同样分段。
- O projection：K=HqD，N=hidden_size。输入为已经按 token/head 合并的 context。
- MLP gate/up：可以分别用 N=I 的控制实例，packed gate_up 用 N=2I。
- MLP down：K=I，N=hidden_size。
- LM head：K=hidden_size，N=vocab_size，输出所有被请求 token 的所有 vocabulary logits。
  不将 argmax、top-k 或 sampling 偷换为 logits 输出；权重 tying 不授权修改共享权重。

独立 reference 做高精度分块 matmul，再按预声明规则舍入。基线应直接接收同一
X/W/bias ABI；候选需要的转置、repacking 或精度转换不得移出计时边界。
来源 A1/A2 支持基础数学与方向，KernelEval 支持角色/形状；BF16 资格重新建立。

### T2 — `qkv_projection_heads`：投影与分段/置换融合

ABI：X[B,S,H]、Wqkv[(Hq+2Hkv)D,H]、可声明的 packed bias →
Q[B,Hq,S,D]、K[B,Hkv,S,D]、V[B,Hkv,S,D]，三份输出分别 contiguous。
逐元素定义：先完成 T1 的 packed 投影与 RN_D，再按 Q/K/V 分段写到对应 (b,h,s,d)。
三个段的起点由 Hq/Hkv/D 唯一派生，不另外输入可冲突的 offsets。

MHA 的三段等长，GQA/MQA 通常不等长。A3 的等 head 数纯置换是来源案例；不能把它
误当成 projection 或直接推广成三等分。参考由独立 matmul 与索引实现组合而成。
QKV 分段、head/token 次序使用非对称、逐 head 可识别输入检验。

范围不含 QK normalization、RoPE、cache append。这些边界以后各自有合同；本任务
不声称已经完成一个真实 Qwen attention layer。多输出融合收益由同 ABI 的完整序列比较。

### T3 — `gemm_swiglu`：门控前端融合

ABI：X[M,H]、Wgu[2I,H] → Z[M,I]。首版不带 bias；需要 bias 时生成独立实例。
按列前后两半定义：

    G = RN_D(X Wgᵀ); U = RN_D(X Wuᵀ)
    Z = RN_D(silu(FP32(G)) * FP32(U))

G/U 的 storage 舍入边界必须保留；不能用更高精度隐去它再声称等价。A10 是独立
激活数学的来源，T1 提供投影的数学，完整组合是新合同。

现有 `gemm_silu` 只消费一个累加器，不等于本任务。Compiler v72 的 epilogue pass
要求同形状 unary consumer，也不能直接覆盖 [M,2I]→[M,I] 的 gated consumer。
需要先检验成对累加器/分段访问能否表达，再决定 pass 的有边界扩展。

### T4 — `mlp_swiglu`：完整 dense MLP

ABI：X[M,H]、Wgu[2I,H]、Wd[H,I] → Y[M,H]。
语义：先按 T3 得到 Z，再 Y=RN_D(FP32-accumulate(Z Wdᵀ))。
范围不含输入/输出 norm、residual、MoE 路由、dropout 或 backward。

验证每个舍入点的独立参考，再验证完整输出；饱和 gate、抵消和 down 权重放大的
定向输入应能揭露错误的 gate/up 交换和中间舍入删除。诊断分解不改变计时的最终产物。
不要求完整 MLP 塞进一个物理 kernel。完整执行序列及其 workspace 都必须封存并计时。
这是后述多 kernel Evaluation 边界完成后才可准入的任务，不能伪装成已有单 kernel 任务。

### T5 — `sdpa_contiguous`：标准 dense MHA/GQA/MQA

这里 dense 指读取全部可见 K/V；MHA/GQA/MQA 是 head-sharing 维度，不是 sparse/dense
的互斥分类。用一个语义实现，Hq/Hkv 区分实例，不复制三份 oracle。

ABI：Q[B,Hq,Sq,D]、K[B,Hkv,Sk,D]、V[B,Hkv,Sk,D]、
q_lens[B]/kv_lens[B] INT32 → O[B,Hq,Sq,D]，数据 contiguous、D 首版为64或128。
约束：Hq%Hkv=0；长度不超过物理容量；每请求 q_len、kv_len 均正。
causal 实例还要求 kv_len>=q_len，query 对应 cache 最后的 q_len 个 token。

    group = Hq / Hkv
    kv_head(h) = floor(h / group)
    visible(b,q,k) = k < kv_lens[b]
    causal 时再要求 k <= kv_lens[b] - q_lens[b] + q
    score(k) = dot(Q[b,h,q,:], K[b,kv_head(h),k,:]) / sqrt(D)
    O[b,h,q,:] = RN_D(sum_k softmax_over_visible(score)(k) * V[b,kv_head(h),k,:])

mask 为 none 或上述 bottom-right causal，在合同内固定。无 denominator epsilon、
ALiBi、softcap、RoPE、dropout 或任意用户 mask。q>=q_lens 的 padding 输出必须写正零；
有效 query 不允许空可见集合。非法长度/非整数 head ratio 明确拒绝，不能静默修复。

MHA：Hkv=Hq；GQA：1<Hkv<Hq；MQA：Hkv=1。每个请求拥有自己的 K/V；需要共享 KV
的 KernelEval 原语作为单独的来源复现，不冒充独立 batch。为了实现方便而复制或
展开 KV 的成本必须计入候选，不能放在计时前藏起来。

oracle 使用独立高精度分块 QK/稳定 softmax/PV，可物化 expanded K/V 作正确性参考，
但 timed baseline 必须接受原始 Hkv ABI，或将其转换成本计入同一边界。
A5/A6/A7 分别支持 causal、noncausal 与 decode 语义来源；GQA/MQA、ragged batch
及 BF16 是新扩展，需重新资格验证。Qwen3 所需 QK norm/RoPE 应在本任务输入之前完成。

### T6 — `paged_decode_attention`：真实页表边界

ABI：Q[B,Hq,D]；分开的 K/V_pages[P,Hkv,T,D]；page_indptr[B+1]、page_indices[J]、
last_page_len[B] INT32 → O[B,Hq,D]。首版 D=128、T=16/64；buffer 布局固定。
逻辑长度由页数及尾页长度唯一派生，不同时传入另一份 kv_len 权威。

每请求至少一页；1<=last_page_len<=T；页表单调，indices 在 [0,P) 内。
逻辑页通过 indices 映射到物理页，head-sharing 按 T5 推导，数学使用标准稳定 SDPA。
允许只读共享前缀和重复物理页引用，按逻辑 token 次数参与归一化；不允许候选自行去重。
输出与任意输入/metadata 不别名。写 cache 的操作不包含在本任务。

A8 提供页表/尾页语义，A9 只提供 grouped head/不同物理布局的对照。A9 的 ALiBi、
分母1e-6和 packed K 布局均不直接继承。以独立页表展开+T5 数学作 oracle；所有
逻辑位置完整覆盖，尾页未使用元素写入强干扰值，检查它们不会影响输出。

## 3. 模型几何与主矩阵

第一轮选择两个尺寸特征不同、且有 KernelEval 角色锚点的 profile。

| Profile | H | I | Hq | Hkv | D | Vocab |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | 3584 | 18944 | 28 | 4 | 128 | 152064 |
| Qwen3-4B | 2560 | 9728 | 32 | 8 | 128 | 151936 |

Qwen2 QKV 有 bias、O projection 无 bias，依据
[Transformers v4.51.3 Qwen2 实现](https://github.com/huggingface/transformers/blob/v4.51.3/src/transformers/models/qwen2/modeling_qwen2.py)。
Qwen3 config 明确 attention_bias=false。首版 MLP 和 LM head 均按无 bias 的声明实例。
这些 profile 使用模型几何和确定性构造数据，不声称输入来自 checkpoint capture。

T1 每个 profile 选以下五个角色，M 分别为1、8、512，共2×5×3=30个固定 shape cell：

| 角色 | Qwen2.5 (K,N) | Qwen3 (K,N) | 形状依据 |
|---|---|---|---|
| QKV packed | (3584,4608) | (2560,6144) | 官方 Hq/Hkv/D 推导，不能用3H |
| O projection | (3584,3584) | (4096,2560) | K=HqD；Qwen2.5 对应 KernelEval QG_61 |
| gate_up packed | (3584,37888) | (2560,19456) | QG_63/QG_75 的单投影 I，组合后2I |
| down | (18944,3584) | (9728,2560) | QG_62/QG_74 |
| LM head | (3584,152064) | (2560,151936) | QG_64/QG_76 |

Qwen3 的 HqD=4096 不等于 H=2560；不能以方阵 O projection 替代它。
M=1/8 分别默认 B=1/8、每请求1 token；M=512 默认 B=1、S=512。
对纯 linear 数学 M 可展平，但对 T2/T5 必须保留真实 B/S 分解，不能将512 queries
共享一个 cache 解释为512个独立 decode 请求。

T2/T3/T4 各在两个 profile 的 M=1、512 上选 cell，共3×2×2=12个融合/组合 cell。
T5 采用下列四种 head 配置（MHA/MQA 是机制覆盖实例，不冒充 Qwen 模型配置）：

| 配置 | Hq/Hkv/D |
|---|---|
| MHA | 32/32/128 |
| GQA-g4 | 32/8/128 |
| GQA-g7 | 28/4/128 |
| MQA | 32/1/128 |

每种配置选择三种主工作量，共12个 attention cell：

- prefill：B=1、Sq=Sk=512，causal，全长有效。
- decode-small：B=1、Sq=1、Sk=4096，无额外 mask，全长有效。
- decode-batched：B=8、Sq=1、Sk=8192，kv_lens=[512,1023,2048,4095,4096,6143,8191,8192]。

另用 prefix/chunk cell（B=2、Sq=128、Sk=4096，q_lens=[128,96]、kv_lens=[4096,1024]）
验证 bottom-right causal；不把它与无前缀 prefill 混在同一 performance cell。
noncausal self-attention、D=64、Sq/Sk=37/65/129 的非整 tile 情形进入正确性/迁移集。
T6 待 T5 与页表执行能力通过后展开，至少覆盖乱序页、共享前缀、重复页和部分尾页。

主矩阵54个 cell是后续扩展范围，不是一开始就启动的实验预算。首个 pilot 只取8个：
Qwen3 的 QKV-M8、O-M1、gate_up-M512、LM-head-M1、T3-M512，加 MHA decode-small、
GQA-g7 prefill、MQA decode-batched。它同时覆盖用户要求的角色、head-sharing 和不同工作量。

## 4. 正确性、计时与失败语义

所有 shape/输入分布在作者开始搜索前固定。每个 cell 至少有一般随机、零/非零 bias、
抵消/混合幅值、舍入中点、可识别 head/segment 的结构化数据；零权重不能令全部输出
退化为零。模型尺度数据用按 K 缩放的权重控制有限范围，另设有限离群值压力分布。
输入生成器的幅值/缩放参数和有限输出域必须进入正式合同；不能把有限案例的通过
推广成任意有限数值都不会溢出。SiLU oracle 使用按符号分支的稳定表达式，softmax
先减最大值，避免复现既有大负值 exp 溢出缺陷。
未经明确授权不读取 AKA 的固定 datasets/test 来挑选这些任务。

设计阶段提出如下初始数值预算，须先由独立 oracle 与可信 baseline 完成资格审查，
才能写入正式 Workload；它们不是从 parent 继承的通过结果：

| 算术类 | 全输出逐元素检查 |
|---|---|
| FP32 conformance | abs_error <= 3e-4 + 3e-4*abs(reference) |
| FP16 storage / FP32 accumulation | abs_error <= 2e-3 + 2e-3*abs(reference) |
| BF16 storage / FP32 accumulation | abs_error <= 2e-2 + 2e-2*abs(reference) |
| 纯 layout/index/copy 子操作 | 对已舍入输入逐位相等；padding 正零、metadata 与输入不变 |

还必须证明以下错误控制会失败：漏 bias、恒零结果、Q/K/V 段交换、head/token 轴交换、
GQA 错用 h%Hkv、所有请求共用一个 KV、causal 错误对齐、读入尾页垃圾、漏写末尾 vocabulary。
若预算不能区分这些错误，或可信 baseline 未通过，先返回合同设计/数值诊断阶段；
不得开始搜索后为某个候选放宽。任何正式容差变更创建 successor。

MLP 的内部舍入是数学合同的一部分，采用定向放大输入检验；不能只检查几个最终 token。
正式 framework handoff 还需其自己的输出/轨迹 gate，算子通过不自动成为模型资格。

性能比较维护两个独立用途：同一冻结 Compiler 下的固定基线衡量搜索收益；相同 dtype/
ABI 的外部 incumbent 衡量实际替换价值。KernelEval 的旧 Q4 timing 不能成为 BF16 任务的
baseline，也不能混用旧 GPU 结果。计时范围覆盖完整输出以及候选特有的转换、repacking、
workspace 清理；不允许通过缓存上次结果或只写部分输出取胜。

使用现有 Open-Cake 配对 timing、CUPTI、冷 L2、噪声和 profiler 协议，由 CampaignLock
绑定具体执行，不复制另一套采样常数。正确性先于 timing，错误/不支持/噪声分别保留。
报告 operator/fragment latency，不把它命名为整模型 TPOT、TTFT 或 serving throughput。

## 5. 实现依赖和准入顺序

1. T1：复用现有 Workload/TensorABI 与 GEMM 基础，固定 [N,K] 权重方向和实际 dtype。
   不为 QKV、O、down、LM head 各写 Compiler conformance 或 profile table。
2. T2/T3：实现多输出不同 head 数和 gated consumer 的任务 oracle；独立探测分段访问、
   store 形状及舍入能否由现有 IR 表达。缺口要具体指向 primitive/分析/lowering。
3. T5：先实现独立数学 oracle和 MHA/GQA/MQA/causal/ragged 参数验证，再探测完整 Schedule。
   重点是 grouped-head 索引、有效长度、causal 域及长 K 的归约状态。不要改成共享 KV 或
   展开后的 Hq cache 来掩盖表达缺口，也不因任务名注册成功而标记 ready。
4. T4/T6：在前述能力及执行边界具备后接入。继续使用 matched_search，不增加 serving mode。

两个必须先解决的 Evaluation 依赖：

- 当前共同 CUDA 评测要求一次 sealed kernel launch。T2 的独立基线和 T4 的完整 MLP
  可能需要多个 kernel。应建立有确切入口、顺序、workspace 和 side effects 的封存执行
  序列并计时整个边界，复用现有 Program/评测经验；不能把隐藏多次 launch 的 wrapper
  报成 kernel_calls=1。该边界未通过前，T4 标为 implementation-blocked，不运行假基线。
- 真实 LM head 的全输出很大：M512×V152064 的 BF16 输出为155,713,536 bytes。
  现有 Python flat-list/JSON 传输不宜直接承接此规模。需要有界内存的 tensor/binary
  物化、分块独立 oracle 和完整输出验证；检查所有元素，不能用抽样或 top-k 替代。
  baseline/candidate 及 fresh-output cohorts 的峰值空间在 preflight 明确检查，失败则不运行。

每个任务只有在严格合同加载、输入生成、独立 oracle、source/target 准入、实际编译、
完整输出与 sanitizer、固定 baseline 和共同 timing/profiler 全部具备后才称 GPU-ready。
当前设计不预判这些门已通过，也不要求为一轮设计生成 Compiler/Executor Revision。

## English summary

The proposed suite has six semantic boundaries: shared linear projection, head-aware QKV
projection, GEMM+SwiGLU, complete dense MLP, contiguous SDPA and paged decode attention.
Projection roles share one mathematical owner; MHA/GQA/MQA share one attention definition
with explicit head grouping and bottom-right causal positions. KernelEval supplies deployed
shape/format regimes, AKA v7 supplies qualified but narrowly scoped parent semantics, and
official model configs supply head geometry. None transfers GPU or performance qualification.
The first pilot selects eight cells before expanding to a 54-cell core matrix. This is a design
proposal: no new runtime task, frozen contract or experiment is created by this document.
