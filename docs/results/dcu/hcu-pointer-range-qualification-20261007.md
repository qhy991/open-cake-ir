# DCU 冻结张量跨度：Compiler 主机调用优化

这轮优化把已验证的连续张量跨度传给 DTK，减少每次启动时的重复推导。
相同 Schedule 的完整调用加速约 1.09–1.10 倍。收益属于生成的主机接口；GPU 算术和 kernel AST 不变。
这份记录是两道固定开发 Task 的对照，不构成独立 Bench 或三小时搜索能力的结论。

## 机制与边界

原路径先验证 shape、dtype、device 和 contiguous，再由 HCU Triton 重算每个张量的物理字节跨度。
实际 DTK 的 `HIPBackend.is_within_4gb` 优先读取参数的 `ptr_range()`；否则计算
`(sum((shape_i - 1) * stride_i) + 1) * element_size()`。对已通过检查的连续张量，
这一值就是 Schedule 声明的 `elements * itemsize`。它不包含 backing-storage 的前缀，也不乘 staged allocation 数。

Compiler 在原检查之后新建轻量参数，保留原 tensor 引用、dtype 和该字节跨度。
`data_ptr()` 每次转发到当前 tensor。地址、对齐、设备、stream 和 native launcher 均不被缓存或替换。
DTK 自己保留 `<= 2**32 - 8` 的范围判断、当前地址的对齐特化和 `hipPointerGetAttribute` 检查。
这不是新 ISA、布局代数或自动选取 tile 的策略。只有已资格的 gfx938 路由采用它。

实现 `3aa81aaf`，共享 PR [380](https://github.com/qhy991/open-cake-ir/pull/380)。
生成源资格固定在 `7926c279`；补充组件生成源在 `fef90175`。
后续 `32f26850` 仅缩小名称冲突策略并补录证据，合法 Schedule 的生成源未变。
[Finding F-2026-10-07-007](../../../findings/2026-10-07-007-hcu-frozen-pointer-range.json) 维护提升决定。

## 固定对照

证据根目录：`/data3/testuser01/experiments/bw1100-transpose-comparison-20261007/run`。
正式生成源的记录在 `probe/generated-range/`，运行脚本和生成源在 `kit-generated-range/`。
`probe/manifest.json` 保留先前源码的来源；原 `d0d0acab` Workload、oracle 和容差沿用不变。
两题均为 M1024 / K256 / N64、FP32，比较未改动 kernel 的直接调用和新 Compiler 调用。
Torch 参考为 `torch.addmm`、`F.silu(torch.addmm(...))`，设置最高 FP32 精度并禁用 TF32。

计时边界为完整 callable 的主机墙钟，加调用后的设备同步；调用前同步位于区间外。
采用连续同实现调用块：每块预热 10 次，测量 30 次，保留正反次序。
三个实现各有同函数 A/A 重复块。每个 Task 的全部五个原始输入案例均保留。
同一 HCU5 上执行两次预定的独立 admission。表中每个案例先取正反方向较小的比值，再对五个案例取几何平均。

| Task | 直接调用 / 新调用，第1次 | 第2次 | Torch / 新调用，第1次 | 第2次 | A/A 最大差异，第1次 / 第2次 |
|---|---:|---:|---:|---:|---:|
| GEMM | 1.0994 | 1.0922 | 0.9317 | 0.9244 | 1.70% / 1.09% |
| GEMM+SiLU | 1.1000 | 1.0916 | 1.0083 | 1.0056 | 0.89% / 0.76% |

新调用相对直接调用的延迟减少约 8.4–9.1%。GEMM 仍比 Torch 慢约 7–8%；
GEMM+SiLU 与 Torch 基本持平，微小差异接近 A/A 波动，不据此声称超越社区基线。
这些数据不估计其他形状、混合调用顺序、其他 Task 或 agent 固定预算搜索的收益。

早期 `probe/paired1`、`paired2` 的混合调用协议存在显著顺序偏差。
同实现控制及独立的稳态协议保存在 `probe/aa-controls`、`probe/steady-protocol.json`。
这些原始记录保持不变，未与本表合并。`probe/static-range/` 是先行主机工程原型，
也没有计入正式 Compiler 成绩。

## 正确性、归因与释放

- `qualification/summary.json`：两题 × 五案例 × 三实现，共 30/30 检查通过。
- `guards/summary.json`：10/10 调用场景和 14/14 错误 ABI 启动前拒绝；覆盖偏移、重绑定、输出和 stream。
- `components/summary.json`：18/18 通过。包含 FP32、FP16、BF16、INT32、FP8 的位模式复制，
  同一对象重绑定到不同内容、多输出顺序和原子 state 更新。该补充关闭了仅换地址但内容相同的测试盲点。
- `steady1/summary.json`、`steady2/summary.json`：每次计时后再次通过全部 30 个原题检查。
- `trace/manifest.json`：六个 Task/实现组合各 20 次调用；共 140 个匹配实际 kernel 名称的 dispatch。
  GEMM 的直接/新调用 kernel 中位数均约 14.40 µs；SiLU 约 14.56/14.08 µs。
  这是仪器化归因，未用于本表的完整调用成绩。
- `host-profile/`：每题 300 次仪器化调用产生 1200 次常数范围查询、0 次物理跨度重算。
  原路径同等调用有 1200 次重算。Profiler 时间不作为未仪器化延迟。
- 七份 `*-admission-terminal.json` 均记录 exit0、HCU5 显存0%、无可见 KFD context、无运行容器。
  现有网关负责用户内串行化，未宣称能排除其他用户的所有物理活动。HCU0 的其他负载未动。

199 个 Corpus 的语义决策和诊断保持不变。20 个 HCU 主机源码快照显式采纳，kernel AST 全部不变，
总计 111 个 lowering 快照通过。独立评审覆盖源码、原 DTK 协议及名称作用域边界。
本地 `fef90175` 全部契约 2930 passed / 26 skipped；最后的 namespace 小修通过定向检查。
共享与硬件 PR 的精确提交 CI 负责集成验收。原实验没有重启、修补或重新归类。

提升项是这一个已测得收益的 gfx938 主机参数机制。后续固定预算实验应冻结新的 Compiler 提交，
单独检验 agent 是否能在同样时间内找到更好的 kernel；本轮结果不提前回答那个问题。
