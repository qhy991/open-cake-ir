# Triton TileLoop 作用域

Compiler 拥有本次源代码 lowering 契约；现有 TileLoop body 是唯一控制流表示。
R1 支持零层、单层，以及静态两层嵌套：外层遍历输出块，内层归约或沿 K 累积
MMA，外层消费内层结果并写回。根作用域和外层的不变量按声明位置执行。
不增加 IR primitive 或 authoring mode。

展开子循环后必须保持 operations 的连续声明顺序。循环坐标只在本层和子层
可见。归约结果只越过其直接循环边界，不自动越过外层输出循环；归约 identity
和 MMA 零累加器按结果 shape 在所属循环开始前初始化。

本轮不支持并列循环树、更深嵌套、嵌套动态 stop、flatten、warp specialization、
嵌套 argmin/top-k/online-softmax 或沿祖先循环累积的 MMA；preflight 定位拒绝。
嵌套 MMA 的两输入须直接由 load 产生，以便既有 access-map 查询证明 K 累积轴。
每层 tl.range 保留自己的调度选项，不合并成一个全局 stage 参数。
循环内 scan 继续拒绝，因为现有 primitive 没有跨块 prefix carry。循环内普通
store 只支持 output buffer，并要求 affine 坐标唯一覆盖所有活动循环和 program
轴。indexed/state store 保持既有边界，不增加 role、TMA 或 TMEM transfer 支持。

验收使用 CPU 契约检查、生成源码的测试专用张量解释执行和独立标量参考，及
单层源码回归。它们不证明 Triton 编译、GPU 正确性、性能、profiler 或目标框架
验收；这些证据属于 R2。Corpus 采纳、完整 Gate 和独立批准的发布由集成负责。
