# 用 Python 编写 Schedule

本接口由 Compiler 维护。目标是让作者用变量、表达式、角色作用域和分块循环编写现有 Schedule，
再交给同一个类型模型、Verifier 和后端。JSON 仍是规范 Schedule 的序列化格式。
前端不扩展算子语义、后端覆盖或 GPU 正确性结论，也不改变已冻结实验的作者环境。

## 最小合同

- 一份源文件包含一个 `@cake.schedule(...)` 函数；参数用 `cake.Tensor` 声明固定形状、dtype 和 mode。
- `lm.role`、存储声明、`lm.pipeline` 和 `lm.barrier` 对应现有资源；`with role` 指定操作所属角色。
- `lm.program` 指定并行分块；`for tile in lm.range(...)` 构造一个 TileLoop，不在宿主 Python 中展开。
- `lm.load`、计算表达式、`lm.store` 建立数据依赖。跨角色同步仍须显式声明 waits、signals 和 pipeline。
- `lm.fma(a, b, c)` 保留现有单次 RN-even 舍入合同；`a * b + c` 构造两个独立操作。
- 广播通过 `lm.broadcast(value, axis=...)` 明示使用现有 `broadcast_axis`，不会增加 splat 或 reshape。
- CLI 解析 Python AST，不执行源文件、导入、函数体或任意 Python 控制流。未支持的语法会带源码位置拒绝。
- 语义与后端合法性仍由现有 Compiler 决定。源码位置只用于诊断展示，不加入规范 Schedule 或 Assessment。

## 验收边界

FMA、softmax 和现有 CuTe 流水线示例必须生成与既有 JSON 语义相同的规范 Schedule，复用同一源码生成路径。
错误形状、类型、同步与后端条件必须保留原有拒绝，且能定位回 Python 文件。
必须覆盖符号循环未被展开、显式 FMA 与乘加分离、拒绝执行宿主副作用，以及 JSON CLI 的兼容性。
Compiler 变更需通过完整 Corpus Gate；后继正式发布仍消费外部审批，不能改写旧发布记录。

## 运行

使用项目的 Python 3.10 或更新环境。在开发分支中，`compiler/revision.json` 指定当前待审草案；
正式发布后使用 `compiler/revision.lock.json`。已有发布锁不适用于改过绑定源码的开发分支。

```bash
PYTHONPATH=src python3 -m open_cake_ir.cli compiler assess \
  --revision compiler/revision.json examples/python/fma.py --format text

PYTHONPATH=src python3 -m open_cake_ir.cli compiler lower \
  --revision compiler/revision.json examples/python/fma.py \
  --output /tmp/cake-fma-generated.py --format text
```

输出文件采用新建语义；已有同名文件会被拒绝，避免覆盖原始产物。
这两步只检查和生成源码，不编译 GPU 二进制、不分配 GPU，也不运行数值测试。

- [FMA](../examples/python/fma.py)：对应位置的三输入融合乘加。
- [Softmax](../examples/python/softmax.py)：读取、行归约、显式广播和写回。
- [CuTe 流水线](../examples/python/kmeans_pipeline.py)：共享存储与 tensor memory、四种角色、同步和两层符号循环。

例子沿用既有算子的固定形状。新例子的名称和元数据不继承旧实验的身份或正确性结论。
`id=` 仅用于显式命名操作、与既有计划对照；省略时由结果变量或目标 Buffer 推导。

## 编写规则

`cake.Tensor(shape, dtype, mode="input")` 声明 global Buffer；输出参数写 `mode="output"`，
调用者拥有的可变状态写 `mode="state"`。中间寄存器结果的形状和 dtype 从操作推导。
`lm.buffer` 声明需要显式命名的结果或暂存区域，`lm.smem` / `lm.tmem` 声明存储，
存储的 `.view(...)` 保留 `byte_offset`、`stages`、`swizzle` 等具体承诺。

读取使用 `lm.load(x[program, :], ...)`。一个下标对应一个维度：程序坐标、循环 tile，
或连续的静态切片。首版不接受整数坐标、间隔切片、动态间接索引或省略维度，
也不把它们近似成现有 AccessMap。TMA 的显式结果、descriptor_box 和同步信息仍须写明。
下标表达式只作为操作地址使用；不能再对它进行嵌套索引，或把它当作 `lm.program`、`lm.range`、
`lm.broadcast` 所需的 Buffer 名称。这些位置会明确拒绝切片，不会丢弃其范围。

算术可写 `a * b + c`、`x * 2.0`，或现有数学原语调用，例如 `lm.exp(x)`、`lm.fma(a, b, c)`。
有广播时在第二操作数上写 `lm.broadcast(row_value, axis=0)`，或在数学调用中明确 `broadcast_axis`。
归约使用 `lm.reduce(x, op="sum", axis=..., scope="cta")`；需要多个结果或目标存储的操作用
`out=buffer` 或 `out=(buffer1, buffer2)`。显式 `out` 不再赋给另一个变量。
`lm.store(destination, value)` 写回；没有独立的返回指令，输出集合由 Buffer 的 mode 决定。

变量只赋值一次。前端从读写关系推导依赖，包含覆写前的读取；`depends_on` 可补充已有操作 id，
但它不能代替跨角色同步。`waits`、`signals`、`pipeline` 仍由作者指定并由 Verifier 检查。

`lm.range(buffer, dimension=..., tile=..., name=...)` 声明循环，循环体里的操作只构造一次。
其 `num_stages` 和 `loop_unroll_factor` 默认为 1，其余现有布尔 range options 默认为 false；
性能相关的非默认选择应显式书写。它不等同于任意 Python `range`、运行时 while 或条件分支。
前端可表达嵌套循环和多角色，并不意味着每个后端都允许这些组合。

## Python API 与诊断

```python
from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.frontend import read_schedule

compiler = Compiler.load(".", "compiler/revision.json")
authored = read_schedule("examples/python/fma.py")
assessment = compiler.assess(authored.document)
for finding in assessment.findings:
    print(finding.code, finding.path, authored.location_for(finding.path))
lowering = compiler.lower(assessment)
```

`Compiler.assess_file` 同样接受 `.py`，返回的 Assessment 与对应 JSON 的完全相同。
需要源码位置的调用者使用 `read_schedule` 返回的伴随位置表。CLI 的 JSON 诊断增加 `source`
（文件名、起止行列），原有 Finding code/path 不变；文本输出显示相同位置。
源码构造错误抛出 `FrontendError`，并携带 `PYTHON_SYNTAX` 或原模型的 `SCHEDULE_STRUCTURE` 代码。
错误的 Python 候选不能生成输出文件。

也可以正常导入示例：装饰器把函数转换成 `ScheduleSource`，不调用函数体。
正常 Python 导入仍会执行模块顶层和参数表达式，因此接收候选源码的工具应使用 `read_schedule`，
让 AST 入口拒绝任意导入、宿主调用、条件分支和其他不支持的语法。
位置表是展示用投影，不进入冻结 Schedule、Assessment 或生成源码的身份。
