# 用 Python 写计划：从一个乘加例子开始

[中文首页](README.md) · [English](../en/PYTHON_FRONTEND.md) · [完整接口约定](../PYTHON_FRONTEND.md)

**现在也可以用 Python 的变量和表达式写 Schedule，再交给原来的编译器检查。** 它是同一份计划的另一种编写入口；JSON 仍是规范保存格式，计算规则和 GPU 判对标准没有改变。

## 1. 先认清它在做什么

像用表格或文字填写同一张分工单，两种写法最后都要说明数据、角色、操作和地址。Python 前端先读语法，构造已有 Schedule，然后进入同一个 Verifier 和代码生成器。可以写出一种形式，不代表每个后端都能执行它。

CLI 读取 `.py` 文件时只解析语法树，不真正执行其中的函数、导入或任意 Python 代码。这里的 for 表示计划里的循环，不是在 CPU 上先跑一遍循环。

## 2. 完整 FMA 例子

下面来自仓库的 [FMA 示例](../../examples/python/fma.py)，计算每个对应位置的 `a*b+c`：

```python
from open_cake_ir.compiler import frontend as cake


@cake.schedule(name="fma-b8-smoke", target="sm_100a", backend="triton",
               entry_point="cake_fma_b8_smoke",
               residency={"ctas_per_multiprocessor": 4, "registers_per_thread": 64})
def fma(lm, a: cake.Tensor((8, 128), "fp32"), b: cake.Tensor((8, 128), "fp32"),
        c: cake.Tensor((8, 128), "fp32"), y: cake.Tensor((8, 128), "fp32", mode="output")):
    compute = lm.role(warps=[0, 1, 2, 3])
    batch = lm.program(a, axis=0, dimension=0, tile=1)
    with compute:
        a_tile = lm.load(a[batch, :], reuse="streamed", id="load_a")
        b_tile = lm.load(b[batch, :], reuse="streamed", id="load_b")
        c_tile = lm.load(c[batch, :], reuse="streamed", id="load_c")
        y_tile = lm.fma(a_tile, b_tile, c_tile, id="fma")
        lm.store(y[batch, :], y_tile, id="store_y")
```

先看函数中间几行：三次 load 读取 a、b、c 的同一行，fma 做融合乘加，store 写回 y。`batch` 是行号，冒号表示这一行的列。

再看外面的说明：`cake.Tensor((8,128),"fp32")` 表示八行、一百二十八列的 FP32 数据；y 标为 output。`compute` 是计算角色，with 里的操作交给它。最上方装饰器写目标、后端、函数入口和资源约定。

`lm.fma(a,b,c)` 要求最终只舍入一次。`a*b+c` 会构造独立乘法与加法，不能把两种数值约定当作同一种写法。参见[基本操作](../wiki/primitives.md#elementwise)。

## 3. 不用 GPU，检查并生成源码

先有[入门教程](../GETTING_STARTED.md)里的项目环境，在仓库根目录运行。这里使用正式发布 lock，新输出放在临时目录：

```bash
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler assess \
  --revision compiler/revision.lock.json examples/python/fma.py --format text

CAKE_PYTHON_OUTPUT=$(mktemp -d)
PYTHONPATH=src .venv/bin/python -m open_cake_ir.cli compiler lower \
  --revision compiler/revision.lock.json examples/python/fma.py \
  --output "$CAKE_PYTHON_OUTPUT/fma.py" --format text
```

应该看到“结构检查：通过”和“生成代码：允许”，随后得到 Triton 源码位置。它没有启动 GPU，也没有验证数值或速度；已有同名输出会被拒绝。

## 4. 最容易误解的规则

- 每个变量只赋值一次。读写关系自动形成依赖；跨角色等待、信号和流水线仍要明确写。
- Buffer 参数声明输入、输出或调用者状态。临时结果的形状和类型由操作推导，不随意贴标签。
- 一个下标对应一个维度。首个接口支持 program 坐标、loop tile 和连续静态切片；不支持的整数、间隔切片、动态间接索引或省略维度会拒绝。
- 切片只能作为操作地址，不能拿切片冒充 lm.program、lm.range 或 lm.broadcast 所需的 Buffer 本身，也不能嵌套下标。
- `lm.broadcast` 明示已有广播轴，不增加 splat 或 reshape。归约写明 sum/max、轴和协作范围。
- `out=buffer` 或多个输出 Buffer 指定结果后，不再把这个操作赋给另一个变量。普通输出来自 Buffer mode，没有另加 return 指令。
- `lm.range` 创建符号循环，不能换成任意 Python range、while 或 if。默认 stages 和展开因子为 1，其他布尔选项为 false，性能相关选择应明确写。

## 5. 更多例子与错误位置

[Softmax](../../examples/python/softmax.py)展示行归约和显式广播；[CuTe 流水线](../../examples/python/kmeans_pipeline.py)展示存储、角色、同步和嵌套循环。它们沿用已有任务形状，不继承旧实验的正确性成绩。

CLI 的诊断保留原 Finding code/path，并指出 Python 文件的行列。程序调用者使用 `read_schedule` 得到计划及伴随位置表，再交给 Compiler；完整 API 在[接口约定](../PYTHON_FRONTEND.md)。位置表只帮人找错误，不改变固定计划或生成源码的身份。

正常 Python import 仍可能执行模块顶层代码和参数表达式。因此接收候选的工具使用只解析的 `read_schedule`，不能把普通导入当成同样的边界。

## 类型转换与源码诊断

`lm.cast(x, to="fp32")` 使用规范 `to` 参数并自动推导结果 dtype；显式 `out=` 仍可用。
同 dtype 算术保持类型，FP32 与一种 16 位浮点混合得到 FP32，BF16/FP16 混合需显式 cast。
两种操作数顺序使用同一提升规则。仅支持 CTA 的归约 `scope` 默认 `cta`。
参见 [cast](../../examples/python/cast.py) 与[混合 dtype](../../examples/python/mixed_dtype.py) 示例。
诊断指向输出参数、装饰器关键字或最近的程序轴声明；构造错误使用 Python 名称，
`FrontendError.canonical_path` 保留规范路径供工具关联。
