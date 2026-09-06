# 第一次使用：先看懂一个乘加计划

[中文首页](zh-CN/README.md) · [English](en/GETTING_STARTED.md) · [中英文对照](README.md)

你只需要会打开终端。本教程检查并生成代码，**不需要 GPU，也不会调用 AI 或提交实验**。
看完后，你应能说清楚：输入是什么、怎样算、检查有没有通过、下一步还缺什么。

[返回 Wiki](wiki/README.md) · [下一课：读懂执行计划](wiki/schedule.md)

## 1. 准备项目

克隆仓库，然后进入目录：

```bash
git clone https://github.com/qhy991/open-cake-ir.git
cd open-cake-ir
python3 --version
```

需要 Python 3.10 或更新版本。若显示 3.9 等旧版本，先换用已安装的新版本命令（如 `python3.12`），再创建环境。
下面创建项目自己的 Python 环境，并安装项目及检查所需的依赖：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[test]'
```

已有合适环境时直接使用它。下文命令都在仓库根目录运行。

## 2. 先理解计算

本例计算 `y = a × b + c`。例如 `a=2、b=3、c=4`，答案就是 `10`。
实际文件一次处理 8 行、每行 128 个数，对应位置分别做这个计算。

输入 `a、b、c` 不被修改，答案写到 `y`。
采用 FP32 浮点数；融合乘加 FMA 只在最终结果处舍入一次。
这和“先把乘积舍入，再相加”可能有不同答案，见 [FMA 说明](wiki/primitives.md#elementwise)。

完整计划在 [`fma-b8-smoke.json`](../corpus/schedules/fma-b8-smoke.json)。暂时不用自己写 JSON。

## 3. 检查计划

```bash
.venv/bin/open-cake-ir compiler assess --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/fma-b8-smoke.json
```

关注两行：

```text
结构检查：通过
生成代码：允许
```

它们分别表示计划满足已建模的规则、所选后端能够生成源码。
可能出现 `[提示] RESIDENCY_BOUND`：这是一条资源分析提示，不是拒绝计划，也不是性能成绩。
诊断后面的 `roles`、`operations[3]` 等是 JSON 中的位置。

`--format text` 便于人读；去掉它，会输出供工具使用的完整 JSON，包括原始诊断和分析。
详细解释见 [怎样读结果](wiki/results.md)。

## 4. 生成源码

新文件放到仓库外，避免和源文件或旧实验混在一起：

```bash
CAKE_TUTORIAL_DIR=$(mktemp -d)
.venv/bin/open-cake-ir compiler lower --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/fma-b8-smoke.json \
  --output "$CAKE_TUTORIAL_DIR/fma.py"
```

终端会给出文件位置和入口函数 `cake_fma_b8_smoke`。
打开生成的文件，可以看到读取数据、执行 `fma.rn.f32`、写回结果等步骤。
若输出文件已经存在，命令会拒绝覆盖；需要另选一个新文件。

此时得到的是 Triton 源码，还不是 GPU 二进制，也没有验证 GPU 答案。
**请不要把“源码已生成”写成“GPU 测试已通过”。**

## 5. 故意看一个错误

仓库已有一个“FMA 少了输入”的反例：

```bash
.venv/bin/open-cake-ir compiler assess --format text \
  --revision compiler/revision.lock.json \
  corpus/schedules/fma-b8-smoke-arity-drift.json
```

这里应出现“结构检查：未通过”，并返回非零退出码。
它验证编译器会拒绝错误计划，不是教程失败。原来的正确计划没有被修改。

最后检查整套编译器语料：

```bash
.venv/bin/open-cake-ir compiler check-corpus --format text \
  --revision compiler/revision.lock.json
```

“符合预期”既包括正确计划被接受，也包括错误计划被拒绝。
数量以当前输出为准，不需要从旧文章复制。

## 6. 下一步

- 想改计划：[逐项解释这份 JSON](wiki/schedule.md)。
- 想换一个计算：[算子图解](wiki/operators.md)。
- 想跑 GPU：[实验流程](wiki/experiments.md)，先核对当前 Executor、机器和任务合同。
- 想查看历史 GPU 教学结果：[Flash-KMeans 教学验收记录](../inventory/GPU_QUICKSTART_QUALIFICATION_V3_20260823.json)。它绑定当时的版本，不能当作当前环境的使用说明。
