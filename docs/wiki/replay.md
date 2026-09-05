# 怎样回放旧任务

**主线可以升级，旧实验仍使用它当时固定的完整源码。**
不要把旧 lock 放到新代码旁边，再把报错改掉。

[返回 Wiki](README.md) · [怎样读结果](results.md)

## 先分清三样东西

- **源码提交：** 当时的 Compiler、任务生成工具与合同文件。
- **运行环境：** Python、GPU 工具链和机器条件。
- **结果记录：** 原始输出、计时和验收记录。

找回源码并生成准备包，只完成第一步，不等于又跑了一次 GPU，也不会补出缺失的历史结果。
新版 Compiler 的说明即便只改了几句话，发布身份仍然会变化；旧任务不能假装跟着升级。

## 选择已经固定的源码

| 任务 | 完整源码提交 | 固定 Compiler | 可以重建什么 |
| --- | --- | --- | --- |
| state-store v5 | `c40bb399c7bebe026bcd30e27255dfc15ad23f5c` | v41 | 准备包和静态正反例检查；GPU 验收仍待执行 |
| FMA correctness | `d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d` | v41 | 普通与嵌套 FMA 准备包；历史结果另看其记录 |

这些提交是历史定位，不是建议把 main 回退到那里。
旧 state-store v4 的完整来源则是 `82491bb7658a4da92be444fa029b26b02635a154`，使用 Compiler v40。

## 一个不需要 GPU 的回放

先完成 [入门环境](../GETTING_STARTED.md)，在主仓库根目录创建新的外部工作区：

```bash
CAKE_REPLAY_DIR=$(mktemp -d)
git worktree add --detach "$CAKE_REPLAY_DIR/source" \
  c40bb399c7bebe026bcd30e27255dfc15ad23f5c
.venv/bin/python \
  "$CAKE_REPLAY_DIR/source/examples/gpu/state_store_b200_correctness/prepare_candidate.py" \
  --output-root "$CAKE_REPLAY_DIR/bundle"
```

这不会切换当前工作区的分支。命令使用旧工作区的真实 Compiler 源码，并核对固定 lock 和完整源码范围。
应看到 v41、正例被接受，以及两个状态所有权/轴错误被拒绝。

新目录中的 `candidate/provenance.json` 指明 Compiler 来源，`preflight.json` 是本次准备结果。
实际 GPU 部署和评测还没有发生。输出目录已存在时会拒绝覆盖。

## 回放 FMA 时

用表中的 FMA 提交创建同样的外部工作区，再调用其中的
`examples/gpu/fma_b200_correctness/prepare.py`。它另外需要：

- `--python`：准备包中评测进程要用的 Python 可执行文件。
- `--judge-cwd`：评测进程的工作目录。

只有检查准备步骤时，可以使用当前合适的 Python；真实 GPU 运行必须核对实际机器环境。
准备工具没有调用 GPU、provider 或 broker。详细任务范围见 [FMA 说明](../../examples/gpu/fma_b200_correctness/README.md)。

## 不要把“文件存在”当成“可回放”

浅克隆或源码压缩包可能没有上述 Git 对象；应取得明确提交的完整历史，而不是跳过来源检查。
混入另一个工作区的 Python 模块、改旧权限、换一份更宽容的标准答案，都不能恢复原实验的证据。
