# MetaX C550 接入调查与实施顺序

2026-09-20，基于 `main@f09a3e85`，分支
`codex/metax-c550-support-20260920`。本次完成了 `c550-1` 实机属性调查、
开发镜像查找和只查询设备属性的工具。**尚未接通 Compiler / Executor，
也没有 GPU kernel 正确性或性能结论。**

后续实现与固定范围验收见 [C550 当前可用路径](metax-c550.md)。本页和
[初始任务可运行性检查](metax-task-readiness.md) 保留接入前的调查结论。

## 本次可复用的交付

`tools/probe_metax_host.py` 使用系统 Python 标准库，采集 `mx-smi`、
`macainfo`、编译目标工具、`mxcc --version` 和当前 Python 包的可见性。
它还在临时目录用系统 C++ 编译一个普通主机程序，通过已安装 SDK 的
`mcGetDeviceCount` / `mcDeviceGetAttribute` 查询每张卡。
枚举由 SDK 头文件解释，不复制 CUDA/HIP 的数值或设备结构体 ABI。
没有设备内存分配、kernel launch、计时、软件安装或设备配置变更。

从 checkout 运行，输出保存到 checkout 外：

```sh
ssh -o BatchMode=yes c550-1 python3 - < tools/probe_metax_host.py > /tmp/metax-host-survey.json
```

退出码 0 表示这些查询成功；缺少 PyTorch/Triton 会明确记录，但不使设备查询
本身失败。命令失败、API 错误或超时返回 2，保留 stdout/stderr，失败属性为
`null`，不补零。退出码从不代表平台可用性或 Executor admission。

本次原始观测留在 checkout 外：

`/Users/haiyan-infiniai/Agent4Kernel/open-cake-ir-evidence/metax-c550-20260920/20260920T055904Z-host-survey.json`

## 已观察到的硬件与软件

8 张卡的 18 项 runtime 属性均查询成功且一致。下表是上述原始记录的投影，
不是第二份 Target 文档；正式接入后硬件常量只归 Target 所有。

| 项目 | 观察值 | 来源 / 用途 |
| --- | --- | --- |
| GPU / 显存 | 8 × MetaX C550，65536 MiB/卡 | `mx-smi`；观察时无用户进程，不代表分配授权 |
| CPU / OS | x86_64 / Linux 5.15.0-112-generic | SSH 主机调查 |
| KMD / MACA | 3.6.11 / 3.5.3.18 | `mx-smi` |
| 设备 ISA | `METAX-MXC-MXMACA--XCORE1002` | `macainfo` |
| 编译架构工具 | `xcore1000`，共 8 行 | `__metaxgpu_arch`；必须与设备 ISA 分开 |
| 编译器 | mxcc 1.0.0，build `6477545d4d` | `/opt/maca/mxgpu_llvm/bin/mxcc --version` |
| warp / wave | 64 / 64 | runtime 两个属性均为 64 |
| 计算单元 | 104 | `MultiProcessorCount` |
| 每 CTA 最大线程数 | 1024 | `MaxThreadsPerBlock` |
| 每维 block 上限 | 1024 / 1024 / 1024 | runtime；还受总线程上限约束 |
| 每维 grid 上限 | 2147483647 / 2147483647 / 2147483647 | runtime；`macainfo` 报 4294967295，不采用后者作为 launch 限制 |
| 每 CTA / 计算单元共享内存 | 65536 / 65536 bytes | runtime |
| 每计算单元最大线程数 | 2048 | runtime |
| 每 CTA / 计算单元寄存器 | 131072 / 131072 | SDK 定义的 32-bit register 计数；未校准 occupancy |
| L2 | 8388608 bytes | runtime；未验证 flush 方案 |
| MACA capability pair | 10 / 2 | 原生 MACA runtime；不是 NVIDIA compute capability，也未验证容器内 PyTorch 的兼容 API 返回值 |
| 系统 Python | 无 torch、triton、numpy | 本次 Python 环境范围，不声称全盘不存在其他环境 |
| 容器运行时 | containerd / nerdctl / crictl | 已列出的容器和镜像均为集群系统组件 |

SDK 存在 `libmcruntime.so`、`libmcpti.so`、`mcTracer`、`mcpti` 头文件及
CUDA/HIP 兼容头文件。这只证明接口文件存在，不能据此宣称拥有 dispatch
计时或 profiler evidence。也没有带宽、算力或指令数值精度标定。

## 可获取的开发镜像

首选调查对象：

```text
harbor.baai.ac.cn/flagtree/flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04:202604-0.5.1
```

[FlagTree 的 C550 安装文档](https://docs.flagos.io/projects/FlagTree/en/latest/getting_started/multi-backend-prebuilt-docker-image-install/install-metax.html)
列出此镜像：Python 3.12、PyTorch 2.8.0、MACA 3.5.3.x、Ubuntu 22.04，
预装 FlagTree 0.5.1 的 MetaX 后端（文档标记基于 Triton 3.0）。它与本机
MACA 版本系列匹配，因此优先用于接入验证；精确的 KMD/镜像兼容性仍需容器
内设备枚举验证。

从 `c550-1` 对 registry 的只读查询得到 HTTP 200、`linux/amd64`、22 层、
压缩层合计 **8749346904 bytes**。镜像未拉取，包版本尚未在容器内复核。
文档列出的镜像体积是 28.1 GB；它与 registry 压缩层字节数不是同一口径。

[同版本离线包](https://baai-cp-web.ks3-cn-beijing.ksyuncs.com/trans/flagtree-metax-py312-torch2.8.0-vllm0.15.0-metax3.5.3.x-ubuntu22.04.202604-0.5.1.tar.gz)
的 HEAD 也返回 HTTP 200、Content-Length **8644051579**，未下载归档。
查询记录：

`/Users/haiyan-infiniai/Agent4Kernel/open-cake-ir-evidence/metax-c550-20260920/20260920T060016Z-image-lookup.json`

另查到 [MetaX 官方 vLLM 0.24.0 镜像](https://vllm-metax.readthedocs.io/en/latest/release/0.24.0/index.html)，
使用 MACA 3.8.2.5 / PyTorch 2.10，版本系列与此主机不同。
`cr.metax-tech.com` 的 manifest 和 tag API 本次均返回 401，未验证可拉取性。
当前接入不需要为 vLLM 升级宿主驱动。

镜像准备阶段应使用独立的 containerd namespace 和工作目录，先检查磁盘余量，
再拉取镜像。先在无 GPU 的容器内核对包与离线编译；设备查询使用最小必要的
MetaX device 挂载。不要覆盖镜像自带 `/opt/maca`，否则调查到的 SDK 与实际
编译插件可能来自两套版本。完整 GPU 验证随后走平台分配和验收流程。

## 接入需要修改的 owner

建议先复用 Schedule 的 `triton` lowering，完成 FP32 elementwise / reduction
这一条闭环。新平台需要自己的能力判定、编译产物适配、运行时和测量实现；
是否增加独立 `LoweringBackend`，由容器里实际可复用的 emitter 语义决定。
不复制完整 Triton emitter，也不先实现 native MACA C++、矩阵指令或多卡路线。

| 顺序 | 归属 / 改动 | 通过条件 |
| --- | --- | --- |
| 1 | 独立开发容器 | 记录实际 torch / triton / plugin / mxcc，读取 `GPUTarget` 和编译产物；宿主 SDK 信息不能代替容器信息 |
| 2 | `compiler/target.py`、`compiler/targets/` | 显式 MetaX vendor、实际 code object、精确设备身份；本次 runtime limits 成为唯一硬件声明，未测 contract / peak 不填 |
| 3 | `compiler/backends/triton.py`、`compiler/toolchain.py` | MetaX 自己的 preflight 与 code-object route；无 GPU 句柄的 AOT 编译、exact-target 检查、64-lane 线程数、shared/scratch metadata 均由实际产物验证 |
| 4 | `evaluation/platforms.py` 及 MetaX runtime 模块 | 注册实际产物、manifest ABI、加载/卸载/launch 和资源报告；目标、参数顺序、地址、dtype/shape、错误清理均可拒绝，不经过 CUDA/HIP driver |
| 5 | `lab/executor.py`、`lab/runtime_config.py`、`tools/capture_executor_host.py` | 平台独有 host kind、SDK/plugin/library admission、编译 jail 闭包和设备分配；再生成唯一 `runtime/hosts/<target>.json` |
| 6 | `tasks/evaluate.py` 的平台注册、MACA 测量模块 | 有资源分配后验证外部 oracle；随后验证 MCPTI 或另一原生 timer 的 dispatch 边界、单位、分辨率、每样本 reset 和 profiler 关联 |
| 7 | Corpus 与对应 contract tests | 正向/负向案例、错误 target、64-lane 边界、unsupported ops、跨平台路由拒绝；全 Corpus Gate 在干净 commit 的独立 worktree 运行，合入 main 时独立 review |

这里有两处已经确认的共享层假设需要处理：

* `toolchain.triton_route` 目前只允许 CUDA route 使用整数 architecture。
  [FlagTree 当前 MetaX driver 源码](https://github.com/flagos-ai/FlagTree/blob/main/third_party/metax/backend/driver.py)
  使用 `GPUTarget("maca", integer_capability, 64)`。整数架构并非 CUDA 专属，
  shape 校验应归 route；不能因此给 MetaX 填 CUDA 的 Target 字段。
* `CodeObjectRoute.target_pattern` 目前只有 AMDGCN 和 PTX 两种处理。
  [FlagTree 当前 MetaX compiler 源码](https://github.com/flagos-ai/FlagTree/blob/main/third_party/metax/backend/compiler.py)
  声明 `mcfatbin`，并有 `ttir/ttgir/mlir/llir/mcfatbin` 阶段。
  MetaX 的二进制与 target 检查必须由自己的产物适配承担，不能落到 PTX 分支。

以上上游源码是调查线索，**不是已选镜像的实测行为**。下一步必须检查镜像中
固定版本的实现、实际 `compiled.asm`、metadata、binary symbol 和 launcher ABI。
尤其不能把原生 capability 10.2、PyTorch CUDA 兼容 capability、XCORE1002
设备 ISA 与 xcore1000 编译 family 视为同一个字段。

## 最小合同与验收边界

第一阶段目标是单卡、固定输入形状的 FP32 kernel，沿 canonical Schedule →
Compiler → sealed artifact → Evaluation 路径得到完整输出。操作必须来自
已有 IR；若表达不了，具体指出缺失 primitive，再同时补其 typing / analyses。
失败必须定位到 Target、backend、toolchain、host 或 numerical obligation。
没有 timer 时保持 measurement coverage unavailable，不继承 CUPTI 或 HIP 的测量语义。

这保持 P1/P3 的单一 authoring 形式、P2 的显式硬件和 lowering、P4/P5/P7 的
typing 与分析同行；P6 由完整 Corpus Gate 承担，P8 由精确 Target 的实机观测承担。
初次 device-property 查询不触发 kernel 经验 promotion，也不新增 pass 或校准。

本次验证只覆盖探测工具：在 `c550-1` 成功构建/运行主机查询，所有命令返回 0，
8 张卡 × 18 属性均成功；没有 Compiler 更改，因此没有本次 MetaX Corpus 或
GPU qualification。平台仍处于调查与接入准备阶段。
