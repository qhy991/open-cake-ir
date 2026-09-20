#!/usr/bin/env python3
"""Generate the hardware results reading views from retained, cited observations.

No GPU access, promotion or global-best selection occurs here. Findings and original
Campaign reports remain authoritative. The B300 JSON is an allowlisted, dated report
projection produced by read_task_results.py, not a new acceptance ledger.

  python tools/render_hardware_results.py --write
  python tools/render_hardware_results.py --check
  # Matplotlib 3.10.6 is needed only to regenerate/check the README figure:
  python tools/render_hardware_results.py --write --figures
"""
from __future__ import annotations

import argparse
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GITHUB = "https://github.com/qhy991/open-cake-ir/blob/task/core-hardware-results/"
PLATFORMS = ("NVIDIA", "Apple", "Hygon DCU", "AMD")
F_WIDTH = "findings/2026-09-15-005-explicit-triton-cta-width.json"
F_M4 = "findings/2026-09-12-001-r2-row-reduction-scalarization.json"
F_M1 = "findings/2026-09-11-004-metal-output-column-specialization.json"
DCU = "findings/data/2026-09-18-dcu-confirmatory-medians.json"
SERVICE = "docs/results/b300-service-20260920.json"
TINY = "docs/NVIDIA_CAKE_REPRODUCTION.md"
F_AMD = "findings/2026-09-17-002-two-device-timers-disagree-on-gfx1151.json"
NOTES = {
    "NVIDIA": "B300 的服务实验、CTA 宽度验证与 CAKE 改写各自保留基线和版本，不混成一个榜。B200 单列正确性证据。",
    "Apple": "M1 Pro、M4、M2 分开。Metal 测量整个 command buffer，并按重复 dispatch 均摊；不能与 CUPTI 绝对耗时横比。晋升标签仅表示保留记录记载过晋升，未重新读取远端 registry 的当前代。",
    "Hygon DCU": "BW1101 / gfx938：保留首个合格结果，同时展示未合格任务和后续重复运行。首个合格结果不等于历史最快；使用原记录的分类，不从显示的小数重新判定晋升。",
    "AMD": "gfx1151 的两种设备计时器尚未对齐。保留支持状态与调查入口，不给出合格性能或加速比。",
}


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def number(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Retained prose changed; inspect source before updating selector: {pattern}")
    return float(match.group(1))


def row(platform, hardware, target, task, *, source, workload, date,
        baseline=None, candidate=None, speedup=None, status="已确认", baseline_name="任务固定初始基线",
        version="见来源", locator="", note="", kind="paired", history=None):
    return dict(platform=platform, hardware=hardware, target=target, task=task, source=source,
                workload=workload, date=date, baseline_us=baseline, candidate_us=candidate,
                speedup=speedup, status=status, baseline_name=baseline_name, version=version,
                locator=locator, note=note, ratio_kind=kind, history=history or [])


def collect() -> list[dict]:
    rows = []
    service = read(SERVICE)
    for workspace in service["workspaces"]:
        name = Path(workspace["workspace"]).name
        task, day, _ = name.rsplit("-", 2)
        for run in workspace["runs"]:
            qualified = run["status"] == "reported_qualified"
            speed = run["paired_speedup"]
            rows.append(row("NVIDIA", "B300 · 服务实验", workspace["target"], task,
                source=SERVICE, workload=workspace["workload"], date=f"{day[:4]}-{day[4:6]}-{day[6:]}",
                baseline=run["baseline_ms"] * 1000 if qualified else None,
                candidate=run["candidate_ms"] * 1000 if qualified else None, speedup=speed,
                status="报告已确认" if qualified else "未合格终点",
                version=workspace["compiler_revision"].split("+")[0],
                locator=f"B300-M2:{workspace['workspace']}/report.json; {run['run']}; event={run['confirmation_event']}; candidate={run['candidate_id']}",
                note=f"该 Campaign 记录的最佳合格候选；基线选择={workspace['baseline_selection']}。仅投影原审计结果，不是当前 registry 冠军。" if qualified else f"{run['protocol_adherence']}; {run['endpoint_observation']}; {run['terminal_reason']}",
                history=[dict(turn=p["turn"], event=p["event"], us=p["candidate_ms"] * 1000,
                              speedup=p["paired_speedup"]) for p in run["confirmations"]]))
    if service["unavailable"]:
        raise ValueError("B300 collection contains unavailable reports; account for them explicitly")

    finding = read(F_WIDTH)
    verified = finding["verified_by"]
    for task in ("rmsnorm_input_gradient", "swiglu", "softmax_backward", "cosine_similarity"):
        owner = verified if task in verified else verified["additional_task_validation"]
        value = owner[task]
        rows.append(row("NVIDIA", "B300 · CTA 宽度验证", "sm_103a", task,
            source=F_WIDTH, workload="FP32 · R=128, C=1024", date=finding["date"],
            baseline=value["baseline_ms"] * 1000, candidate=value["candidate_ms"] * 1000,
            speedup=value["paired_speedup"], version=owner["source_commit"][:8],
            locator=owner["evidence_root"] + "/" + value.get("confirmation", value.get("receipt", "")),
            note=f"显式选择 {value['selected_num_warps']} warps；10/10 配对胜出，五种输入与稳定性通过，保留 NCU。无通用最优宽度结论。"))

    tiny = (ROOT / TINY).read_text()
    for line in tiny.splitlines():
        if re.match(r"\| \d+ / \d+ / \d+ \|", line):
            cells = [c.strip() for c in line.strip("|").split("|")]
            for stages, index, ratio_index in ((2, 3, 5), (4, 4, 6)):
                rows.append(row("NVIDIA", "B300 · CAKE 对照", "sm_103a", f"TinyGEMM2 s{stages}",
                    source=TINY, workload=f"BF16+bias · B/N/K={cells[0]}", date="2026-09-20",
                    baseline=float(cells[2]), candidate=float(cells[index]),
                    speedup=1 / float(cells[ratio_index].rstrip("x")), kind="inverse_rounded_slowdown",
                    baseline_name="官方 CAKE stage4 固定导出", status="正确；性能落后", version="539c6c81",
                    locator="B300-M2:cake-tinygemm-compare-b300-539c6c81-a98903082f5c/stages/comparison/checks/report.json",
                    note=f"30/30 Open-Cake 与 15/15 CAKE 检查通过。此加速比由报告中四舍五入的 {cells[ratio_index]} 耗时倍数取倒数，约数；非新配对计算。s2/s4 在短 K 无循环时不是两种流水线。"))
    for task in ("KDA prefill", "KDA decode", "Alpha-MoE"):
        rows.append(row("NVIDIA", "B300 · CAKE 对照", "sm_103a", task, source=TINY,
            workload="完整任务尚未完成", date="2026-09-20", status="未实测",
            baseline_name="官方 CAKE 导出", note="已有源参考与准备规格；尚无完整合格候选。"))
    rows.append(row("NVIDIA", "B200", "sm_100a", "FMA / affine / state-store", source="docs/AKA_FMA_B200_CORRECTNESS_20260904.md",
        workload="固定正确性实例，详见各报告", date="2026-09-06", status="仅正确性",
        note="没有此组实例的合格计时；另见 AFFINE_PARENT_B200_CANARY_20260906 与 STATE_STORE_B200_CORRECTNESS_20260903。"))

    m4 = read(F_M4)
    for task, index, pattern, date in (
        ("channel_absmax_scale", 0, r"baseline ([0-9.]+) ms", "2026-09-12"),
        ("bias_gradient_reduction", 3, r"baseline ([0-9.]+) ms", "2026-09-13"),
    ):
        evidence = m4["evidence"][index]
        note = evidence["note"]
        rows.append(row("Apple", "M4", "apple_gpu_family9", task, source=F_M4,
            workload="FP32 · R=2；完整 C/协议由原 Workload 固定", date=date,
            baseline=number(pattern, note) * 1000, candidate=number(r"candidate ([0-9.]+) ms", note) * 1000,
            speedup=number(r"([0-9.]+)x speedup", note), status="有晋升记录", version="v74/v104" if index == 0 else "v78/v113",
            locator=evidence["workspace"] + f"; {evidence['run']}; event {evidence['event']}",
            note="记录中的 generation 0；并非本轮重新查询的当前冠军。channel_absmax 的晋升另见 F-2026-09-12-003。"))
    m1 = read(F_M1)
    first = m1["evidence"][0]
    rows.append(row("Apple", "M1 Pro", "apple_gpu_family7", "gemm_silu", source=F_M1,
        workload="FP32 · M=128, K=256, N=32", date="2026-09-11",
        baseline=number(r"baseline ([0-9.]+) ms", first["note"]) * 1000,
        candidate=number(r"candidate median ([0-9.]+) ms", first["note"]) * 1000,
        speedup=number(r"speedup ([0-9.]+)", first["note"]), status="历史确认", version="v73/v98",
        locator=first["workspace"] + "; confirmatory event 37; candidate 450d8350",
        note="固定小形状相对朴素基线；后继合格候选为另一种 K-split 机制，未证明跨版本持续加速。"))
    rows.append(row("Apple", "M2", "apple_gpu_family8", "gemm", source="findings/2026-09-10-010-contraction-regime-shows-material-headroom.json",
        workload="原记录的三个 GEMM 候选", date="2026-09-10", status="计时未通过",
        note="搜索曾观察 6–7×，均未通过测量稳定性门；不列入性能图。"))

    dcu = read(DCU)
    for task, value in sorted(dcu["medians_ms"].items()):
        at_floor = value["candidate"] == value["baseline"] == dcu["floor_ms"]
        status = ("测量分辨能力待查" if at_floor else {
            "first_arm_faster": "确认加速", "second_arm_faster": "确认变慢", "close_null": "未检出显著差异",
        }[value["classification"]])
        rows.append(row("Hygon DCU", "BW1101", "gfx938", task, source=DCU,
            workload="原 Campaign 的固定形状（汇总未转录尺寸）", date="2026-09-18",
            baseline=value["baseline"] * 1000, candidate=value["candidate"] * 1000,
            speedup=None if at_floor else value["baseline"] / value["candidate"], kind="ratio_of_medians",
            status=status, version=value["compiler_revision"],
            locator=f"bw1100:/home/testuser01/oci-dcu-runs/{value['workspace']}/campaign-evidence/{value['receipt']}",
            note="每任务首个合格运行，不是 fastest-of-all。" + ("两者均读到 5.439 µs；不写成 1× 性能持平。" if at_floor else "比值来自确认中位数；是否超过提升门槛沿用原 classification。")))
    for task, status in (("gemm_bias", "基线未通过"), ("gemm_silu", "候选未通过")):
        rows.append(row("Hygon DCU", "BW1101", "gfx938", task, source="docs/dcu-gfx938-results.md",
            workload="保留 sweep 的验证形状", date="2026-09-18", status=status,
            note="本集合无合格终点；保留在任务覆盖分母中。"))
    for task, repeats in dcu["excluded_reruns"].items():
        for value in repeats:
            rows.append(row("Hygon DCU", "BW1101 · 后续重复", "gfx938", task, source=DCU,
                workload="原 Campaign 的固定形状（汇总未转录尺寸）", date="2026-09-18",
                baseline=value["baseline"] * 1000, candidate=value["candidate"] * 1000,
                status="保留的后续运行", version=value["compiler_revision"],
                locator=value["workspace"] + "/campaign-evidence/" + value["receipt"],
                note="按原汇总的首个合格选择规则未替换主行；不作为新的最佳实现。gelu_tanh 后继落在待查的 5.439 µs 读数。"))
    rows.append(row("AMD", "Radeon / Strix Halo", "gfx1151", "rmsnorm smoke", source=F_AMD,
        workload="gfx1151-rmsnorm-b8-smoke", date="2026-09-17", status="计时边界未解决",
        locator="infplane；原调查未产生 Campaign", note="31.858 与 24.224 µs 来自不同计时器，不作为合格延迟或加速比。"))
    for index, value in enumerate(rows):
        value["id"] = f"result-{index + 1:03d}"
        if not (ROOT / value["source"]).is_file():
            raise ValueError(f"Missing source {value['source']}")
    return rows


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def counts(rows):
    service = [r for r in rows if r["hardware"] == "B300 · 服务实验"]
    dcu = [r for r in rows if r["hardware"] == "BW1101"]
    return dict(service_runs=len(service), service_tasks=len({r["task"] for r in service}),
                service_qualified=sum(r["candidate_us"] is not None for r in service),
                service_failed=sum(r["candidate_us"] is None for r in service),
                dcu_tasks=len(dcu), dcu_qualified=sum(r["candidate_us"] is not None for r in dcu),
                dcu_failed=sum(r["candidate_us"] is None for r in dcu),
                dcu_repeats=sum(r["hardware"] == "BW1101 · 后续重复" for r in rows),
                dcu_floor=sum(r["status"] == "测量分辨能力待查" for r in rows))


def render_markdown(rows):
    n = counts(rows)
    out = ["# 按硬件查看实验结果", "", "2026-09-20 整理。由 `tools/render_hardware_results.py` 生成；原报告与 Finding 为依据。",
           "", "[交互目录](results/index.html) · [返回首页](../README.md) · [CAKE 复现](NVIDIA_CAKE_REPRODUCTION.md) · [最佳实现维护](zh-CN/TASK_INCUMBENTS.md)", "",
           "**加速比 = 基线耗时 ÷ 候选耗时，超过 1× 表示更快。** 各行仅在自己的硬件、Workload、版本与计时协议内比较，不求跨平台平均值。",
           "", "这是有明确来源的结果目录，不是全项目实时冠军榜。保留了失败、未实测与无显著差异的条目。服务实验展示每个 Campaign 的最佳合格候选；晋升记录和历史确认另作标记。", "",
           "## 数据范围与更新", "", f"- B300 服务实验：本轮只读收集 {n['service_runs']} 份 report，涉及 {n['service_tasks']} 个不同任务；{n['service_qualified']} 个合格终点及 {n['service_failed']} 次未合格尝试。报告中的审计结论照录，没有重新审计或晋升。",
           f"- DCU：保留 {n['dcu_tasks']} 个任务中的 {n['dcu_qualified']} 个首个合格结果、{n['dcu_failed']} 个未合格任务，以及 {n['dcu_repeats']} 次后续重复。{n['dcu_floor']} 个双方均为 5.439 µs 的结果不写成性能持平。原主机本轮 SSH 超时，尺寸与协议未从远端重新提取；表内明确保留缺项。",
           "- Apple 晋升与跨版本结果来自注明日期的 Finding；尚未刷新 M4 registry 当前代，不能称作今天的全量最佳实现。",
           "- 每项来源下列出原 Workload、版本、事件与产物定位。远端产物需要相应主机权限；文档与图不授予新验收资格。", "",
           "更新流程：用 `tools/read_task_results.py --runs /原实验目录` 只读导出到仓库外的新文件，核对范围后保留带日期的公开字段投影；更新来源选择，再运行 `python tools/render_hardware_results.py --write`。",
           "图使用 Matplotlib 3.10.6，运行 `--write --figures` 更新，`--check --figures` 检查图与目录是否过期。不要手改生成表格；原始实验报告继续保留在原证据根。", ""]
    for platform in PLATFORMS:
        out += [f"## {platform}", "", NOTES[platform], "",
                "| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |",
                "|---|---|---|---:|---:|---:|---|---|"]
        for r in rows:
            if r["platform"] == platform:
                out.append(f"| {r['hardware']} | `{r['task']}` | {r['workload']} | {fmt(r['baseline_us'])} | {fmt(r['candidate_us'])} | {fmt(r['speedup'])}{'×' if r['speedup'] is not None else ''} | {r['status']} | [{r['id']}](#{r['id']}) |")
        out += [""]
    out += ["## 演进记录怎么读", "",
            "交互页展开 B300 服务实验时，可以看到同一 Campaign 内已通过确认的候选时间序列；它保留变慢点，不跨 Campaign 拼接成长期晋升曲线。",
            "M1 Pro 的 GEMM-SiLU 曾在旧版本确认 23.60×，后继版本保留了计时未通过的 33.9× 搜索值以及另一个机制的 14.19× 合格终点；这些不是单调提升曲线。",
            "M4 两项记录的 generation 0 证明晋升已发生，不足以画多代曲线。TinyGEMM2 的 22/30 到 30/30 是数值与资源处理进展，早期无有效计时。",
            "跨编译器版本时分别看固定基线的变化与候选相对基线的改善，完整失败过程见各条来源。", "", "## 逐项来源", ""]
    for r in rows:
        out += [f"### {r['id']}", "", f"**{r['hardware']} · {r['task']}** — {r['date']} / {r['status']}", "",
                f"- Workload：`{r['workload']}`；目标：`{r['target']}`；版本：`{r['version']}`。",
                f"- 基线：{r['baseline_name']}；比值口径：`{r['ratio_kind']}`。",
                f"- 来源：[{r['source']}](../{r['source']})。",
                f"- 原记录 / 实现定位：`{r['locator'] or '见来源所引用的原始实验'}`。", f"- {r['note']}", ""]
        if r["history"]:
            out += ["| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |", "|---|---|---:|---:|"]
            out += [f"| {p['turn']} | {p['event']} | {p['us']:.3f} | {p['speedup']:.3f}× |" for p in r["history"]]
            out += [""]
    return "\n".join(out)


def figure_rows(rows):
    def select(hardware, tasks):
        return [next(r for r in rows if r["hardware"] == hardware and r["task"] == task and r["speedup"] is not None) for task in tasks]
    return [
        ("NVIDIA B300", "Retained campaign endpoints / starter baseline", select("B300 · 服务实验", ["pairwise_sqdist", "gelu_tanh_backward", "adamw", "layernorm"])),
        ("Apple M4", "Recorded promotions / starter baseline", select("M4", ["bias_gradient_reduction", "channel_absmax_scale"])),
        ("Hygon BW1101", "First qualified run / starter baseline", select("BW1101", ["per_channel_moments", "momentum_sgd", "pairwise_sqdist"])),
        ("Apple M1 Pro", "Historical fixed shape / naive baseline", select("M1 Pro", ["gemm_silu"])),
    ]


def render_figure(rows):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    matplotlib.rcParams.update({"svg.hashsalt": "open-cake-results", "font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 6.7))
    fig.set_facecolor("#f8fafc")
    for ax, (title, subtitle, selected), color in zip(axes.flat, figure_rows(rows), ("#0e7490", "#6d5cc7", "#b45309", "#6d5cc7")):
        ax.set_facecolor("#f8fafc")
        values = [r["speedup"] for r in selected]
        labels = [r["task"] for r in selected]
        ax.barh(labels, values, height=.46, color=color)
        ax.set_ylim(max(3, len(selected)) - .5, -.5)
        ax.axvline(1, color="#334155", ls="--", lw=1)
        for i, value in enumerate(values):
            ax.text(value + max(values) * .025, i, f"{value:.3f}x", va="center", weight="bold", color="#0f172a")
        ax.set_xlim(0, max(1.6, max(values) * 1.28))
        ax.set_title(title + "\n" + subtitle, loc="left", fontsize=10, pad=13)
        ax.set_xlabel("Baseline / candidate time (higher is faster)", fontsize=9)
        ax.tick_params(axis="y", labelsize=9, length=0)
        ax.tick_params(axis="x", labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.grid(axis="x", color="#e2e8f0", lw=.7)
        ax.set_axisbelow(True)
    fig.suptitle("Open-Cake  |  Retained results by hardware", x=.03, ha="left", fontsize=18, weight="bold", color="#0f172a")
    fig.text(.03, .02, "2026-09-20 snapshot  •  Different panel scales and timing protocols  •  No cross-hardware ranking or external-library claim", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.21, right=.97, top=.83, bottom=.12, wspace=.95, hspace=.7)
    out = io.StringIO()
    fig.savefig(out, format="svg", metadata={"Date": None})
    plt.close(fig)
    return "\n".join(line.rstrip() for line in out.getvalue().splitlines()) + "\n"


def readme_block(rows):
    n = counts(rows)
    service = [r for r in rows if r["hardware"] == "B300 · 服务实验" and r["task"] == "pairwise_sqdist"][0]
    return f"""<!-- hardware-results:start -->
## 按硬件查看成果

2026-09-20 整理：展示已有确认结果与最佳实现维护记录。**加速比 = 各自固定基线耗时 ÷ 候选耗时**，超过 1× 表示更快。
各平台的输入与计时协议不同，图中各面板使用独立刻度；这些是选定任务的结果，不是跨硬件排名，也不是相对厂商最优库的成绩。

![按硬件分组的代表性确认结果](docs/results/overview.svg)

| 硬件 | 收录范围 | 阅读入口 |
|---|---|---|
| NVIDIA B300 / B200 | B300 服务实验 {n['service_tasks']} 个任务、{n['service_runs']} 次尝试，含 `pairwise_sqdist` {service['speedup']:.3f}×；另含 CTA 宽度验证与 CAKE 对照。B200 保留正确性边界 | [NVIDIA](docs/RESULTS.md#nvidia) |
| Apple M1 Pro / M4 / M2 | M1 Pro 固定形状的历史确认、M4 两项晋升记录；M2 未通过稳定性的结果单列 | [Apple](docs/RESULTS.md#apple) |
| Hygon DCU BW1101 | {n['dcu_tasks']} 个任务中 {n['dcu_qualified']} 个首个合格结果，含加速、变慢、未检出差异及计时分辨能力待查的条目 | [DCU](docs/RESULTS.md#hygon-dcu) |
| AMD gfx1151 | 两种设备计时器尚未对齐，暂不列合格性能 | [AMD](docs/RESULTS.md#amd) |

[完整任务与证据目录](docs/RESULTS.md) · [可筛选页面源码 / 下载后打开](docs/results/index.html) · [最佳实现如何维护](docs/zh-CN/TASK_INCUMBENTS.md)

目录区分“Campaign 内最佳合格候选”“有晋升记录”“历史确认”和“尚未合格”；每项可追溯到输入、版本、基线与原记录。
同一 Campaign 的确认历史保留在详情中；当前没有足够的跨代数据为每个任务绘制长期晋升曲线。

### 与原始 CAKE 的距离

TinyGEMM2 在同一 B300 上与 CAKE 的固定 FlashInfer 导出比较，Open-Cake 通过 30/30 严格数值检查，已测配置耗时仍约为 CAKE 的 11–37 倍。
KDA prefill、KDA decode、Alpha-MoE 尚无完整改写实测。[查看形状、计时协议和演进过程](docs/NVIDIA_CAKE_REPRODUCTION.md)。
<!-- hardware-results:end -->"""


def render_html(rows):
    template = (ROOT / "tools/hardware_results.html").read_text()
    payload = json.dumps({"rows": rows, "notes": NOTES, "github": GITHUB, "counts": counts(rows)}, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("__RESULTS_DATA__", payload)


def outputs(figures=False):
    rows = collect()
    readme = (ROOT / "README.md").read_text()
    block = readme_block(rows)
    if "<!-- hardware-results:start -->" in readme:
        readme = re.sub(r"<!-- hardware-results:start -->.*?<!-- hardware-results:end -->", lambda _: block, readme, flags=re.S)
    else:
        readme = readme.replace("## 快速开始", block + "\n\n## 快速开始", 1)
    rendered = {"README.md": readme, "docs/RESULTS.md": render_markdown(rows), "docs/results/index.html": render_html(rows)}
    if figures:
        rendered["docs/results/overview.svg"] = render_figure(rows)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    stale = []
    for relative, content in outputs(args.figures).items():
        path = ROOT / relative
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        elif not path.is_file() or path.read_text() != content:
            stale.append(relative)
    if stale:
        raise SystemExit("Outdated results views: " + ", ".join(stale))
    print("Hardware results views " + ("written" if args.write else "match retained inputs"))


if __name__ == "__main__":
    main()
