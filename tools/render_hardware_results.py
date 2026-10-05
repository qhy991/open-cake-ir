#!/usr/bin/env python3
"""Render platform-owned publication records; no GPU access, auditing or promotion.

Each maintained branch owns docs/results/<branch>/records.json. Platform updates
render only that directory. Main aggregates the records present in its checkout;
it never fetches moving branch heads or modifies the input records.

  python tools/render_hardware_results.py --platform nvidia --write --figures
  python tools/render_hardware_results.py --write --figures  # main integration
  python tools/render_hardware_results.py --check

Matplotlib 3.10.6 is needed only with --figures. See docs/RESULTS_MAINTENANCE.md.
"""
from __future__ import annotations

import argparse
import html
import io
import json
import math
import re
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/qhy991/open-cake-ir"
BRANCHES = ("nvidia", "metal", "dcu", "amd")


def load_datasets(platform=None):
    datasets = []
    for branch in (BRANCHES if platform is None else (platform,)):
        path = ROOT / "docs/results" / branch / "records.json"
        data = json.loads(path.read_text())
        if data.get("schema_version") != 1 or data.get("branch") != branch:
            raise ValueError(f"{path}: schema or owning branch differs")
        if not data.get("platform") or not re.fullmatch(r"[0-9a-f]{40}", data.get("source_commit", "")):
            raise ValueError(f"{path}: platform and immutable source commit are required")
        ids = set()
        for record in data["records"]:
            identifier = record.get("id", "")
            if not identifier.startswith(branch + "-") or identifier in ids:
                raise ValueError(f"{path}: duplicate or foreign record id {identifier}")
            ids.add(identifier)
            if "platform" in record:
                raise ValueError(f"{path}: platform is owned by the document, not each row")
            for key in ("baseline_us", "candidate_us", "speedup"):
                value = record[key]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int))
                                          or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"{identifier}: {key} must be a positive finite value or null")
            source = Path(record["source"])
            commit = record.get("source_commit", data["source_commit"])
            if source.is_absolute() or ".." in source.parts or not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError(f"{identifier}: source must be a repository path at a fixed commit")
        for panel in data["figures"]:
            for identifier in panel["records"]:
                if identifier not in ids:
                    raise ValueError(f"{path}: figure refers to missing record {identifier}")
        datasets.append(data)
    return datasets


def collect(platform=None):
    rows = []
    for data in load_datasets(platform):
        for record in data["records"]:
            commit = record.get("source_commit", data["source_commit"])
            rows.append({**record, "platform": data["platform"], "branch": data["branch"],
                         "source_url": f"{REPOSITORY}/blob/{commit}/{quote(record['source'])}"})
    return rows


def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def render_markdown(datasets, rows, platform=None):
    heading = "按硬件查看实验结果" if platform is None else datasets[0]["platform"] + " 实验结果"
    if platform is None:
        navigation = "[交互目录](results/index.html) · [返回首页](../README.md) · [分支维护流程](RESULTS_MAINTENANCE.md)"
    else:
        navigation = f"[交互目录](index.html) · [发布数据](records.json) · [main 汇总]({REPOSITORY}/blob/main/docs/RESULTS.md) · [维护流程](../../RESULTS_MAINTENANCE.md)"
    out = [f"# {heading}", "", "由 `tools/render_hardware_results.py` 从各平台发布数据生成。原始报告与 Finding 保留权威。", "", navigation, "",
           "**加速比 = 基线耗时 ÷ 候选耗时，超过 1× 表示更快。** 各行仅在自己的硬件、Workload、版本与计时协议内比较，不求跨平台平均值。", "",
           "平台分支分别更新自己的发布数据；main 展示已经合入当前检出的版本。这里不会在线抓取平台分支最新值，也不是实时冠军榜。失败、无显著差异与未实测记录均保留。", "",
           "## 数据归属", "", "| 平台 | 维护分支 | 数据日期 | 观察条目 | 发布数据 |", "|---|---|---|---:|---|"]
    if platform is not None and datasets[0]["figures"]:
        out[6:6] = [f"![{heading}代表结果](overview.svg)", ""]
    for data in datasets:
        branch = data["branch"]
        local = f"results/{branch}/records.json" if platform is None else "records.json"
        out.append(f"| {data['platform']} | [{branch}]({REPOSITORY}/tree/{branch}/docs/results/{branch}) | {data['as_of']} | {len(data['records'])} | [{branch}/records.json]({local}) |")
    out += ["", "观察条目数不等于任务数：同一任务可以有不同形状、实验集合和历史尝试。", ""]
    for data in datasets:
        out += [f"## {data['platform']}", "", data["notes"], ""]
        out += [f"- {detail}" for detail in data["details"]]
        out += ["", "| 设备 / 集合 | Task | 输入 / Workload | 基线 µs | 候选 µs | 加速比 | 状态 | 详情 |", "|---|---|---|---:|---:|---:|---|---|"]
        for r in rows:
            if r["branch"] == data["branch"]:
                out.append(f"| {r['hardware']} | `{r['task']}` | {r['workload']} | {fmt(r['baseline_us'])} | {fmt(r['candidate_us'])} | {fmt(r['speedup'])}{'×' if r['speedup'] is not None else ''} | {r['status']} | [{r['id']}](#{r['id']}) |")
        out += [""]
    out += ["## 演进与更新", "",
            "交互页保留同一 Campaign 内已通过确认的候选时间序列，包括变慢点。它不把不同形状、版本和计时协议拼成长期晋升曲线。",
            "历史晋升记录不等于当前 registry 冠军。跨编译器版本时分别看固定基线的变化与候选相对基线的改善；完整失败过程见来源。",
            "新增实验先在对应平台的 records.json 追加有稳定 id 和固定来源提交的观察，再生成该平台页面。合入 main 的集成分支重建总览；不要手工改生成表格。具体命令见维护流程。", "", "## 逐项来源", ""]
    for r in rows:
        out += [f"### {r['id']}", "", f"**{r['hardware']} · {r['task']}** — {r['date']} / {r['status']}", "",
                f"- Workload：`{r['workload']}`；目标：`{r['target']}`；版本：`{r['version']}`。",
                f"- 基线：{r['baseline_name']}；比值口径：`{r['ratio_kind']}`。",
                f"- 来源：[{r['source']}]({r['source_url']})。",
                f"- 原记录 / 实现定位：`{r['locator'] or '见来源所引用的原始实验'}`。", f"- {r['note']}", ""]
        if r["history"]:
            out += ["| 确认轮次 | 事件 | 候选 µs | 该轮配对加速比 |", "|---|---|---:|---:|"]
            out += [f"| {p['turn']} | {p['event']} | {p['us']:.3f} | {p['speedup']:.3f}× |" for p in r["history"]]
            out += [""]
    return "\n".join(out)


def figure_rows(rows, datasets=None):
    result = []
    for data in datasets if datasets is not None else load_datasets():
        for panel in data["figures"]:
            selected = [next(r for r in rows if r["id"] == identifier) for identifier in panel["records"]]
            if any(r["speedup"] is None for r in selected):
                raise ValueError("Figure selection includes an observation without a reportable ratio")
            result.append((panel["title"], panel["subtitle"], selected))
    return result


def render_figure(rows, datasets=None):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    panels = figure_rows(rows, datasets)
    if not panels:
        return None
    matplotlib.rcParams.update({"svg.hashsalt": "open-cake-results", "font.family": "DejaVu Sans", "font.size": 10})
    columns = min(2, len(panels))
    lines = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(lines, columns, figsize=(12.4 if columns == 2 else 8, 3.4 * lines + .5), squeeze=False)
    fig.set_facecolor("#f8fafc")
    for ax in axes.flat:
        ax.set_visible(False)
    for ax, (title, subtitle, selected) in zip(axes.flat, panels):
        ax.set_visible(True)
        ax.set_facecolor("#f8fafc")
        values = [r["speedup"] for r in selected]
        ax.barh([r["task"] for r in selected], values, height=.46, color="#0e7490")
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
    fig.text(.03, .02, "Platform-owned publication records  •  Independent scales / protocols  •  No cross-hardware ranking", fontsize=9, color="#475569")
    fig.subplots_adjust(left=.21 if columns == 2 else .3, right=.96, top=.83, bottom=.14, wspace=.95, hspace=.7)
    out = io.StringIO()
    fig.savefig(out, format="svg", metadata={"Date": None})
    plt.close(fig)
    return "\n".join(line.rstrip() for line in out.getvalue().splitlines()) + "\n"


def readme_block(datasets):
    out = ["<!-- hardware-results:start -->", "## 按硬件查看成果", "",
           "各平台维护自己的发布数据，main 汇总已合入的版本。观察条目包括形状、实验集合和历史尝试，**不是任务总数**。", "",
           "| 硬件 | 维护分支 | 已收录观察 | 结果页面 |", "|---|---|---:|---|"]
    for data in datasets:
        branch = data["branch"]
        out.append(f"| {data['platform']} | [{branch}]({REPOSITORY}/tree/{branch}/docs/results/{branch}) | {len(data['records'])} | [{data['card']}](docs/results/{branch}/README.md) |")
    out += ["", "[完整结果与原始证据](docs/RESULTS.md) · [FlashInfer 逐任务对比](docs/results/nvidia/FLASHINFER_STATUS.md) · [CAKE 对照与改写](docs/NVIDIA_CAKE_REPRODUCTION.md)", "",
            "[交互目录（下载后打开）](docs/results/index.html) · [发布数据维护流程](docs/RESULTS_MAINTENANCE.md)", "",
            "**加速比 = 各自固定基线耗时 ÷ 候选耗时**，超过 1× 表示更快。正确性、计时质量与性能收益分别记录；各平台协议不同，不作跨硬件排名。", "",
            "<details>", "<summary>展开代表结果图（各平台独立刻度）</summary>", "",
            "![按硬件分组的代表性确认结果](docs/results/overview.svg)", "",
            "图表展示选定的历史观察；完整表格保留失败、无显著差异和未实测记录。各自任务基线不等于厂商最优库。", "",
            "</details>", "<!-- hardware-results:end -->"]
    return "\n".join(out)


def render_html(datasets, rows, platform=None):
    template = (ROOT / "tools/hardware_results.html").read_text()
    title = "按硬件查看实验结果" if platform is None else datasets[0]["platform"] + " 实验结果"
    scope = "main 汇总 · 已合入发布数据" if platform is None else platform + " 分支 · 独立维护"
    cards = "\n".join(f'<div class="card"><b>{html.escape(d["platform"])}</b><span>{html.escape(d["card"])}<br>{len(d["records"])} 条观察 · {html.escape(d["as_of"])}</span></div>' for d in datasets)
    links = [{"label": "main 汇总", "url": f"{REPOSITORY}/blob/main/docs/RESULTS.md"},
             {"label": "分支维护流程", "url": f"{REPOSITORY}/blob/main/docs/RESULTS_MAINTENANCE.md"}]
    links += [{"label": d["branch"] + " 维护入口", "url": f"{REPOSITORY}/tree/{d['branch']}/docs/results/{d['branch']}"} for d in datasets]
    payload = json.dumps({"rows": rows, "notes": {d["platform"]: d["notes"] for d in datasets}, "links": links}, ensure_ascii=False).replace("<", "\\u003c")
    return (template.replace("__RESULTS_DATA__", payload).replace("__PAGE_TITLE__", html.escape(title))
            .replace("__PAGE_SCOPE__", html.escape(scope)).replace("__PLATFORM_CARDS__", cards))


def outputs(figures=False, platform=None):
    datasets, rows = load_datasets(platform), collect(platform)
    rendered = {}
    for data in datasets:
        branch = data["branch"]
        selected = [r for r in rows if r["branch"] == branch]
        prefix = f"docs/results/{branch}"
        rendered[prefix + "/README.md"] = render_markdown([data], selected, branch)
        rendered[prefix + "/index.html"] = render_html([data], selected, branch)
        if figures:
            picture = render_figure(selected, [data])
            if picture is not None:
                rendered[prefix + "/overview.svg"] = picture
    if platform is None:
        readme = (ROOT / "README.md").read_text()
        block = readme_block(datasets)
        readme = re.sub(r"<!-- hardware-results:start -->.*?<!-- hardware-results:end -->", lambda _: block, readme, flags=re.S)
        rendered.update({"README.md": readme, "docs/RESULTS.md": render_markdown(datasets, rows),
                         "docs/results/index.html": render_html(datasets, rows)})
        if figures:
            rendered["docs/results/overview.svg"] = render_figure(rows, datasets)
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--platform", choices=BRANCHES)
    parser.add_argument("--figures", action="store_true")
    args = parser.parse_args()
    stale = []
    for relative, content in outputs(args.figures, args.platform).items():
        path = ROOT / relative
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        elif not path.is_file() or path.read_text() != content:
            stale.append(relative)
    if stale:
        raise SystemExit("Outdated results views: " + ", ".join(stale))
    print((args.platform or "main aggregate") + ": results views " + ("written" if args.write else "match publication records"))


if __name__ == "__main__":
    main()
