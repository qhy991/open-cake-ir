"""Read-only catalog views and exact-cell source checks; never execute a Run."""
from __future__ import annotations

from collections import Counter
import json

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.source_identity import checkout_commit
from . import catalog
from .devices import BACKENDS
from .workloads import create_task, validate_workload_document


RUNTIME_CHECKS = ("toolchain_build", "provider_qualification", "gpu_allocation",
                  "device_correctness", "measurement_quality", "independent_confirmation")


def check_cell(compiler: Compiler, row: dict, backend: str, *,
               rows: int | None = None, columns: int | None = None,
               depth: int | None = None) -> dict:
    """Check the actual factory, frontend, verifier and emission for one shape."""
    if backend not in BACKENDS:
        raise ValueError(f"unknown task backend {backend!r}")
    result = {**row, "backend": backend, "target": BACKENDS[backend]["target"],
              "source_status": "not_examined", "findings": [],
              "runtime_status": "not_examined"}
    if row["authoring"] != "registered" or row.get("collection_status", "ready") != "ready":
        result.update(source_status="not_integrated", refused_at="authoring")
        return result
    shape = row["shape"]
    resolved_rows = shape["rows"] if rows is None else rows
    resolved_columns = shape["columns"] if columns is None else columns
    resolved_depth = shape.get("depth") if depth is None else depth
    result["shape"] = {"rows": resolved_rows, "columns": resolved_columns, "depth": resolved_depth}
    stage = "workload"
    try:
        document, source = create_task(row["task"], backend=backend, rows=resolved_rows,
                                       columns=resolved_columns, depth=resolved_depth)
        validate_workload_document(document)
        result["workload_id"] = document["workload_id"]
        result["validation_cases"] = len(document["cases"])
        stage = "frontend"
        parsed = frontend.parse(source, filename=f"{row['task']}/starter.py")
        stage = "assessment"
        assessment = compiler.assess(parsed.document)
        result["findings"] = [finding.to_dict() for finding in assessment.findings]
        if not assessment.lowering_eligible:
            result.update(source_status="refused", refused_at=stage)
            return result
        stage = "lowering"
        lowered = compiler.lower(assessment)
        result["source_status"] = "source_generated" if lowered.generated else "refused"
        if not lowered.generated:
            result["refused_at"] = stage
    except ValueError as error:
        result.update(source_status="refused", refused_at=stage, reason=str(error))
    return result


def report(args) -> dict:
    root = args.project_root
    rows = catalog.entries(root, suite=args.suite, family=args.family)
    if args.task_command == "show":
        # Show accepts either the launcher name or the reference collection id.
        rows = [row for row in rows if args.task in (row["id"], row["task"])]
        if not rows and args.suite == "all":
            rows = [row for row in catalog.entries(root, suite="flashinfer-rewrites")
                    if args.task in (row["id"], row["task"])]
        if not rows:
            raise ValueError(f"unknown task {args.task!r} in suite {args.suite!r}")
    elif args.task_command == "check" and args.task:
        requested = set(args.task)
        available = {key for row in rows for key in (row["id"], row["task"])}
        if requested - available:
            raise ValueError(f"unknown tasks in selected suite: {sorted(requested - available)}")
        rows = [row for row in rows if requested & {row["id"], row["task"]}]
    if args.task_command == "check":
        if not rows:
            raise ValueError("no tasks match the selected suite and family")
        if len(rows) != 1 and any(getattr(args, name) is not None for name in ("rows", "columns", "depth")):
            raise ValueError("shape overrides require exactly one selected task")
        # Identity belongs to the existing source boundary, not to a catalog snapshot.
        commit = checkout_commit(root)
        compiler = Compiler.load(root, root / "compiler/revision.json")
        rows = [check_cell(compiler, row, args.backend, rows=args.rows,
                           columns=args.columns, depth=args.depth) for row in rows]
        scope = "fixed_shape_source_checks_only"
    else:
        commit = None
        scope = "task_registration_only"
    backends = {name: {"target": device["target"], "lowering": device["route"],
                       "host_capture": "present" if (root / "runtime/hosts" / (device["target"] + ".json")).is_file() else "missing"}
                for name, device in BACKENDS.items()}
    if args.task_command == "check":
        backends = {args.backend: backends[args.backend]}
    return {"scope": scope, "source_commit": commit, "suite": args.suite,
            "task_count": len(rows), "families": dict(Counter(row["family"] for row in rows)),
            "source_status_counts": dict(Counter(row["source_status"] for row in rows))
            if args.task_command == "check" else None,
            "backends": backends, "runtime_checks": {name: "not_examined" for name in RUNTIME_CHECKS},
            "corpus_gate": "not_examined", "tasks": rows}


def _shape(row: dict) -> str:
    shape = row["shape"]
    if shape["rows"] is None or shape["columns"] is None:
        return "待绑定完整任务"
    text = f"{shape['rows']} × {shape['columns']}"
    if shape.get("depth") is not None:
        text += f"; K={shape['depth']}"
    elif shape.get("depth_required"):
        text += "; K 必须指定"
    return text


def render(document: dict, output_format: str) -> str:
    if output_format == "json":
        return json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    checked = document["scope"] == "fixed_shape_source_checks_only"
    lines = [f"任务目录：{document['suite']}，{document['task_count']} 项。",
             "检查范围：固定形状的构造、验证与源码生成。" if checked else
             "检查范围：任务注册与参考包接线；未检查目标支持。",
             "GPU、工具链、作者资格、计时及最终确认均未检查。"]
    lines.append("分类：" + "，".join(f"{name}={count}" for name, count in document["families"].items()))
    if document["source_commit"]:
        lines.append(f"源码提交：{document['source_commit']}")
    labels = {"registered": "已注册", "not_integrated": "未接通",
              "source_generated": "源码已生成", "refused": "拒绝",
              "ready": "作者入口已接通", "skipped": "作者入口未接通", "blocked": "接入受阻"}
    if output_format == "markdown":
        lines += ["", "| Task | 分类 | 形状（rows × columns） | 状态 | 原因 |",
                  "| --- | --- | --- | --- | --- |"]
    else:
        lines.append("")
    for row in document["tasks"]:
        status = row.get("source_status", row.get("collection_status", row["authoring"]))
        reason = row.get("reason") or "; ".join(
            f"{f['code']}@{f['path']}" for f in row.get("findings", [])
            if f.get("blocks_acceptance") or f.get("blocks_lowering"))
        ident = row["id"]
        if ident != row["task"]:
            ident += " → " + row["task"]
        cells = (ident, row["family"], _shape(row), labels.get(status, status), reason or "—")
        if output_format == "markdown":
            lines.append("| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in cells) + " |")
        else:
            lines.append(" · ".join(cells))
    lines += ["", "目标入口与 host capture（文件存在不代表环境已验收）："]
    for name, backend in document["backends"].items():
        lines.append(f"- {name}: {backend['target']} / {backend['lowering']} / host {backend['host_capture']}")
    return "\n".join(lines)


def run(args) -> int:
    document = report(args)
    print(render(document, args.output_format))
    if args.task_command == "check":
        return 0 if all(row["source_status"] == "source_generated" for row in document["tasks"]) else 1
    return 0
