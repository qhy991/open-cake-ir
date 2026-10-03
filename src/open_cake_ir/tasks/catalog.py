"""Authoring task selection and shape defaults, derived from task-owned registries.

This catalog owns launcher selection, not operator semantics or device qualification.
The reference collection retains ownership of its materials and integration status.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from . import add_rmsnorm, metax_fp8_gemm
from .activation import workload as activation
from .aka_v3 import workload as aka
from .contraction import workload as contraction
from .gemm import workload as gemm
from .normalization import workload as normalization
from .optimizers import workload as optimizers
from .reductions import workload as reductions
from .rowwise import workload as rowwise
from .solx_fib import gemm as fib_gemm, workload as fib_normalization
from .tinygemm import reproduction as tinygemm


@dataclass(frozen=True)
class Task:
    name: str
    family: str
    owner: str


# Families own their member names. Only the launcher grouping and order live here.
_FAMILIES = (
    ("normalization", normalization, tuple(normalization.TASKS)),
    ("activation", activation, tuple(activation.TASKS)),
    ("rowwise", rowwise, tuple(rowwise.TASKS)),
    ("reductions", reductions, tuple(reductions.TASKS)),
    ("optimizers", optimizers, tuple(optimizers.TASKS)),
    ("contraction", contraction, tuple(contraction.TASKS)),
    ("solx_fib_normalization", fib_normalization, fib_normalization.launchable_tasks()),
    ("solx_fib_gemm", fib_gemm, tuple(fib_gemm.TASKS)),
    ("gemm", gemm, tuple(gemm.TASKS)),
    ("add_rmsnorm", add_rmsnorm, (add_rmsnorm.TASK,)),
    ("aka_v3", aka, tuple(aka.LAUNCHABLE_TASKS)),
    ("tinygemm", tinygemm, (tinygemm.TASK,)),
    ("metax_fp8_gemm", metax_fp8_gemm, (metax_fp8_gemm.TASK,)),
)
TASKS = tuple(Task(name, family, owner.__name__)
              for family, owner, names in _FAMILIES for name in names)
if len({task.name for task in TASKS}) != len(TASKS):
    raise ValueError("authoring task names must be unique across families")

PORTABLE_FAMILIES = frozenset({"normalization", "activation", "rowwise", "reductions",
                               "optimizers", "contraction", "gemm"})
SUITES = ("all", "portable", "flashinfer-rewrites")
COLLECTION_PATH = Path("experiments/flashinfer_rewrites/catalog.json")


def task_names(*, family: str | None = None, suite: str = "all") -> tuple[str, ...]:
    if suite not in {"all", "portable"}:
        raise ValueError("task name selection requires all or portable")
    if family is not None and family not in {task.family for task in TASKS}:
        raise ValueError(f"unknown task family {family!r}")
    return tuple(task.name for task in TASKS
                 if (family is None or task.family == family)
                 and (suite == "all" or task.family in PORTABLE_FAMILIES))


def task_entry(name: str) -> Task:
    for task in TASKS:
        if task.name == name:
            return task
    raise ValueError(f"unknown authoring task {name!r}; use open-cake-ir tasks list")


def default_shape(name: str, rows: int | None = None,
                  columns: int | None = None) -> tuple[int, int]:
    """The single-task launcher's defaults; explicit extents always win."""
    family = task_entry(name).family
    if family == "metax_fp8_gemm":
        default_rows, default_columns = metax_fp8_gemm.SIZE, metax_fp8_gemm.SIZE
    elif family == "aka_v3":
        default_rows, default_columns = (
            (1024, 16) if name == "aka_histogram" else
            (2, 8) if name == "aka_max_pool1d" else (8, 256)
        )
    elif family == "add_rmsnorm":
        default_rows, default_columns = 128, 2560
    elif family == "tinygemm":
        default_rows, default_columns = 1, 128
    elif family == "contraction":
        # Keep the bounded contraction seed, rather than the elementwise tile.
        default_rows, default_columns = 1024, 64
    elif family == "solx_fib_gemm":
        spec = fib_gemm.SPECS[name]
        default_rows, default_columns = min(spec["batches"]), spec["N"]
    elif family == "solx_fib_normalization":
        default_rows, default_columns = fib_normalization.default_rows(name), fib_normalization.SPECS[name]["hidden"]
    else:
        default_rows, default_columns = 128, 1024
    return (default_rows if rows is None else rows,
            default_columns if columns is None else columns)


def shape_description(name: str) -> dict:
    task = task_entry(name)
    rows, columns = default_shape(name)
    depth = None
    depth_required = task.family in {"contraction", "gemm"}
    if task.family == "solx_fib_gemm":
        depth = fib_gemm.SPECS[name]["K"]
    elif task.family == "tinygemm":
        depth = 720
    elif task.family == "metax_fp8_gemm":
        depth = metax_fp8_gemm.SIZE
    elif name == "aka_gemm_nt_bias":
        depth = 32
    return {"rows": rows, "columns": columns, "depth": depth,
            "depth_required": depth_required}


def matrix_depth(name: str, requested: int | None) -> int | None:
    """Preserve the existing batch driver's K forwarding and bounded default."""
    family = task_entry(name).family
    if family == "tinygemm":
        return requested  # Its Workload factory owns the absent K.
    if family in {"contraction", "gemm"} or name == "aka_gemm_nt_bias":
        return 256 if requested is None else requested
    return None


def collection_rows(project_root: Path) -> list[dict]:
    """Read the existing reference inventory and refuse stale ready declarations."""
    rows = json.loads((project_root / COLLECTION_PATH).read_text(encoding="utf-8"))["tasks"]
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("reference collection ids must be unique")
    names = set(task_names())
    for row in rows:
        if row["status"] == "ready" and row["task"] not in names:
            raise ValueError(f"{row['id']}: ready collection task is not registered for authoring")
    return rows


def select_collection_tasks(project_root: Path, ids: list[str] | None) -> list[dict]:
    rows = collection_rows(project_root)
    selected = [row["id"] for row in rows if row["status"] == "ready"] if ids is None else ids
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("select a nonempty set of unique tasks")
    by_id = {row["id"]: row for row in rows}
    result = []
    for ident in selected:
        row = by_id.get(ident)
        if row is None or row["status"] != "ready":
            raise ValueError(f"{ident}: {row['reason'] if row else 'unknown task'}")
        result.append(row)
    return result


def entries(project_root: Path, *, suite: str = "all", family: str | None = None) -> list[dict]:
    """Project registration and collection facts without inventing runtime readiness."""
    collection = collection_rows(project_root)
    references = {}
    for row in collection:
        references.setdefault(row["task"], []).append(row["id"])
    if suite == "flashinfer-rewrites":
        result = []
        for row in collection:
            registered = row["task"] in task_names()
            task = task_entry(row["task"]) if registered else None
            result.append({"id": row["id"], "task": row["task"],
                           "family": task.family if task else "not_integrated",
                           "authoring": "registered" if registered else "not_integrated",
                           "collection_status": row["status"], "reason": row.get("reason"),
                           "reference_status": row["reference_status"], "source": row["source"],
                           "backend": row.get("backend"),
                           "shape": {key: row.get(key) for key in ("rows", "columns", "depth")}})
        if family is not None:
            if family not in {task.family for task in TASKS} | {"not_integrated"}:
                raise ValueError(f"unknown task family {family!r}")
            result = [row for row in result if row["family"] == family]
        return result
    return [{"id": name, "task": name, "family": task_entry(name).family,
             "owner": task_entry(name).owner, "authoring": "registered",
             "default_matrix": task_entry(name).family in PORTABLE_FAMILIES,
             "matrix_depth_argument": matrix_depth(name, None),
             "shape": shape_description(name), "reference_tasks": references.get(name, [])}
            for name in task_names(suite=suite, family=family)]
