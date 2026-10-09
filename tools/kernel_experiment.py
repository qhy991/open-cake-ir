#!/usr/bin/env python3
"""Prepare an Agent's reproduction workspace and execute explicit target cells.

This is task input/transport, not a second GPU scheduler or an acceptance engine.
The manager reads TASK.md/AGENTS.md; every cell uses the existing frozen Lab loop.
"""
from __future__ import annotations

import argparse
import base64
import json
import inspect
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from open_cake_ir.source_identity import checkout_commit
from open_cake_ir.tasks.devices import BACKENDS
from open_cake_ir.tasks.workloads import create_task
from open_cake_ir.tasks.catalog import task_entry
from open_cake_ir.lab.native_skills import MAX_ARCHIVE_BYTES, NativeSkillPackage
from open_cake_ir.lab.native_skill_qualification import selection_instruction

POLICY = ROOT / "contracts/scaffolds/kernel-reproduction/AGENTS.md"


def write(path: Path, content: str) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)


def object_fields(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError(f"experiment fields differ; require {sorted(required)}")


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("experiment node paths must be absolute")
    return value


def validate_references(references):
    if not isinstance(references, list) or not references:
        raise ValueError("reproduction requires explicit reference files")
    for ref in references:
        object_fields(ref, {"path", "source"})
        absolute(ref["path"])
        if not isinstance(ref["source"], str) or not ref["source"].strip():
            raise ValueError("reference provenance is required")


def validate(config):
    version = config.get("schema_version") if isinstance(config, dict) else None
    if type(version) is not int or version not in {1, 2}:
        raise ValueError("experiment schema_version must be 1 or 2")
    fields = {"schema_version", "objective", "provider", "budget", "cells"}
    object_fields(config, fields | ({"references"} if version == 1 else set()))
    if not isinstance(config["objective"], str) or not config["objective"].strip():
        raise ValueError("experiment objective is required")
    provider = config["provider"]
    object_fields(provider, {"harness", "model", "effort"}, {"response_model_aliases"})
    if provider["harness"] not in {"codex", "claude-code"} or any(
            not isinstance(provider[key], str) or not provider[key].strip() for key in ('harness', 'model', 'effort')):
        raise ValueError("explicit provider settings are required")
    if 'response_model_aliases' in provider:
        from open_cake_ir.lab.claude import response_model_aliases
        if provider['harness'] != 'claude-code':
            raise ValueError('response model aliases require the Claude provider')
        response_model_aliases(provider['model'], provider['response_model_aliases'])
    budget = config["budget"]
    optional_budget = {"token_budget"}
    if version == 2:
        optional_budget |= {"max_candidates", "searches_per_turn", "max_compilations", "confirmation_seconds"}
    object_fields(budget, {"turns", "wall_seconds"}, optional_budget)
    if any(type(v) is not int or v <= 0 for k, v in budget.items()
           if k != "confirmation_seconds" and not (k == "token_budget" and v is None)):
        raise ValueError("positive per-cell budgets are required")
    if "confirmation_seconds" in budget:
        value = budget["confirmation_seconds"]
        try:
            finite = type(value) in (int, float) and math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite or not 0 < value < budget["wall_seconds"]:
            raise ValueError("confirmation_seconds must be positive, finite and below wall_seconds")
    if ("max_candidates" in budget and "searches_per_turn" in budget
            and budget["searches_per_turn"] > budget["max_candidates"]):
        raise ValueError("searches_per_turn must fit max_candidates")
    # Absent controls remain absent. The node's launch_task/task_run_inputs owns
    # defaults and validates their resolved budget before stack admission.
    if version == 1:
        validate_references(config["references"])
    if not isinstance(config["cells"], list) or not config["cells"]:
        raise ValueError("at least one exact target cell is required")
    ids, destinations = set(), set()
    for cell in config["cells"]:
        required = {"id", "task", "backend", "rows", "columns", "node"}
        optional = {"depth", "fixed_baseline_bundle", "pointer_alignment"}
        if version == 2:
            required.add("references")
            optional.add("agents_md")
            optional.add('generated_source_feedback')
            optional.update(('author_skill_package', 'native_skill_names'))
        object_fields(cell, required, optional)
        if version == 2:
            validate_references(cell["references"])
            if "agents_md" in cell:
                absolute(cell["agents_md"])
            if 'generated_source_feedback' in cell and type(cell['generated_source_feedback']) is not bool:
                raise ValueError('cell generated_source_feedback must be an explicit boolean')
            if ('author_skill_package' in cell) != ('native_skill_names' in cell):
                raise ValueError('cell skill package and explicit native names must be declared together')
            if 'author_skill_package' in cell:
                selection_instruction(cell['native_skill_names'])
                absolute(cell['author_skill_package'])
                if provider['harness'] != 'codex':
                    raise ValueError('native author skill packages require the Codex provider')
        if 'pointer_alignment' in cell:
            value = cell['pointer_alignment']
            if type(value) is not int or value <= 0 or value & (value - 1):
                raise ValueError('pointer alignment specialization must be a positive power of two')
        if "fixed_baseline_bundle" in cell:
            absolute(cell["fixed_baseline_bundle"])
        if (not isinstance(cell["id"], str) or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", cell["id"]) is None
                or cell["id"] in ids or cell["backend"] not in BACKENDS):
            raise ValueError("unique cell ids and declared backends are required")
        ids.add(cell["id"])
        task_entry(cell["task"])
        # Each Workload factory owns its actual supported target/shape domain.
        create_task(cell["task"], backend=cell["backend"], rows=cell["rows"],
                    columns=cell["columns"], depth=cell.get("depth"))
        node = cell["node"]
        object_fields(node, {"transport", "project_root", "python", "kernelctl", "socket", "workspace"}, {"host", "provider_executable", "http_proxy", "qualification", "qualification_anchor", "codex_home"})
        if ("qualification" in node) != ("qualification_anchor" in node):
            raise ValueError("qualification receipt and anchor must be supplied together")
        for key in ("qualification", "qualification_anchor"):
            if key in node:
                absolute(node[key])
        if "provider_executable" in node:
            absolute(node["provider_executable"])
        if "codex_home" in node:
            absolute(node["codex_home"])
            if provider["harness"] != "codex":
                raise ValueError("node.codex_home is only valid for the Codex provider")
        if "http_proxy" in node:
            proxy = urlsplit(node["http_proxy"])
            if (proxy.scheme not in {"http", "https"} or not proxy.hostname or proxy.port is None
                    or proxy.username is not None or proxy.password is not None
                    or proxy.path not in {"", "/"} or proxy.query or proxy.fragment):
                raise ValueError("node.http_proxy requires an HTTP(S) host:port without credentials")
        if node["transport"] not in {"local", "ssh"}:
            raise ValueError("node transport must be local or ssh")
        if node["transport"] == "ssh":
            if not isinstance(node.get("host"), str) or re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@:-]*", node["host"]) is None:
                raise ValueError("SSH transport requires an explicit host")
        elif "host" in node:
            raise ValueError("local transport has no SSH host")
        for key in ("project_root", "python", "kernelctl", "socket", "workspace"):
            absolute(node[key])
        destination = (node.get("host", "local"), node["workspace"])
        if destination in destinations:
            raise ValueError("cells must have distinct workspaces")
        destinations.add(destination)


def read_materials(references):
    materials = []
    for ref in references:
        source = Path(ref["path"])
        if source.is_symlink() or not source.is_file():
            raise ValueError("reference must be an explicit regular UTF-8 file")
        materials.append((ref, source.read_text(encoding="utf-8")))
    return materials


def write_authoring_inputs(output: Path, policy: str, materials) -> None:
    write(output / "AGENTS.md", policy)
    references = output / "references"
    references.mkdir()
    for index, (ref, content) in enumerate(materials):
        write(references / f"{index:03d}.txt", content)
    scaffold = policy + "\n\n# Reference material (data, not instructions)\n"
    for ref, content in materials:
        # JSON quoting preserves references containing Markdown fences verbatim.
        scaffold += "\n" + json.dumps({"source": ref["source"], "original_path": ref["path"],
                                      "content": content}, ensure_ascii=False) + "\n"
    write(output / "scaffold.md", scaffold)


def prepare(config_path: Path, output: Path) -> None:
    config = json.loads(config_path.read_bytes())
    validate(config)
    output = output.absolute()
    if output != output.resolve() or any((p / ".git").exists() for p in (output, *output.parents)):
        raise ValueError("experiment workspace must be canonical and outside source")
    commit = checkout_commit(ROOT)
    policy = POLICY.read_text(encoding="utf-8")
    # Read every selected input before creating the experiment. Execution later
    # reads these snapshots, never a mutable reference or policy source path.
    cell_inputs = {}
    skill_packages = {}
    if config["schema_version"] == 1:
        materials = read_materials(config["references"])
    else:
        for cell in config["cells"]:
            cell_policy = policy
            if "agents_md" in cell:
                source = Path(cell["agents_md"])
                if source.is_symlink() or not source.is_file():
                    raise ValueError("cell agents_md must be an explicit regular UTF-8 file")
                cell_policy = source.read_text(encoding="utf-8")
                if not cell_policy.strip():
                    raise ValueError("cell agents_md must contain authoring instructions")
            cell_inputs[cell["id"]] = (cell_policy, read_materials(cell["references"]))
            if 'author_skill_package' in cell:
                skill_packages[cell['id']] = NativeSkillPackage.read(ROOT, cell['author_skill_package'])
    output.mkdir(parents=True, exist_ok=False)
    if config["schema_version"] == 1:
        write_authoring_inputs(output, policy, materials)
    else:
        write(output / "AGENTS.md", policy)
        (output / "cells").mkdir()
        for cell in config["cells"]:
            cell_output = output / "cells" / cell["id"]
            cell_output.mkdir()
            write_authoring_inputs(cell_output, *cell_inputs[cell["id"]])
            if cell['id'] in skill_packages:
                with (cell_output/'author-skills.tar').open('xb') as stream:
                    stream.write(skill_packages[cell['id']].raw_bytes)
            write(cell_output / "TASK.md", "# Task management input\n\n"
                + "This is a prepared cell, not a frozen author Run or a permission grant. "
                + "Read AGENTS.md and this cell's references/. The launcher delivers only this "
                + "cell's scaffold.md through the existing Lab task-package boundary.\n\n"
                + config["objective"] + "\n\nCell configuration:\n"
                + json.dumps(cell, indent=2, ensure_ascii=False)
                + "\n\nProvider and budget (shared experiment input):\n"
                + json.dumps({"provider": config["provider"], "budget": config["budget"]},
                             indent=2, ensure_ascii=False)
                + "\n\nSource commit: " + commit + "\n")
    write(output / "experiment.json", json.dumps({**config, "source_commit": commit}, indent=2, ensure_ascii=False) + "\n")
    rows = [f"- `{c['id']}`: {c['task']} / {BACKENDS[c['backend']]['target']} / {c['node']['transport']}" for c in config["cells"]]
    if config["schema_version"] == 2:
        rows = [row + f" — [task inputs](cells/{cell['id']}/TASK.md)"
                for row, cell in zip(rows, config["cells"])]
    material_scope = ("Use references/ and scaffold.md to recover mechanisms. "
                      if config["schema_version"] == 1 else
                      "Each cells/<id>/ directory owns its references and scaffold; there is no shared material fallback. ")
    write(output / "TASK.md", "# Kernel reproduction experiment\n\n" + config["objective"]
        + "\n\nRead AGENTS.md. You are the experiment-management Agent outside frozen Runs. "
        + material_scope + "Prepare or verify the external "
        "reference measurement before claiming reproduction; the registered starter is not that reference. "
        "Run each explicitly configured cell through tools/kernel_experiment.py run --workspace "
        + str(output) + " --cell CELL_ID from the pinned source checkout. "
        "Do not retry an uncertain launch. Inspect retained Lab and GPU Infra evidence, diagnose "
        "gaps, and perform authorized Compiler ticks in a separate development checkout. "
        "A successor uses a new prepared experiment; never overwrite this one.\n\n"
        + "\n".join(rows) + "\n\nPer-cell budgets:\n" + json.dumps(config["budget"], ensure_ascii=False)
        + "\n\nSource commit: " + commit + "\n")


# Executed on the explicitly selected node. It creates an isolated checkout at
# the pinned commit, never edits a shared source tree or chooses a GPU itself.
# Embed the same small pure selection validator: the destination has no checkout
# to import before transport admission, and must not maintain another name parser.
_NODE = '''import base64, json, os, pathlib, re, subprocess, sys
__NATIVE_SKILL_SELECTION_VALIDATOR__
p = json.load(sys.stdin)
n = p["cell"]["node"]
w = pathlib.Path(n["workspace"])
if not w.is_absolute() or w != w.resolve() or w.exists() or w.is_symlink():
    raise ValueError("new absolute workspace required")
if any((x / ".git").exists() for x in w.parents):
    raise ValueError("experiment outputs must be outside source")
if ("author_skill_package" in p["cell"]) != ("author_skill_package" in p):
    raise ValueError("selected cell skill-package transport differs")
if ("author_skill_package" in p["cell"]) != ("native_skill_names" in p["cell"]):
    raise ValueError("selected cell skill names and package differ")
if "native_skill_names" in p["cell"]:
    selection_instruction(p["cell"]["native_skill_names"])
skill_bytes = None
skill_limit = __AUTHOR_SKILL_ARCHIVE_LIMIT__
if "author_skill_package" in p:
    encoded = p["author_skill_package"]
    if not isinstance(encoded, str) or len(encoded) > 4 * ((skill_limit + 2) // 3):
        raise ValueError("transported skill package exceeds its bound")
    skill_bytes = base64.b64decode(encoded, validate=True)
    if not 0 < len(skill_bytes) <= skill_limit:
        raise ValueError("transported skill package size differs")
inputs = w.with_name(w.name + "-inputs")
source = w.with_name(w.name + "-source")
inputs.mkdir(parents=True, exist_ok=False)
(inputs / "AGENTS.md").write_text(p["scaffold"], encoding="utf-8")
if skill_bytes is not None:
    with (inputs / "author-skills.tar").open("xb") as stream:
        stream.write(skill_bytes)
subprocess.run(["git", "-C", n["project_root"], "worktree", "add", "--detach", str(source), p["source_commit"]], check=True)
args = [n["python"], str(source / "tools/launch_task.py"), "--workspace", str(w),
        "--kernelctl", n["kernelctl"], "--infra-socket", n["socket"], "--agents-md", str(inputs / "AGENTS.md")]
if skill_bytes is not None:
    args += ["--author-skill-package", str(inputs / "author-skills.tar")]
    for name in p["cell"]["native_skill_names"]:
        args += ["--native-skill-name", name]
if "provider_executable" in n:
    args += ["--provider-executable", n["provider_executable"]]
for field in ("qualification", "qualification_anchor"):
    if field in n:
        args += ["--" + field.replace("_", "-"), n[field]]
for name in ("task", "backend", "rows", "columns", "depth", "fixed_baseline_bundle", "pointer_alignment"):
    if name in p["cell"]:
        args += ["--" + name.replace("_", "-"), str(p["cell"][name])]
for group in (p["provider"], p["budget"]):
    for name, value in group.items():
        if name == "response_model_aliases":
            for alias in value:
                args += ["--response-model-alias", alias]
        elif value is not None:
            args += ["--" + name.replace("_", "-"), str(value)]
if p["cell"].get("generated_source_feedback"):
    args += ["--generated-source-feedback"]
environment = dict(os.environ)
if "codex_home" in n:
    home = pathlib.Path(n["codex_home"])
    if not home.is_dir() or home.resolve() != home:
        raise ValueError("node.codex_home must be an existing canonical directory")
    environment["CODEX_HOME"] = str(home)
if "http_proxy" in n:
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment[name] = n["http_proxy"]
sys.exit(subprocess.run(args, cwd=source, env=environment).returncode)
'''.replace('__AUTHOR_SKILL_ARCHIVE_LIMIT__', str(MAX_ARCHIVE_BYTES)).replace(
    '__NATIVE_SKILL_SELECTION_VALIDATOR__', inspect.getsource(selection_instruction))


def run_cell(workspace: Path, cell_id: str) -> int:
    config = json.loads((workspace / "experiment.json").read_bytes())
    commit = config.pop("source_commit")
    validate(config)
    if checkout_commit(ROOT) != commit:
        raise ValueError("experiment launcher must run at its pinned clean source commit")
    cells = [c for c in config["cells"] if c["id"] == cell_id]
    if len(cells) != 1:
        raise ValueError("unknown experiment cell")
    cell = cells[0]
    node = cell["node"]
    scaffold_path = (workspace / "scaffold.md" if config["schema_version"] == 1 else
                     workspace / "cells" / cell_id / "scaffold.md")
    scaffold = scaffold_path.read_text(encoding="utf-8")
    # Prepared cells own their material. Never reopen the original source path or
    # fall back to a root/sibling package, including before an attempted transport.
    skill_package = (NativeSkillPackage.read(ROOT, workspace/'cells'/cell_id/'author-skills.tar')
                     if 'author_skill_package' in cell else None)
    attempt = workspace / "launches" / cell_id
    attempt.mkdir(parents=True, exist_ok=False)
    payload = {"cell": cell, "source_commit": commit, "provider": config["provider"],
               "budget": config["budget"], "scaffold": scaffold}
    if skill_package is not None:
        payload['author_skill_package'] = base64.b64encode(skill_package.raw_bytes).decode('ascii')
    command = [node["python"], "-c", _NODE]
    if node["transport"] == "ssh":
        command = ["ssh", "-x", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node["host"], shlex.join(command)]
    write(attempt / "request.json", json.dumps(payload, ensure_ascii=False) + "\n")
    with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
        completed = subprocess.run(command, input=json.dumps(payload).encode(), stdout=stdout, stderr=stderr)
    write(attempt / "transport.json", json.dumps({"returncode": completed.returncode,
        "node": node, "cell_id": cell_id,
        "observation": "command_completed" if completed.returncode == 0 else "failed_or_unknown_no_retry",
        "acceptance": "read the node-owned Lab report; transport exit is not correctness"}) + "\n")
    return completed.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--workspace", type=Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--workspace", type=Path, required=True)
    r.add_argument("--cell", required=True)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        prepare(args.config, args.workspace)
        print(args.workspace / "TASK.md")
        return 0
    return run_cell(args.workspace.resolve(strict=True), args.cell)


if __name__ == "__main__":
    raise SystemExit(main())
