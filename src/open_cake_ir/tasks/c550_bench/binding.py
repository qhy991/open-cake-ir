"""Import original Bench cases without replacing their inputs or numerical oracle.

The external checkout owns definitions, scalar values and comparison. This module
owns the binding to one fixed-shape CAKE Workload. Raw definitions stay outside
the repository and outside the candidate's authoring directory.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
from typing import Any

from open_cake_ir.source_identity import checkout_commit

BENCH_COMMIT = "ababa4c0656b89bdf0a9c60ef0d2e90e0aebb425"
OPERATOR = "c550_bench_original_case"
TARGET = "xcore1002"
DTYPES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
          "int32": "int32", "int64": "int64", "bool": "bool"}


@dataclass(frozen=True)
class BenchProblem:
    root: Path
    task_id: str
    api: Any
    definition: Any
    workloads: tuple
    raw_workloads: tuple
    task_root: Path

    @classmethod
    def open(cls, root: Path, task_id: str) -> "BenchProblem":
        root = root.resolve(strict=True)
        if checkout_commit(root) != BENCH_COMMIT:
            raise ValueError("C550 Bench requires its clean, fixed ababa4c0 checkout")
        spec = importlib.util.spec_from_file_location("cake_original_c550_bench", root / "c550bench.py")
        api = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api)
        upstream = api.upstream()
        import sol_execbench.core.bench.correctness as comparison
        if not Path(comparison.__file__).resolve().is_relative_to(upstream.resolve()):
            raise ValueError("The loaded Bench comparison module came from another source root")
        definition, workloads, raw, task_root = api.load_problem(api.task_record(task_id))
        if len(workloads) != 16 or len({row.uuid for row in workloads}) != 16:
            raise ValueError("A C550 Bench task must retain all 16 original workload IDs")
        return cls(root, task_id, api, definition, tuple(workloads), tuple(raw), task_root)

    def selected(self, uuid: str):
        for workload, raw in zip(self.workloads, self.raw_workloads, strict=True):
            if workload.uuid == uuid:
                return workload, raw
        raise ValueError("The selected workload ID is absent from the original Bench task")

    def workload_document(self, uuid: str) -> dict:
        selected, raw = self.selected(uuid)
        definition = self.definition
        tensors, scalars = {}, {}
        inputs, outputs = [], []
        for mode, specs, shapes in (
            ("input", definition.inputs, definition.get_input_shapes(selected.axes)),
            ("output", definition.outputs, definition.get_output_shapes(selected.axes)),
        ):
            for name, item in specs.items():
                shape = shapes[name]
                if shape is None:
                    value = raw.get("inputs", {}).get(name)
                    if mode != "input" or not isinstance(value, dict) or value.get("type") != "scalar":
                        raise ValueError("Only original fixed scalar inputs may be specialized")
                    scalars[name] = {"dtype": item.dtype.value, "value": value["value"]}
                    continue
                if (not shape or any(type(n) is not int or n <= 0 for n in shape)
                        or item.dtype.value not in DTYPES or name in tensors):
                    raise ValueError("The original tensor ABI needs an explicit supported storage binding")
                tensors[name] = {"shape": list(shape), "dtype": DTYPES[item.dtype.value],
                                 "layout": "contiguous_row_major"}
                (inputs if mode == "input" else outputs).append(name)
        if not inputs or not outputs:
            raise ValueError("Bench tasks require explicit tensor inputs and outputs")
        seed = self.api.document("suite.json")["seed"]
        return {
            "schema_version": 1, "workload_id": f"c550-bench-{self.task_id.replace('/', '-')}-{uuid}",
            "revision": "1", "state": "frozen", "operator": OPERATOR,
            "provenance": [{"kind": "original_c550_bench", "path": str(self.root),
                            "commit": BENCH_COMMIT, "task": self.task_id, "workload_uuid": uuid}],
            "cases": [{"case_id": "primary", "shape": dict(selected.axes),
                       "seed": seed, "mode": "original_bench_input_factory"}],
            "tensors": tensors,
            "semantics": {
                "target": TARGET, "candidate_abi": {"inputs": inputs, "outputs": outputs},
                "input_effects": "unchanged", "output_storage": "fresh_contiguous_nonaliasing",
                "benchmark": {"root": str(self.root), "commit": BENCH_COMMIT,
                              "task": self.task_id, "workload_uuid": uuid},
                "fixed_scalar_inputs": scalars,
                "original_input_specifications": raw["inputs"],
                "ordered_original_inputs": list(definition.inputs),
                "ordered_original_outputs": list(definition.outputs),
                "oracle_preparation": "original_factory_and_reference_on_leased_target",
                "search_scope": "one_original_case_one_seed",
                "final_acceptance": "original_all_16_workloads_10_rounds",
            },
            "oracle": {"kind": "original_c550_bench_reference", "device": TARGET,
                       "reference_access": "high_level_reference_read",
                       "reference_source": definition.reference,
                       "custom_inputs_entrypoint": definition.custom_inputs_entrypoint,
                       "implementation": "fixed_Bench_load_problem_and_reference_run"},
            "validation": {"primary_case": "primary", "all_cases_required": True,
                           "comparison": "original_c550_bench",
                           "raw_tolerance": raw.get("tolerance", {}),
                           "effective_tolerance": selected.tolerance.model_dump(),
                           "qualification": "single_original_case_search_only"},
        }

    def prepare_on_target(self, uuid: str, *, runtime_library: str):
        """Run the original factory/reference under an already acquired MACA lease.

        Return CPU tensors with their original storage dtype. No conversion to
        Python float lists or substitute CPU mathematical reference is allowed.
        Callers must keep this phase outside all timed candidate intervals.
        """
        from open_cake_ir.evaluation.triton_metax import observe_local_metax
        admission = observe_local_metax(TARGET, runtime_library=runtime_library)
        import torch
        from sol_execbench.core.bench.correctness import set_seed
        from sol_execbench.core.bench.io import gen_inputs, normalize_outputs
        from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype

        selected, raw = self.selected(uuid)
        reference = self.api.load_module(self.task_root / "reference.py", "cake_bench_reference")
        custom = (getattr(reference, self.definition.custom_inputs_entrypoint)
                  if self.definition.custom_inputs_entrypoint else None)
        set_seed(self.api.document("suite.json")["seed"])
        arguments = gen_inputs(self.definition, selected, "cuda:0", custom_inputs_fn=custom)
        original = {name: value.detach().cpu().clone() for name, value
                    in zip(self.definition.inputs, arguments, strict=True)
                    if isinstance(value, torch.Tensor)}
        expected = reference.run(*self.api.cloned_inputs(arguments))
        torch.cuda.synchronize(0)
        names = list(self.definition.outputs)
        expected = normalize_outputs(expected, device="cuda:0", output_names=names,
            output_dtypes={name: dtype_str_to_torch_dtype(item.dtype)
                           for name, item in self.definition.outputs.items()})
        expected = {name: value.detach().cpu().clone() for name, value in expected.items()}
        declared_scalars = self.workload_document(uuid)["semantics"]["fixed_scalar_inputs"]
        for name, value in zip(self.definition.inputs, arguments, strict=True):
            if not isinstance(value, torch.Tensor) and value != declared_scalars[name]["value"]:
                raise ValueError("Original input generation changed a specialized scalar value")
        if any(not value.is_contiguous() for value in (*original.values(), *expected.values())):
            raise ValueError("Original Bench storage requires a separately qualified strided ABI")
        return original, expected, admission

    def compare(self, uuid: str, expected, observed) -> dict:
        """Delegate the verdict and all error metrics to the original comparator."""
        selected, _ = self.selected(uuid)
        return self.api.compare_outputs(observed, expected, self.definition,
                                        selected.tolerance, selected.axes)


def validate_document(document):
    binding = document["semantics"]["benchmark"]
    problem = BenchProblem.open(Path(binding["root"]), binding["task"])
    expected = problem.workload_document(binding["workload_uuid"])
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Bench Workload differs from its original ABI, scalar, oracle or tolerance")
