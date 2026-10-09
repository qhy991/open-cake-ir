"""Import original Bench cases without replacing their inputs or numerical oracle.

The external checkout owns definitions, scalar values and comparison. This module
owns the binding to one fixed-shape CAKE Workload. Raw definitions stay outside
the repository and outside the candidate's authoring directory.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import ast
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from open_cake_ir.source_identity import checkout_commit

BENCH_COMMIT = "ababa4c0656b89bdf0a9c60ef0d2e90e0aebb425"
OPERATOR = "c550_bench_original_case"
TARGET = "xcore1002"
DTYPES = {"float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
          "int32": "int32", "int64": "int64", "bool": "bool"}


_ORACLE_BOOLEAN_SETTINGS = (
    "allow_tf32", "allow_fp16_reduced_precision_reduction",
    "allow_bf16_reduced_precision_reduction",
)
_ORACLE_INITIALIZATION_OVERRIDE = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"


def validate_oracle_numerics(value) -> dict:
    """Copy the explicitly observed reference policy, without consulting Torch."""
    keys = {"float32_matmul_precision", "initialization", *_ORACLE_BOOLEAN_SETTINGS}
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError("Bench oracle numerics requires all four effective settings and initialization source")
    if (value["float32_matmul_precision"] not in ("highest", "high", "medium")
            or any(type(value[name]) is not bool for name in _ORACLE_BOOLEAN_SETTINGS)):
        raise ValueError("Bench oracle numerics has an invalid effective matrix setting")
    initialization = value["initialization"]
    if (not isinstance(initialization, Mapping)
            or set(initialization) != {_ORACLE_INITIALIZATION_OVERRIDE}
            or (initialization[_ORACLE_INITIALIZATION_OVERRIDE] is not None
                and not isinstance(initialization[_ORACLE_INITIALIZATION_OVERRIDE], str))):
        raise ValueError("Bench oracle numerics requires its observed initialization override")
    return {"float32_matmul_precision": value["float32_matmul_precision"],
            **{name: value[name] for name in _ORACLE_BOOLEAN_SETTINGS},
            "initialization": dict(initialization)}


def observe_oracle_numerics() -> dict:
    """Read effective Torch matrix settings and the narrow initialization input."""
    import torch
    return validate_oracle_numerics({
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        **{name: getattr(torch.backends.cuda.matmul, name) for name in _ORACLE_BOOLEAN_SETTINGS},
        "initialization": {_ORACLE_INITIALIZATION_OVERRIDE: os.environ.get(_ORACLE_INITIALIZATION_OVERRIDE)},
    })


def require_oracle_numerics(expected, *, phase: str) -> None:
    """Refuse drift at the caller's oracle boundary; never change math settings."""
    expected = validate_oracle_numerics(expected)
    observed = observe_oracle_numerics()
    changed = [name for name in expected if expected[name] != observed[name]]
    if changed:
        raise ValueError(f"Bench oracle numerics changed at {phase}: {', '.join(changed)}")


def factory_scalar(definition, name: str):
    """Resolve a literal original return value without executing the input factory.

    Runtime preparation still checks the actual scalar. A dynamic scalar or a
    reassigned value needs an explicit ABI; it cannot inherit a guessed constant.
    """
    functions = [node for node in ast.parse(definition.reference).body
                 if isinstance(node, ast.FunctionDef)
                 and node.name == definition.custom_inputs_entrypoint]
    if len(functions) != 1 or not functions[0].body or not isinstance(functions[0].body[-1], ast.Return):
        raise ValueError(f"Original custom scalar {name!r} requires an explicit factory binding")
    factory = functions[0]
    result = factory.body[-1].value
    if isinstance(result, ast.Dict):
        keys = [node.value for node in result.keys
                if isinstance(node, ast.Constant) and isinstance(node.value, str)]
        if len(keys) != len(result.keys) or len(set(keys)) != len(keys) or name not in keys:
            raise ValueError(f"Original custom scalar {name!r} has no static named return binding")
        expression = result.values[keys.index(name)]
    elif isinstance(result, (ast.List, ast.Tuple)) and len(result.elts) == len(definition.inputs):
        expression = result.elts[list(definition.inputs).index(name)]
    else:
        raise ValueError(f"Original custom scalar {name!r} has no static ordered return binding")
    if isinstance(expression, ast.Name):
        symbol = expression.id
        writes = [node for node in ast.walk(factory)
                  if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == symbol]
        assignments = [node for node in factory.body
                       if isinstance(node, ast.Assign) and len(node.targets) == 1
                       and isinstance(node.targets[0], ast.Name) and node.targets[0].id == symbol]
        if len(writes) != 1 or len(assignments) != 1:
            raise ValueError(f"Original custom scalar {name!r} is not one immutable literal")
        expression = assignments[0].value
    try:
        value = ast.literal_eval(expression)
    except (ValueError, TypeError, SyntaxError) as error:
        raise ValueError(f"Original custom scalar {name!r} is not a literal") from error
    if type(value) not in (int, float, bool):
        raise ValueError(f"Original custom scalar {name!r} is not a numeric scalar")
    return value


def physical_input_view(value, order):
    """Expose a declared dense physical order without copying or computing data."""
    order = tuple(order)
    if any(type(axis) is not int for axis in order) or sorted(order) != list(range(value.ndim)):
        raise ValueError('Bench input view must be a complete axis permutation')
    physical = value.permute(order)
    inverse = tuple(order.index(axis) for axis in range(value.ndim))
    restored = physical.permute(inverse)
    if (not physical.is_contiguous() or physical.data_ptr() != value.data_ptr()
            or physical.storage_offset() != value.storage_offset()
            or physical.untyped_storage().nbytes() != value.untyped_storage().nbytes()
            or tuple(restored.shape) != tuple(value.shape)
            or tuple(restored.stride()) != tuple(value.stride())):
        raise ValueError('Original Bench input does not admit the declared zero-copy dense permutation')
    return physical


def validate_input_view_observation(problem, observation):
    """Check complete original-case metadata before selecting a physical ABI."""
    from open_cake_ir.evaluation.triton_metax import validate_maca_admission
    if (observation.get('bench_commit') != BENCH_COMMIT or observation.get('status') != 'complete'
            or observation.get('target') != TARGET or observation.get('task') != problem.task_id
            or observation.get('scope') != 'original_input_factory_metadata_only'):
        raise ValueError('Bench input-view observation identity or scope differs')
    rows = observation.get('cases')
    if (not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows)
            or [row.get('workload_uuid') for row in rows] != [item.uuid for item in problem.workloads]):
        raise ValueError('Bench input-view observation must cover the 16 original cases in order')
    result = {}
    for row in rows:
        selected, _ = problem.selected(row['workload_uuid'])
        shapes = {name: tuple(shape) for name, shape in problem.definition.get_input_shapes(selected.axes).items()
                  if shape is not None}
        metadata, views = row.get('metadata'), row.get('input_views')
        if (row.get('zero_copy_verified') is not True or not isinstance(metadata, dict)
                or not isinstance(views, dict) or set(metadata) != set(shapes) or set(views) != set(shapes)):
            raise ValueError('Bench input-view metadata must cover every original tensor')
        admission = row.get('device_admission')
        validate_maca_admission(admission, target_id=TARGET, job_id=observation.get('job_id'))
        if admission['pci_bus_id'] != observation.get('expected_pci'):
            raise ValueError('Bench input-view device differs from the physical lease')
        for name, shape in shapes.items():
            item, order = metadata[name], views[name]
            if (not isinstance(item, dict) or not isinstance(item.get('shape'), list)
                    or any(type(n) is not int for n in item['shape']) or item['shape'] != list(shape)
                    or item.get('dtype') != 'torch.' + problem.definition.inputs[name].dtype.value
                    or item.get('physical_axes') != order or not isinstance(order, list)
                    or any(type(axis) is not int for axis in order) or sorted(order) != list(range(len(shape)))):
                raise ValueError('Bench input-view original shape, dtype or permutation differs')
            strides = item.get('strides')
            if not isinstance(strides, list) or len(strides) != len(shape) or any(type(n) is not int or n < 0 for n in strides):
                raise ValueError('Bench input-view original strides differ')
            step = 1
            for axis in reversed(order):
                if shape[axis] != 1 and strides[axis] != step:
                    raise ValueError('Bench input-view metadata does not describe a dense permutation')
                step *= shape[axis]
        result[row['workload_uuid']] = views
    return result


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

    def workload_document(self, uuid: str, *, oracle_numerics, input_views=None) -> dict:
        oracle_numerics = validate_oracle_numerics(oracle_numerics)
        selected, raw = self.selected(uuid)
        definition = self.definition
        tensors, scalars, views, original_shapes = {}, {}, {}, {}
        requested_views = {} if input_views is None else input_views
        inputs, outputs = [], []
        for mode, specs, shapes in (
            ("input", definition.inputs, definition.get_input_shapes(selected.axes)),
            ("output", definition.outputs, definition.get_output_shapes(selected.axes)),
        ):
            for name, item in specs.items():
                shape = shapes[name]
                if shape is None:
                    value = raw.get("inputs", {}).get(name)
                    if mode != "input" or not isinstance(value, dict) or value.get("type") not in {"scalar", "custom"}:
                        raise ValueError("Only original fixed scalar inputs may be specialized")
                    scalar = value["value"] if value["type"] == "scalar" else factory_scalar(definition, name)
                    scalars[name] = {"dtype": item.dtype.value, "value": scalar,
                                     "binding": "literal_input" if value["type"] == "scalar" else "original_factory_literal"}
                    continue
                if (not shape or any(type(n) is not int or n <= 0 for n in shape)
                        or item.dtype.value not in DTYPES or name in tensors):
                    raise ValueError("The original tensor ABI needs an explicit supported storage binding")
                original_shapes[name] = list(shape)
                if mode == 'input':
                    order = requested_views.get(name, list(range(len(shape))))
                    if (not isinstance(order, (list, tuple)) or any(type(n) is not int for n in order)
                            or sorted(order) != list(range(len(shape)))):
                        raise ValueError('Bench input view must be one complete axis permutation')
                    views[name] = list(order)
                    shape = tuple(shape[axis] for axis in order)
                tensors[name] = {"shape": list(shape), "dtype": DTYPES[item.dtype.value],
                                 "layout": "contiguous_row_major"}
                (inputs if mode == "input" else outputs).append(name)
        if not inputs or not outputs:
            raise ValueError("Bench tasks require explicit tensor inputs and outputs")
        if set(requested_views) - set(inputs):
            raise ValueError('Bench input views name a scalar, output or unknown tensor')
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
                "original_tensor_shapes": original_shapes,
                "input_views": views,
                "input_view_contract": "zero_copy_dense_axis_permutation_before_native_submission",
                "oracle_numerics": oracle_numerics,
                "oracle_numerics_contract": (
                    "The original reference uses these observed Torch matrix settings. "
                    "FP32 storage does not imply IEEE FP32 matrix multiplication. "
                    "Preserve the original comparator; validate arithmetic rewrites on the target. "
                    "The initialization override records this environment, not a Target capability."
                ),
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

    def original_inputs_on_target(self, uuid: str, *, runtime_library: str, oracle_numerics):
        """Materialize the original inputs only after verifying their device lease."""
        from open_cake_ir.evaluation.triton_metax import observe_local_metax
        admission = observe_local_metax(TARGET, runtime_library=runtime_library)
        from sol_execbench.core.bench.correctness import set_seed
        from sol_execbench.core.bench.io import gen_inputs
        selected, _ = self.selected(uuid)
        reference = self.api.load_module(self.task_root / "reference.py", "cake_bench_reference")
        custom = (getattr(reference, self.definition.custom_inputs_entrypoint)
                  if self.definition.custom_inputs_entrypoint else None)
        set_seed(self.api.document("suite.json")["seed"])
        require_oracle_numerics(oracle_numerics, phase="before_input_factory")
        try:
            arguments = gen_inputs(self.definition, selected, "cuda:0", custom_inputs_fn=custom)
        finally:
            require_oracle_numerics(oracle_numerics, phase="after_input_factory")
        return arguments, reference, admission

    def discover_input_views_on_target(self, uuid: str, *, runtime_library: str):
        """Observe concrete input layouts for a later frozen Workload binding."""
        observed_numerics = observe_oracle_numerics()
        arguments, _, admission = self.original_inputs_on_target(uuid, runtime_library=runtime_library,
            oracle_numerics=observed_numerics)
        import torch
        views, metadata = {}, {}
        for name, value in zip(self.definition.inputs, arguments, strict=True):
            if not isinstance(value, torch.Tensor):
                continue
            order = (list(range(value.ndim)) if value.is_contiguous() else
                     sorted(range(value.ndim), key=lambda axis: (-value.stride(axis), axis)))
            physical_input_view(value, order)
            views[name] = order
            metadata[name] = {'shape': list(value.shape), 'strides': list(value.stride()),
                              'dtype': str(value.dtype), 'physical_axes': order}
        torch.cuda.synchronize(0)
        return views, metadata, admission

    def prepare_on_target(self, uuid: str, *, runtime_library: str, oracle_numerics, input_views=None):
        """Run the original factory/reference under an already acquired MACA lease.

        Return CPU tensors with their original storage dtype. No conversion to
        Python float lists or substitute CPU mathematical reference is allowed.
        Callers must keep this phase outside all timed candidate intervals.
        """
        arguments, reference, admission = self.original_inputs_on_target(uuid, runtime_library=runtime_library,
            oracle_numerics=oracle_numerics)
        import torch
        from sol_execbench.core.bench.io import normalize_outputs
        from sol_execbench.core.data.dtypes import dtype_str_to_torch_dtype

        document = self.workload_document(uuid, input_views=input_views, oracle_numerics=oracle_numerics)
        views = document['semantics']['input_views']
        original = {name: physical_input_view(value, views[name]).detach().cpu().clone() for name, value
                    in zip(self.definition.inputs, arguments, strict=True)
                    if isinstance(value, torch.Tensor)}
        require_oracle_numerics(oracle_numerics, phase="before_reference")
        try:
            expected = reference.run(*self.api.cloned_inputs(arguments))
        finally:
            require_oracle_numerics(oracle_numerics, phase="after_reference")
        torch.cuda.synchronize(0)
        names = list(self.definition.outputs)
        expected = normalize_outputs(expected, device="cuda:0", output_names=names,
            output_dtypes={name: dtype_str_to_torch_dtype(item.dtype)
                           for name, item in self.definition.outputs.items()})
        expected = {name: value.detach().cpu().clone() for name, value in expected.items()}
        declared_scalars = document["semantics"]["fixed_scalar_inputs"]
        for name, value in zip(self.definition.inputs, arguments, strict=True):
            if not isinstance(value, torch.Tensor) and value != declared_scalars[name]["value"]:
                raise ValueError("Original input generation changed a specialized scalar value")
        if any(not value.is_contiguous() for value in original.values()):
            raise ValueError("Original Bench storage requires a separately qualified strided ABI")
        return original, expected, admission

    def compare(self, uuid: str, expected, observed) -> dict:
        """Delegate the verdict and all error metrics to the original comparator."""
        selected, _ = self.selected(uuid)
        expected = dict(expected) if isinstance(expected, Mapping) else expected
        observed = dict(observed) if isinstance(observed, Mapping) else observed
        return self.api.compare_outputs(observed, expected, self.definition,
                                        selected.tolerance, selected.axes)


def validate_document(document):
    numerics = validate_oracle_numerics(document["semantics"].get("oracle_numerics"))
    binding = document["semantics"]["benchmark"]
    problem = BenchProblem.open(Path(binding["root"]), binding["task"])
    expected = problem.workload_document(binding["workload_uuid"], input_views=document['semantics']['input_views'],
        oracle_numerics=numerics)
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True, allow_nan=False):
        raise ValueError("Bench Workload differs from its original ABI, scalar, oracle or tolerance")
