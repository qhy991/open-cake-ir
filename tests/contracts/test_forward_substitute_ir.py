"""Canonical BF16-coefficient FP32 row-wise strict-lower solve boundary."""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from open_cake_ir.compiler import Compiler, Schedule, Target
from open_cake_ir.compiler.backends import native_cuda, triton
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler.ir import ForwardSubstituteParameters, ScheduleParseError
from open_cake_ir.compiler.ir import DType
from open_cake_ir.compiler.performance.work import work_bound
from open_cake_ir.compiler.performance.profile import profile_envelope
from open_cake_ir.compiler.schema import schedule_schema
from open_cake_ir.compiler.verifier import verify


ROOT = Path(__file__).resolve().parents[2]


def document() -> dict:
    buffers = [
        {"name": "p", "space": "global", "dtype": "bf16", "shape": [32, 32], "mode": "input"},
        {"name": "rhs", "space": "global", "dtype": "fp32", "shape": [128, 32], "mode": "input"},
        {"name": "out", "space": "global", "dtype": "fp32", "shape": [128, 32], "mode": "output"},
        {"name": "p_tile", "space": "register", "dtype": "bf16", "shape": [32, 32], "mode": "scratch"},
        {"name": "rhs_tile", "space": "register", "dtype": "fp32", "shape": [128, 32], "mode": "scratch"},
        {"name": "u_tile", "space": "register", "dtype": "fp32", "shape": [128, 32], "mode": "scratch"},
    ]
    def access(operation, buffer):
        return {"operation": operation, "buffer": buffer,
                "indices": [{"source": "dimension", "dimension": 0},
                            {"source": "dimension", "dimension": 1}],
                "boundary": "mask_tiled_axes"}
    return {
        "schema_version": 2, "schedule_id": "forward-substitute-contract",
        "target": "sm_103a",
        "lowering": {"backend": "triton", "entry_point": "forward_substitute_contract"},
        "grid": [1, 1, 1], "metadata": {},
        "roles": [{"name": "compute", "execution_groups": [0, 1, 2, 3]}],
        "allocations": [], "buffers": buffers, "pipelines": [], "barriers": [],
        "operations": [
            {"id": "load_p", "kind": "load", "role": "compute", "reads": ["p"],
             "writes": ["p_tile"], "parameters": {"movement": "global"}},
            {"id": "load_rhs", "kind": "load", "role": "compute", "reads": ["rhs"],
             "writes": ["rhs_tile"], "parameters": {"movement": "global"}},
            {"id": "solve", "kind": "forward_substitute", "role": "compute",
             "reads": ["p_tile", "rhs_tile"], "writes": ["u_tile"],
             "parameters": {}, "depends_on": ["load_p", "load_rhs"]},
            {"id": "store", "kind": "store", "role": "compute", "reads": ["u_tile"],
             "writes": ["out"], "parameters": {"coalesced": False},
             "depends_on": ["solve"]},
        ],
        "outputs": ["out"],
        "access_maps": [access("load_p", "p"), access("load_rhs", "rhs"),
                        access("store", "out")],
    }


def target() -> Target:
    value = json.loads((ROOT / "compiler/targets/sm_103a.json").read_text())
    value["operation_kinds"].append("forward_substitute")
    return Target.from_dict(value)


class ForwardSubstituteIr(unittest.TestCase):
    def test_python_authoring_derives_one_fp32_result_tile(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="forward-substitute-python", target="sm_103a", backend="triton", entry_point="run", grid=(1,1,1))
def candidate(lm, p: cake.Tensor((32,32), "bf16"), rhs: cake.Tensor((128,32), "fp32"), out: cake.Tensor((128,32), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0,1,2,3])
    with compute:
        p_tile = lm.load(p[:,:])
        rhs_tile = lm.load(rhs[:,:])
        u = lm.forward_substitute(p_tile, rhs_tile)
        lm.store(out[:,:], u, coalesced=False)
'''
        parsed = parse(source).document
        schedule = Schedule.from_dict(parsed)
        self.assertEqual(schedule.buffer("u").shape, (128, 32))
        self.assertEqual(schedule.buffer("u").dtype.value, "fp32")
        self.assertEqual(schedule.operation("u").kind.value, "forward_substitute")

    def test_canonical_structure_and_analysis(self):
        source = document()
        Draft202012Validator(schedule_schema()).validate(source)
        schedule = Schedule.from_dict(source)
        self.assertIsInstance(schedule.operation("solve").parameters,
                              ForwardSubstituteParameters)
        blocking = [finding for finding in verify(schedule, target())
                    if finding.blocks_lowering]
        self.assertEqual(blocking, [])
        work = work_bound(schedule)
        self.assertIsNotNone(work)
        self.assertEqual(work.flops, 128 * 32 * 31)
        self.assertNotIn("solve", work.uncounted_arithmetic)
        profile = profile_envelope(schedule, target()).as_dict()
        self.assertIn(
            "forward_substitute has ordered token dependencies; no target latency calibration",
            profile["abstentions"],
        )
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        assessment = compiler.assess(source)
        self.assertIn("TARGET_OPERATION_UNSUPPORTED",
                      {finding.code for finding in assessment.findings})
        for backend in (native_cuda, triton):
            self.assertIn("BACKEND_OPERATION_UNEMITTABLE",
                          {finding.code for finding in backend.requirements(schedule)})

    def test_bad_edges_are_rejected_at_construction(self):
        source = document()
        for reads, writes in ((["p_tile"], ["u_tile"]),
                              (["p_tile", "rhs_tile", "out"], ["u_tile"]),
                              (["p_tile", "rhs_tile"], ["u_tile", "out"])):
            changed = copy.deepcopy(source)
            changed["operations"][2]["reads"] = reads
            changed["operations"][2]["writes"] = writes
            with self.subTest(reads=reads, writes=writes), self.assertRaisesRegex(
                ScheduleParseError, "forward_substitute requires exactly"
            ):
                Schedule.from_dict(changed)

    def test_bad_shapes_and_dtypes_are_rejected_at_construction(self):
        cases = (("p_tile", "shape", [32, 16], "P\\[C,C\\]"),
                 ("rhs_tile", "shape", [128, 16], "P\\[C,C\\]"),
                 ("u_tile", "shape", [64, 32], "P\\[C,C\\]"),
                 ("p_tile", "dtype", "fp32", "BF16 P"),
                 ("rhs_tile", "dtype", "bf16", "FP32 RHS/U"),
                 ("u_tile", "dtype", "bf16", "FP32 RHS/U"))
        for name, field, value, message in cases:
            changed = document()
            next(buffer for buffer in changed["buffers"]
                 if buffer["name"] == name)[field] = value
            with self.subTest(name=name, field=field), self.assertRaisesRegex(
                ScheduleParseError, message
            ):
                Schedule.from_dict(changed)

    def test_noncanonical_parameters_are_refused_by_schema_and_parser(self):
        changed = document()
        changed["operations"][2]["parameters"] = {"axis": 1}
        self.assertTrue(list(Draft202012Validator(schedule_schema()).iter_errors(changed)))
        with self.assertRaises(ScheduleParseError):
            Schedule.from_dict(changed)

    def test_verifier_remains_total_for_a_directly_built_invalid_schedule(self):
        valid = Schedule.from_dict(document())
        rhs = valid.buffer("rhs_tile")
        assert rhs is not None
        malformed = replace(valid, buffers=tuple(
            replace(buffer, dtype=DType.BF16) if buffer.name == rhs.name else buffer
            for buffer in valid.buffers
        ))
        self.assertIn("FORWARD_SUBSTITUTE_DTYPE",
                      {finding.code for finding in verify(malformed, target())})


if __name__ == "__main__":
    unittest.main()
