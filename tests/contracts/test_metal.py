"""Canonical Metal lowering, bounded refusals and portable generated-body correctness.

The native-C++ adapter runs the emitted body in 32 threads, synchronizing actual SIMD
intrinsic calls and checking collective participation. This is CPU semantic evidence,
not GPU, timing, physical-resource or MSL-toolchain qualification.
"""

from __future__ import annotations

import ctypes
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from open_cake_ir.compiler import Compiler, CompilerError
from open_cake_ir.compiler import frontend
from open_cake_ir.compiler.backends import metal
from open_cake_ir.compiler.performance.residency import residency_upper_bound
from open_cake_ir.compiler.core import _canonical_json_bytes
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.performance.ranking import cost
from open_cake_ir.compiler.target import Target, TargetParseError

ROOT = Path(__file__).resolve().parents[2]


def make_source(rows=3, width=37, operation="elementwise"):
    if operation == "elementwise":
        arguments = f'x: cake.Tensor(({rows},{width}), "fp32"), y: cake.Tensor(({rows},{width}), "fp32"), out: cake.Tensor(({rows},{width}), "fp32", mode="output")'
        body = '''a = lm.load(x[row,:], id="load_x")
        b = lm.load(y[row,:], id="load_y")
        result = (a + b) * 2.0
        lm.store(out[row,:], result, coalesced=False, id="store")'''
    else:
        arguments = f'x: cake.Tensor(({rows},{width}), "fp32"), out: cake.Tensor(({rows},), "fp32", mode="output")'
        body = f'''a = lm.load(x[row,:], id="load_x")
        result = lm.reduce(a, op="{operation}", axis=0, scope="cta", across_loop=False, id="reduce")
        lm.store(out[row], result, coalesced=False, id="store")'''
    return f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="metal-{operation}", target="apple_gpu_family8", backend="metal", entry_point="cake_metal")
def candidate(lm, {arguments}):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        {body}
'''


def make_document(rows=3, width=37, operation="elementwise"):
    """Unfrozen test input for the integration owner's separate corpus adoption."""
    return frontend.parse(make_source(rows, width, operation)).document


def make_rms_source(rows=2, width=65):
    return f"""from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="weighted-rms", target="apple_gpu_family8", backend="metal", entry_point="cake_rms")
def candidate(lm, x: cake.Tensor(({rows},{width}), "fp32"), weight: cake.Tensor(({width},), "fp32"), out: cake.Tensor(({rows},{width}), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row,:])
        weights = lm.load(weight[:])
        squares = lm.square(values)
        total = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False)
        inv = lm.rsqrt(total / {float(width)!r} + 1e-5)
        result = (values * inv) * weights
        lm.store(out[row,:], result, coalesced=False)
"""


def fp32(value):
    return ctypes.c_float(value).value


class _Program(ctypes.Structure):
    _fields_ = [(name, ctypes.c_uint) for name in "xyz"]


_CPU_SIMD = r"""
#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
using namespace std;
using uint = unsigned int;
using ushort = unsigned short;
using ulong = unsigned long;
struct uint3 { uint x,y,z; };
namespace precise {
    float rsqrt(float value) { return 1.0f / sqrt(value); }
    float exp(float value) { return ::expf(value); }
    float exp2(float value) { return ::exp2f(value); }
    float tanh(float value) { return ::tanhf(value); }
}
struct CpuGroup {
    float values[32];
    uint sources[32];
    int sites[32];
    atomic<bool> failed{false};
    mutex lock;
    condition_variable changed;
    uint arrivals = 0, generation = 0;
    bool sync() {
        unique_lock<mutex> guard(lock);
        if (failed) return false;
        uint before = generation;
        if (++arrivals == 32) {
            arrivals = 0; ++generation; changed.notify_all();
        } else if (!changed.wait_for(guard, chrono::seconds(3), [&] {
            return generation != before || failed;
        })) {
            failed = true; changed.notify_all();
        }
        return !failed;
    }
};
thread_local CpuGroup* cpu_group;
thread_local uint cpu_lane;
float cpu_collective(float value, uint source, int kind, int line) {
    auto& group = *cpu_group;
    group.values[cpu_lane] = value;
    group.sources[cpu_lane] = source;
    group.sites[cpu_lane] = line * 4 + kind;
    if (!group.sync()) return 0.0f;
    for (uint lane = 0; lane < 32; ++lane) {
        if (group.sites[lane] != line * 4 + kind || group.sources[lane] >= 32 ||
            (kind == 0 && group.sources[lane] != source)) group.failed = true;
    }
    float result = 0.0f;
    if (kind < 2) {
        if (source < 32) result = group.values[source];
    } else {
        float tree[32];
        copy(group.values, group.values + 32, tree);
        for (uint step = 1; step < 32; step *= 2)
            for (uint lane = 0; lane < 32; lane += 2 * step)
                tree[lane] = kind == 2 ? tree[lane] + tree[lane + step] : max(tree[lane], tree[lane + step]);
        result = tree[0];
    }
    group.sync(); // Do not overwrite values before all lanes finish reading them.
    return result;
}
#define simd_broadcast(data, source) cpu_collective(data, source, 0, __LINE__)
#define simd_shuffle(data, source) cpu_collective(data, source, 1, __LINE__)
#define simd_sum(data) cpu_collective(data, 0, 2, __LINE__)
#define simd_max(data) cpu_collective(data, 0, 3, __LINE__)
"""


class MetalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="cake-metal-contracts-")
        directory = Path(cls.temporary.name)
        # A draft uses the same exact-target loader as release. No release lock or
        # approval is created, and the owning target reference is computed only here.
        draft = json.loads((ROOT / "compiler/revision.json").read_text())
        document = json.loads((ROOT / "compiler/targets/apple_gpu_family8.json").read_text())
        draft["target_definitions"]["apple_gpu_family8"] = {
            "path": "compiler/targets/apple_gpu_family8.json",
            "canonical_sha256": hashlib.sha256(_canonical_json_bytes(document)).hexdigest(),
        }
        proposal = directory / "draft.json"
        proposal.write_text(json.dumps(draft))
        cls.compiler = Compiler.load(ROOT, proposal)
        cls.target = Target.from_dict(document)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def lower(self, document):
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        return assessment, self.compiler.lower(assessment)

    def execute_body(self, document, inputs):
        """Run the emitted SIMD body with a CPU intrinsic ABI; never dispatch Metal."""
        compiler = shutil.which("clang++") or shutil.which("c++")
        if compiler is None:
            self.skipTest("native C++ compiler required for generated-body CPU execution")
        _, lowering = self.lower(document)
        source = lowering.source.replace("#include <metal_stdlib>", _CPU_SIMD)
        source = source.replace("using namespace metal;", "").replace("#pragma METAL fp contract(off)", "")
        source = re.sub(r"\[\[[^]]*\]\]", "", source)
        source = source.replace("kernel void", 'extern "C" void').replace("device ", "")
        globals_ = [buffer for buffer in Schedule.from_dict(document).buffers if buffer.space.value == "global"]
        arguments = ", ".join(f"float* arg{index}" for index in range(len(globals_)))
        call_arguments = ", ".join(f"arg{index}" for index in range(len(globals_)))
        source += f"""
extern "C" int cpu_dispatch({arguments}, uint3 program) {{
    CpuGroup group;
    vector<thread> lanes;
    for (uint lane = 0; lane < 32; ++lane) lanes.emplace_back([&, lane] {{
        cpu_group = &group; cpu_lane = lane;
        {lowering.route.entry_point}({call_arguments}, program, lane);
    }});
    for (auto& lane : lanes) lane.join();
    return group.failed ? 1 : 0;
}}
"""
        with tempfile.TemporaryDirectory(prefix="cake-metal-body-") as temporary:
            path = Path(temporary)
            (path / "body.cpp").write_text(source)
            completed = subprocess.run([compiler, "-std=c++17", "-shared", "-fPIC", "-pthread", "-ffp-contract=off", "-fno-fast-math", str(path / "body.cpp"), "-o", str(path / "body.so")], capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            library = ctypes.CDLL(str(path / "body.so"))
            kernel = library.cpu_dispatch
            arrays = []
            for buffer in globals_:
                values = inputs.get(buffer.name, [float("nan")] * buffer.elements)
                self.assertEqual(len(values), buffer.elements)
                arrays.append((ctypes.c_float * buffer.elements)(*values))
            kernel.argtypes = [ctypes.POINTER(ctypes.c_float)] * len(arrays) + [_Program]
            kernel.restype = ctypes.c_int
            grid = lowering.toolchain_requirements["threadgroups_per_grid"]
            for position in itertools.product(*(range(extent) for extent in grid)):
                self.assertEqual(kernel(*arrays, _Program(*position)), 0, "SIMD collective participants or call sites diverged")
            return {buffer.name: list(array) for buffer, array in zip(globals_, arrays)}

    def test_m1_pro_exact_target_and_uncalibrated_analysis(self):
        target = Target.load(ROOT / "compiler/targets/apple_gpu_family7.json")
        self.assertEqual(target.device_names, ("Apple M1 Pro",))
        self.assertIsNone(target.compute_capability)
        for field, value in (("compute_capability", [10, 0]), ("occupancy", {}), ("peak", {})):
            document = json.loads((ROOT / "compiler/targets/apple_gpu_family7.json").read_text())
            with self.subTest(field=field), self.assertRaises(TargetParseError):
                Target.from_dict(dict(document, **{field: value}))
        for operation in ("elementwise", "sum", "max"):
            document = make_document(operation=operation)
            document["target"] = "apple_gpu_family7"
            assessment, lowering = self.lower(document)
            self.assertEqual(lowering.toolchain_requirements["target"], "apple_gpu_family7")
            self.assertEqual(lowering.toolchain_requirements["threads_per_threadgroup"], [32, 1, 1])
            self.assertFalse(assessment.calibration_available)
            self.assertEqual(self.compiler.rank([assessment]), ((), (assessment.schedule_id,)))
            self.assertIsNone(residency_upper_bound(Schedule.from_dict(document), target))
            self.assertIsNone(cost(Schedule.from_dict(document), target))
            profile = self.compiler.profile(assessment).as_dict()
            self.assertEqual(profile["ncu_metrics"], [])
            self.assertIsNone(profile["residency"])
            self.assertTrue(any("no calibrated performance model" in reason
                                for reason in profile["abstentions"]))
        wrong = make_document()
        self.assertIn("METAL_TARGET_UNSUPPORTED", [f.code for f in metal.preflight(Schedule.from_dict(wrong), target)])
        wrong["target"] = "apple_gpu_family7"
        from dataclasses import replace
        for changed in (replace(target, device_names=("Apple M1",)), replace(target, architecture="apple8")):
            self.assertIn("METAL_TARGET_UNSUPPORTED", [f.code for f in metal.preflight(Schedule.from_dict(wrong), changed)])

    def test_m1_pro_rmsnorm_formulas_execute_odd_width_on_cpu(self):
        from tools.metal import rmsnorm
        import struct
        for formula in rmsnorm.FORMULAS:
            with self.subTest(formula=formula):
                inputs, oracle = rmsnorm.inputs_and_oracle(2, 65, "mixed_magnitude")
                document = rmsnorm.document(2, 65, formula, target="apple_gpu_family7")
                unpacked = {key: list(struct.unpack(f"<{len(value)//4}f", value)) for key, value in inputs.items()}
                observed = self.execute_body(document, unpacked)["out"]
                for got, expected, tolerance in zip(observed, oracle["out"]["expected"], oracle["out"]["absolute_tolerance"]):
                    self.assertLessEqual(abs(got - expected), tolerance)

    def test_python_json_and_public_launch_abi_are_identical(self):
        document = make_document()
        with tempfile.TemporaryDirectory() as temporary:
            p = Path(temporary)
            (p / "candidate.py").write_text(make_source())
            (p / "candidate.json").write_text(json.dumps(document))
            self.assertEqual(self.compiler.assess_file(p / "candidate.py"), self.compiler.assess_file(p / "candidate.json"))
        assessment, lowering = self.lower(document)
        self.assertIn("METAL_SIMD_EXECUTION", [finding.code for finding in assessment.findings])
        self.assertEqual(lowering.toolchain_requirements, {
            "source_language": "metal", "compiler": "MTLDevice.makeLibrary", "target": "apple_gpu_family8",
            "buffer_order": ["x", "y", "out"], "threadgroups_per_grid": [3, 1, 1],
            "threads_per_threadgroup": [32, 1, 1], "threadgroup_memory_bytes": 0,
            "language_standard": "metal2.3", "fast_math_enabled": False,
            "execution_model": "simd_program_tile", "active_threads_per_threadgroup": 32,
        })
        self.assertIn("#pragma METAL fp contract(off)", lowering.source)
        self.assertEqual(set(lowering.source_map), {operation["id"] for operation in document["operations"]})

    def test_odd_elementwise_shapes_and_distributions_execute(self):
        for rows, width in ((1, 1), (3, 37), (2, 65)):
            with self.subTest(shape=(rows, width)):
                x = [fp32((index % 17 - 8) / 8) for index in range(rows * width)]
                y = [fp32((index % 7 - 3) / 16) for index in range(rows * width)]
                observed = self.execute_body(make_document(rows, width), {"x": x, "y": y})
                self.assertEqual(observed["out"], [fp32(fp32(a + b) * 2) for a, b in zip(x, y)])

    def test_every_admitted_unary_primitive_executes_or_names_why_it_cannot(self):
        """The emitter's unary map and this CPU shim must not drift apart.

        `exp2` was admitted into the vocabulary and mapped to `precise::exp2` while the
        shim still had three names, so no portable semantic check could reach it. `tanh`
        was mapped and unreachable for longer: the IR requires it to name an instruction
        contract and no Apple Target admitted one, so it names the Metal contract here.
        """
        from open_cake_ir.compiler.backends.metal import _UNARY
        from open_cake_ir.compiler.ir.vocabulary import ElementwiseOp

        expected = {ElementwiseOp.SQUARE: lambda v: fp32(v * v),
                    ElementwiseOp.RELU: lambda v: max(v, 0.0),
                    # The shim rounds the square root before the divide, as the emitted float does.
                    ElementwiseOp.RSQRT: lambda v: fp32(1.0 / fp32(math.sqrt(v))),
                    ElementwiseOp.EXP: lambda v: fp32(math.exp(v)),
                    ElementwiseOp.EXP2: lambda v: fp32(2.0 ** v),
                    ElementwiseOp.RECIPROCAL: lambda v: fp32(1.0 / v),
                    ElementwiseOp.TANH: lambda v: fp32(math.tanh(v))}
        self.assertEqual(set(_UNARY), set(expected))
        values = [0.5, 1.0, 2.0, 3.25, 0.125, 4.0, 1.5, 2.75]
        for op, reference in expected.items():
            source = f'''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="unary-{op.value}", target="apple_gpu_family8", backend="metal", entry_point="cake_unary")
def candidate(lm, x: cake.Tensor((2, 4), "fp32"), out: cake.Tensor((2, 4), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row, :], id="load_x")
        result = lm.{op.value}(values, id="unary")
        lm.store(out[row, :], result, coalesced=False, id="store_out")
'''
            with self.subTest(op=op.value):
                if op is ElementwiseOp.TANH:
                    with self.assertRaisesRegex(frontend.FrontendError,
                                                "instruction is required for tanh"):
                        frontend.parse(source)
                    source = source.replace(
                        "lm.tanh(values,",
                        f'lm.tanh(values, instruction={{"contract": "{metal._METAL_TANH_CONTRACT}"}},')
                document = frontend.parse(source).document
                self.assertEqual(self.execute_body(document, {"x": values})["out"],
                                 [reference(value) for value in values])

    def test_sum_and_negative_max_use_scalar_buffer_convention(self):
        for operation in ("sum", "max"):
            with self.subTest(operation=operation):
                document = make_document(3, 65, operation)
                x = [fp32(-(index % 19 + 1) / 8) for index in range(195)]
                expected = []
                for row in range(3):
                    values = x[row * 65:(row + 1) * 65]
                    accumulator = 0.0
                    for value in values:
                        accumulator = fp32(accumulator + value)
                    expected.append(accumulator if operation == "sum" else max(values))
                self.assertEqual(self.execute_body(document, {"x": x})["out"], expected)
                result = next(buffer for buffer in document["buffers"] if buffer["name"] == "result")
                self.assertEqual(result["shape"], [1])

    def test_composed_reduction_broadcast_and_arithmetic(self):
        source = '''from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="composition", target="apple_gpu_family8", backend="metal", entry_point="cake_composition")
def candidate(lm, x: cake.Tensor((2,3,5), "fp32"), out: cake.Tensor((2,3,5), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    batch = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[batch,:,:])
        maximum = lm.reduce(values, op="max", axis=1, scope="cta", across_loop=False)
        shifted = lm.sub(values, lm.broadcast(maximum, axis=0))
        square = lm.square(shifted)
        result = lm.relu(square) / 2.0
        lm.store(out[batch,:,:], result, coalesced=False)
'''
        x = [fp32((index % 11 - 5) / 4) for index in range(30)]
        expected = []
        for row in range(6):
            values = x[row * 5:(row + 1) * 5]
            expected.extend(fp32(fp32(fp32(v - max(values)) ** 2) / 2.0) for v in values)
        self.assertEqual(self.execute_body(frontend.parse(source).document, {"x": x})["out"], expected)

    def test_first_axis_reduction_and_two_program_axes(self):
        source = """from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="first-axis", target="apple_gpu_family8", backend="metal", entry_point="cake_axes")
def candidate(lm, x: cake.Tensor((2,3,5), "fp32"), out: cake.Tensor((2,5), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    batch = lm.program(x, axis=1, dimension=0, tile=1)
    with compute:
        values = lm.load(x[batch,:,:])
        result = lm.reduce(values, op="sum", axis=0, scope="cta", across_loop=False)
        lm.store(out[batch,:], result, coalesced=False)
"""
        x = [float(index - 15) for index in range(30)]
        expected = [sum(x[batch * 15 + row * 5 + col] for row in range(3))
                    for batch in range(2) for col in range(5)]
        self.assertEqual(self.execute_body(frontend.parse(source).document, {"x": x})["out"], expected)
        source = source.replace('out: cake.Tensor((2,5)', 'out: cake.Tensor((2,3)')
        source = source.replace('    with compute:', '    row = lm.program(x, axis=0, dimension=1, tile=1)\n    with compute:')
        source = source.replace('x[batch,:,:]', 'x[batch,row,:]').replace('out[batch,:]', 'out[batch,row]')
        expected = [sum(x[row * 5:(row + 1) * 5]) for row in range(6)]
        self.assertEqual(self.execute_body(frontend.parse(source).document, {"x": x})["out"], expected)

    def test_weighted_rms_and_scalar_broadcast_cover_lane_boundaries(self):
        for width in (1, 7, 32, 65, 257, 1024, 4096):
            with self.subTest(width=width):
                # One epsilon-dominated zero row and one mixed-sign normal row.
                x = [0.0] * width + [fp32((index % 23 - 11) / 7.0) for index in range(width)]
                weight = [fp32((index % 9 - 4) / 3.0) for index in range(width)]
                document = frontend.parse(make_rms_source(width=width)).document
                observed = self.execute_body(document, {"x": x, "weight": weight})["out"]
                expected = []
                for row in range(2):
                    values = x[row * width:(row + 1) * width]
                    inv = 1.0 / math.sqrt(math.fsum(value * value for value in values) / width + 1e-5)
                    expected.extend(value * inv * scale for value, scale in zip(values, weight))
                for actual, reference in zip(observed, expected):
                    self.assertLessEqual(abs(actual - reference), 2e-5 + 2e-5 * abs(reference))
                _, lowering = self.lower(document)
                self.assertIn("precise::rsqrt(", lowering.source)
                self.assertIn("simd_sum(", lowering.source)
                if width > 1:
                    self.assertIn("simd_broadcast(", lowering.source)
                self.assertNotIn("if (lane != 0u) return", lowering.source)

    def test_multislot_broadcast_and_each_reduction_axis_are_collective(self):
        shape = (3, 33, 5)
        for axis in range(3):
            extent = shape[axis]
            output_shape = tuple(value for index, value in enumerate(shape) if index != axis)
            source = f"""from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="broadcast-and-reduce", target="apple_gpu_family8", backend="metal", entry_point="cake_compose")
def candidate(lm, x: cake.Tensor((1,3,33,5), "fp32"), weight: cake.Tensor(({extent},), "fp32"), out: cake.Tensor((1,3,33,5), "fp32", mode="output"), reduced: cake.Tensor((1,{output_shape[0]},{output_shape[1]}), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    batch = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[batch,:,:,:])
        weights = lm.load(weight[:])
        result = lm.sub(values, lm.broadcast(weights, axis={axis}))
        total = lm.reduce(result, op="sum", axis={axis}, scope="cta", across_loop=False)
        lm.store(out[batch,:,:,:], result, coalesced=False)
        lm.store(reduced[batch,:,:], total, coalesced=False)
"""
            x = [float(index % 41 - 20) for index in range(math.prod(shape))]
            weight = [float(index - 17) for index in range(extent)]
            inner = math.prod(shape[axis + 1:])
            expected = [value - weight[(index // inner) % extent] for index, value in enumerate(x)]
            reduced = [sum(expected[(output // inner) * extent * inner + j * inner + output % inner]
                           for j in range(extent)) for output in range(math.prod(output_shape))]
            with self.subTest(axis=axis):
                observed = self.execute_body(frontend.parse(source).document, {"x": x, "weight": weight})
                self.assertEqual(observed["out"], expected)
                self.assertEqual(observed["reduced"], reduced)

    def test_scalar_first_and_second_sub_div_preserve_operand_order(self):
        source = """from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="ordered-scalars", target="apple_gpu_family8", backend="metal", entry_point="cake_order")
def candidate(lm, x: cake.Tensor((2,7), "fp32"), scalar: cake.Tensor((1,), "fp32"), a: cake.Tensor((2,7), "fp32", mode="output"), b: cake.Tensor((2,7), "fp32", mode="output"), c: cake.Tensor((2,7), "fp32", mode="output"), d: cake.Tensor((2,7), "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row,:])
        scale = lm.load(scalar[:])
        first = scale - values
        second = values - scale
        third = scale / values
        fourth = values / scale
        lm.store(a[row,:], first, coalesced=False)
        lm.store(b[row,:], second, coalesced=False)
        lm.store(c[row,:], third, coalesced=False)
        lm.store(d[row,:], fourth, coalesced=False)
"""
        x = [float(index + 1) for index in range(14)]
        observed = self.execute_body(frontend.parse(source).document, {"x": x, "scalar": [2.0]})
        for name, expected in (("a", [2.0 - value for value in x]), ("b", [value - 2.0 for value in x]),
                               ("c", [fp32(2.0 / value) for value in x]), ("d", [value / 2.0 for value in x])):
            self.assertEqual(observed[name], expected)

    def test_private_storage_is_lane_distributed_and_lifetime_bounded(self):
        document = make_document(rows=1, width=4096)
        _, lowering = self.lower(document)
        self.assertIn("float v3[128]", lowering.source)
        self.assertNotIn("float v3[4096]", lowering.source)
        # Three arrays overlap at the add, although four are declared over the DAG.
        self.assertEqual(metal.private_values_per_thread(Schedule.from_dict(document)), 384)
        excessive = self.compiler.assess(make_document(rows=1, width=16384))
        self.assertFalse(excessive.lowering_eligible)
        self.assertIn("METAL_PRIVATE_STORAGE_LIMIT", [f.code for f in excessive.findings])

    def test_slice_coordinates_are_not_operator_or_shape_recognition(self):
        source = make_source(2, 11).replace('name="metal-elementwise"', 'name="a_different_task"')
        source = source.replace('out: cake.Tensor((2,11)', 'out: cake.Tensor((2,7)')
        source = source.replace('x[row,:]', 'x[row,2:9]').replace('y[row,:]', 'y[row,2:9]')
        x, y = list(range(22)), list(range(100, 122))
        observed = self.execute_body(frontend.parse(source).document, {"x": x, "y": y})
        expected = [float((x[row * 11 + col] + y[row * 11 + col]) * 2) for row in range(2) for col in range(2, 9)]
        self.assertEqual(observed["out"], expected)

    def test_consecutive_simd_groups_widen_the_stripe_and_own_their_barriers(self):
        """One group is unchanged; more groups finish every collective in threadgroup memory."""
        single = frontend.parse(make_rms_source(rows=2, width=1024)).document
        _, base = self.lower(single)
        self.assertEqual(base.toolchain_requirements["threads_per_threadgroup"], [32, 1, 1])
        self.assertEqual(base.toolchain_requirements["threadgroup_memory_bytes"], 0)
        self.assertNotIn("threadgroup_barrier", base.source)
        widths = {}
        for groups in (2, 4, 8):
            document = json.loads(json.dumps(single))
            document["roles"][0]["warps"] = list(range(groups))
            with self.subTest(groups=groups):
                _, lowering = self.lower(document)
                threads = 32 * groups
                self.assertEqual(lowering.toolchain_requirements["threads_per_threadgroup"], [threads, 1, 1])
                self.assertEqual(lowering.toolchain_requirements["active_threads_per_threadgroup"], threads)
                # Only the groups need a shared slot each; the scalar reuses slot zero.
                self.assertEqual(lowering.toolchain_requirements["threadgroup_memory_bytes"], groups * 4)
                self.assertIn(f"threadgroup float share[{groups}];", lowering.source)
                self.assertIn("uint lane = thread_position.x;", lowering.source)
                self.assertIn("float grouped = simd_sum(partial);", lowering.source)
                self.assertNotIn("simd_broadcast", lowering.source)
                self.assertEqual(lowering.source.count("threadgroup_barrier(mem_flags::mem_threadgroup);"), 4)
                self.assertIn(f"uint i = s * {threads}u + lane;", lowering.source)
                widths[groups] = metal.private_values_per_thread(
                    Schedule.from_dict(document), metal.lane_width(Schedule.from_dict(document)))
        # A fixed 1024-wide reduction owns fewer values per lane as the stripe widens.
        # Scalars keep their single slot at every width, so the fall is not proportional.
        self.assertEqual(widths, {2: 49, 4: 25, 8: 13})
        self.assertEqual(metal.private_values_per_thread(Schedule.from_dict(single)), 97)

    def test_multi_group_refuses_the_exchange_it_does_not_emit(self):
        document = make_document(rows=4, width=64, operation="elementwise")
        # A narrower non-scalar operand needs a cross-group exchange; one group shuffles.
        document["buffers"][1]["shape"] = [64]
        for access in document["access_maps"]:
            if access["buffer"] == "y":
                access["indices"] = [{"dimension": 1, "source": "dimension"}]
        for operation in document["operations"]:
            if operation["kind"] == "elementwise" and set(operation["reads"]) == {"a", "b"}:
                operation["parameters"]["broadcast_axis"] = 1
        single = self.compiler.assess(json.loads(json.dumps(document)))
        document["roles"][0]["warps"] = [0, 1]
        widened = self.compiler.assess(document)
        codes = {finding.code for finding in widened.findings if finding.blocks_lowering}
        if single.lowering_eligible:
            self.assertIn("METAL_BROADCAST_WIDTH_UNSUPPORTED", codes)
        else:
            self.skipTest("this fixture does not reach the single-group exchange path")

    def test_localized_backend_refusals_never_reach_emission(self):
        mutations = [
            (lambda d: d["operations"][-1]["parameters"].update(coalesced=True), "METAL_COALESCING_UNSUPPORTED", "operations[4].parameters.coalesced"),
            (lambda d: d["operations"][0]["parameters"].update(reuse="streamed"), "METAL_LOAD_UNSUPPORTED", "operations[0].parameters"),
            (lambda d: d["roles"][0].update(warps=[1]), "METAL_ROLE_UNSUPPORTED", "roles"),
            (lambda d: d.update(residency={"registers_per_thread": 64}), "METAL_RESIDENCY_UNSUPPORTED", "residency"),
            (lambda d: d["operations"][3]["parameters"].update(scalar=1e100), "METAL_SCALAR_RANGE_UNSUPPORTED", "operations[3].parameters.scalar"),
            (lambda d: d["operations"][3].update(id="bad\nmarker"), "METAL_OPERATION_ID_UNSUPPORTED", None),
        ]
        for mutation, code, path in mutations:
            document = make_document()
            mutation(document)
            # Keep dependencies canonical when intentionally changing an operation id.
            if code == "METAL_OPERATION_ID_UNSUPPORTED":
                document["operations"][-1]["depends_on"] = ["bad\nmarker"]
            with self.subTest(code=code):
                result = self.compiler.assess(document)
                self.assertFalse(result.lowering_eligible)
                matching = [finding for finding in result.findings if finding.code == code]
                self.assertTrue(matching, result.findings)
                if path is not None:
                    self.assertEqual(matching[0].path, path)
                with self.assertRaises(CompilerError):
                    self.compiler.lower(result)
                with self.assertRaises(metal.EmitError):
                    metal.emit(Schedule.from_dict(document), self.target)

    def test_missing_access_maps_are_refused_at_the_public_boundary(self):
        for absent in (False, True):
            document = make_document()
            if absent:
                document.pop("access_maps")
            else:
                document["access_maps"] = []
            with self.subTest(absent=absent):
                result = self.compiler.assess(document)
                self.assertTrue(result.accepted)
                self.assertFalse(result.lowering_eligible)
                missing = [f.path for f in result.findings if f.code == "METAL_ACCESS_MAP_REQUIRED"]
                self.assertEqual(missing, ["operations[0].reads[0]", "operations[1].reads[0]", "operations[4].writes[0]"])
                with self.assertRaisesRegex(CompilerError, "METAL_ACCESS_MAP_REQUIRED"):
                    self.compiler.lower(result)
                with self.assertRaisesRegex(metal.EmitError, "exactly one access map"):
                    metal.emit(Schedule.from_dict(document), self.target)

    def test_store_address_domain_cannot_discard_or_invent_private_axes(self):
        for component_index, component in (
            (1, {"source": "program", "name": "row"}),
            (0, {"source": "dimension", "dimension": 0}),
        ):
            document = make_document()
            document["access_maps"][-1]["indices"][component_index] = component
            with self.subTest(component=component):
                result = self.compiler.assess(document)
                self.assertFalse(result.lowering_eligible)
                findings = [f for f in result.findings if f.blocks_lowering]
                self.assertEqual([(f.code, f.path) for f in findings],
                                 [("STORE_ACCESS_SHAPE_MISMATCH", "operations[4].reads[0]")])
                self.assertFalse(result.accepted)
                with self.assertRaisesRegex(CompilerError, "STORE_ACCESS_SHAPE_MISMATCH"):
                    self.compiler.lower(result)
                # The shared value-shape gate now fires before backend preflight.
                # Independently preserve Metal's direct capability refusal too.
                backend = metal.preflight(Schedule.from_dict(document), self.target)
                self.assertEqual([f.path for f in backend if f.code == "METAL_ACCESS_VALUE_SHAPE"],
                                 ["access_maps[2].indices"])
                with self.assertRaisesRegex(metal.EmitError, r"operations\[4\].reads\[0\]: store address requires register shape"):
                    metal.emit(Schedule.from_dict(document), self.target)

    def test_store_owns_every_varying_program_axis(self):
        document = make_document()
        document["program_map"]["axes"].append(
            {"name": "other", "axis": 1, "buffer": "y", "dimension": 0, "tile": 1})
        document["access_maps"][1]["indices"][0]["name"] = "other"
        result = self.compiler.assess(document)
        self.assertFalse(result.lowering_eligible)
        ownership = [f for f in result.findings if f.code == "METAL_STORE_OWNERSHIP"]
        self.assertEqual([f.path for f in ownership], ["access_maps[2].indices"])
        self.assertIn("other", ownership[0].message)
        with self.assertRaisesRegex(CompilerError, "METAL_STORE_OWNERSHIP"):
            self.compiler.lower(result)
        with self.assertRaisesRegex(metal.EmitError, "does not own varying program axes"):
            metal.emit(Schedule.from_dict(document), self.target)

        # An omitted extent-one coordinate creates no second threadgroup writer.
        next(buffer for buffer in document["buffers"] if buffer["name"] == "y")["shape"][0] = 1
        x, y = list(range(111)), list(range(37))
        actual = self.execute_body(document, {"x": x, "y": y})["out"]
        self.assertEqual(actual, [float((value + y[index % 37]) * 2) for index, value in enumerate(x)])

    def test_existing_single_writer_rule_covers_cross_operation_ownership(self):
        document = make_document()
        second = dict(document["operations"][-1], id="second_store", depends_on=["store"])
        document["operations"].append(second)
        document["access_maps"].append(dict(document["access_maps"][-1], operation="second_store"))
        result = self.compiler.assess(document)
        self.assertFalse(result.accepted)
        self.assertFalse(result.lowering_eligible)
        self.assertIn("BUFFER_MULTIPLE_WRITERS", [f.code for f in result.findings])
        with self.assertRaises(CompilerError):
            self.compiler.lower(result)
        with self.assertRaises(metal.EmitError):
            metal.emit(Schedule.from_dict(document), self.target)

    def test_identifiers_preserve_source_map_and_entry_point_identity(self):
        separators = ("\n", "\r", "\r\n", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029")
        for token in (*separators, "__SCHEDULE_SHA256__"):
            document = make_document()
            new_id = "store" + token + "tail"
            document["operations"][-1]["id"] = new_id
            document["access_maps"][-1]["operation"] = new_id
            with self.subTest(token=repr(token)):
                result = self.compiler.assess(document)
                self.assertFalse(result.lowering_eligible)
                self.assertIn("METAL_OPERATION_ID_UNSUPPORTED", [f.code for f in result.findings])
                with self.assertRaises(CompilerError):
                    self.compiler.lower(result)
                with self.assertRaises(metal.EmitError):
                    metal.emit(Schedule.from_dict(document), self.target)
        document = make_document()
        document["lowering"]["entry_point"] = "kernel__SCHEDULE_SHA256__tail"
        result = self.compiler.assess(document)
        self.assertFalse(result.lowering_eligible)
        self.assertIn("METAL_ENTRY_POINT_UNSUPPORTED", [f.code for f in result.findings])
        with self.assertRaises(CompilerError):
            self.compiler.lower(result)
        with self.assertRaises(metal.EmitError):
            metal.emit(Schedule.from_dict(document), self.target)

        # Ordinary non-ASCII comment ids remain exact; source projection is the oracle.
        document = make_document()
        document["operations"][-1]["id"] = "store_结果"
        document["access_maps"][-1]["operation"] = "store_结果"
        _, lowering = self.lower(document)
        self.assertEqual(set(lowering.source_map), {op["id"] for op in document["operations"]})
        self.assertIn("kernel void " + lowering.route.entry_point + "(", lowering.source)

    def test_source_map_ids_cannot_continue_the_generated_comment_line(self):
        for suffix in ("\\", "??/"):
            document = make_document()
            operation = document["operations"][-1]
            operation["id"] += suffix
            document["access_maps"][-1]["operation"] = operation["id"]
            with self.subTest(suffix=suffix):
                result = self.compiler.assess(document)
                self.assertFalse(result.lowering_eligible)
                findings = [f for f in result.findings if f.code == "METAL_OPERATION_ID_UNSUPPORTED"]
                self.assertEqual([f.path for f in findings], ["operations[4].id"])
                with self.assertRaisesRegex(CompilerError, "METAL_OPERATION_ID_UNSUPPORTED"):
                    self.compiler.lower(result)
                with self.assertRaisesRegex(metal.EmitError, "lexical line continuation"):
                    metal.emit(Schedule.from_dict(document), self.target)

    def test_unsupported_tile_dtype_and_instruction_do_not_fall_back(self):
        tiled = frontend.parse(make_source().replace('tile=1', 'tile=2')).document
        self.assertIn("METAL_PROGRAM_TILE_UNSUPPORTED", [f.code for f in self.compiler.assess(tiled).findings])
        for source in (make_source().replace('"fp32"', '"bf16"'),
                       make_source().replace('(a + b) * 2.0', 'lm.fma(a,b,a)'),
                       make_source().replace('target="apple_gpu_family8"', 'target="sm_100a"')):
            result = self.compiler.assess(frontend.parse(source).document)
            self.assertFalse(result.lowering_eligible)
            self.assertTrue(any(f.blocks_lowering for f in result.findings))

    def test_cuda_backend_refuses_apple_target_before_source_emission(self):
        document = make_document(width=32)
        document["lowering"]["backend"] = "triton"
        assessment = self.compiler.assess(document)
        self.assertFalse(assessment.lowering_eligible)
        self.assertIn("BACKEND_TARGET_UNSUPPORTED", [f.code for f in assessment.findings])
        with self.assertRaises(CompilerError):
            self.compiler.lower(assessment)

    def test_target_and_profile_do_not_inherit_cuda_resources_or_calibration(self):
        assessment, _ = self.lower(make_document())
        self.assertIsNone(self.target.compute_capability)
        self.assertIsNone(self.target.warps_per_warpgroup)
        self.assertEqual(self.target.resource_limits.maximum_tensor_memory_bytes, 0)
        self.assertIsNone(self.target.occupancy)
        self.assertIsNone(self.target.peak)
        self.assertIsNone(residency_upper_bound(Schedule.from_dict(make_document()), self.target))
        self.assertIsNone(cost(Schedule.from_dict(make_document()), self.target))
        self.assertFalse(assessment.calibration_available)
        self.assertEqual(self.compiler.rank([assessment]), ((), (assessment.schedule_id,)))
        profile = self.compiler.profile(assessment).as_dict()
        self.assertEqual(profile["ncu_metrics"], [])
        self.assertIsNone(profile["residency"])
        self.assertIn("no calibrated performance model", profile["abstentions"][0])
        document = json.loads((ROOT / "compiler/targets/apple_gpu_family8.json").read_text())
        for field, value in (("compute_capability", [10, 0]), ("occupancy", {}), ("peak", {})):
            with self.subTest(field=field), self.assertRaises(TargetParseError):
                Target.from_dict(dict(document, **{field: value}))

    def test_scalar_reduction_result_still_refuses_wrong_extent(self):
        document = make_document(operation="sum")
        next(buffer for buffer in document["buffers"] if buffer["name"] == "result")["shape"] = [2]
        self.assertIn("REDUCE_SHAPE_MISMATCH", [f.code for f in self.compiler.assess(document).findings])


if __name__ == "__main__":
    unittest.main()
