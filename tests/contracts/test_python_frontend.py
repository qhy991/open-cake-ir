"""The Python authoring boundary preserves canonical semantics and never executes text."""

from __future__ import annotations

import ast
import importlib.util
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path

from open_cake_ir.cli import main
from open_cake_ir.compiler import Compiler, CompilerError, emit_triton
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from open_cake_ir.compiler.frontend import FrontendError, ScheduleSource, parse, read_schedule


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/python"
DRAFT = ROOT / "compiler/revision.json"
FMA = (EXAMPLES / "fma.py").read_text()


class PythonFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = Compiler.load(ROOT, DRAFT)

    def test_examples_preserve_canonical_documents_assessments_and_generated_source(self):
        for example, reference in (
            ("fma", "fma-b8-smoke"), ("softmax", "softmax-b8-smoke"),
            ("kmeans_pipeline", "flash-kmeans-assignment-full"),
        ):
            with self.subTest(example=example):
                source = read_schedule(EXAMPLES / f"{example}.py")
                expected = json.loads((ROOT / f"corpus/schedules/{reference}.json").read_text())
                # A new authoring example does not inherit legacy provenance or identity.
                expected["schedule_id"] = source.document["schedule_id"]
                expected["metadata"] = {}
                self.assertEqual(source.document, expected)
                original = self.compiler.assess(expected)
                observed = self.compiler.assess_file(EXAMPLES / f"{example}.py")
                self.assertEqual(observed, original)
                self.assertTrue(observed.accepted)
                self.assertTrue(observed.lowering_eligible)
                self.assertEqual(self.compiler.lower(observed), self.compiler.lower(original))
                ast.parse(self.compiler.lower(observed).source)

    def test_nested_role_pipeline_loops_are_symbolic_and_keep_handoffs(self):
        doc = read_schedule(EXAMPLES / "kmeans_pipeline.py").document
        self.assertEqual(len(doc["operations"]), 6)
        self.assertEqual([loop["body"] for loop in doc["tile_loops"]], [
            ["k_loop", "distance_epilogue"], ["load_tokens", "load_centroids", "dot_mma"],
        ])
        mma = next(op for op in doc["operations"] if op["id"] == "dot_mma")
        self.assertEqual((mma["role"], mma["waits"], mma["signals"], mma["pipeline"]),
                         ("mma", ["tiles_ready"], ["accumulator_ready"], "main"))
        larger = (EXAMPLES / "kmeans_pipeline.py").read_text().replace("(1024, 128)", "(2048, 128)")
        self.assertEqual(parse(larger).document["operations"], doc["operations"])

    def test_multiplication_and_addition_do_not_become_fma(self):
        separate = parse(FMA.replace('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'a_tile * b_tile + c_tile'))
        arithmetic = [op for op in separate.document["operations"] if op["kind"] == "elementwise"]
        self.assertEqual([op["parameters"]["op"] for op in arithmetic], ["mul", "add"])
        self.assertIn(arithmetic[0]["id"], arithmetic[1]["depends_on"])
        self.assertTrue(all("instruction" not in op["parameters"] for op in arithmetic))
        fused = parse(FMA).document["operations"][3]
        self.assertEqual(fused["parameters"]["instruction"], {"contract": "ptx.fma.rn.f32"})
        self.assertTrue(self.compiler.assess(separate.document).lowering_eligible)

    def test_literal_binary_operand_uses_existing_scalar_contract(self):
        source = parse(FMA.replace('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'a_tile * 2.0'))
        operation = source.document["operations"][3]
        self.assertEqual(operation["parameters"], {"op": "mul", "scalar": 2.0})
        self.assertEqual(operation["reads"], ["a_tile"])
        self.assertTrue(self.compiler.assess(source.document).lowering_eligible)

    def scalar_source(self, expression="values * scale", *, scalar_tensor=False, scalar_output=False):
        scalar_shape = "(1,1)" if scalar_tensor else "(1,)"
        scalar_access = "scalar[:,:]" if scalar_tensor else "scalar[:]"
        output_shape = "(2,)" if scalar_output else "(2,32)"
        output_access = "out[row]" if scalar_output else "out[row,:]"
        return f"""from open_cake_ir.compiler import frontend as cake
@cake.schedule(name="scalar-arithmetic", target="sm_100a", backend="triton", entry_point="cake_scalar")
def candidate(lm, x: cake.Tensor((2,32), "fp32"), scalar: cake.Tensor({scalar_shape}, "fp32"), out: cake.Tensor({output_shape}, "fp32", mode="output")):
    compute = lm.role(warps=[0])
    row = lm.program(x, axis=0, dimension=0, tile=1)
    with compute:
        values = lm.load(x[row,:])
        scale = lm.load({scalar_access})
        result = {expression}
        lm.store({output_access}, result)
"""

    def test_canonical_scalar_broadcast_infers_shape_without_reordering_operands(self):
        for expression, reads, operation in (
            ("scale - values", ["scale", "values"], "sub"),
            ("values - scale", ["values", "scale"], "sub"),
            ("scale / values", ["scale", "values"], "div"),
            ("values / scale", ["values", "scale"], "div"),
        ):
            with self.subTest(expression=expression):
                document = parse(self.scalar_source(expression)).document
                self.assertEqual(next(b for b in document["buffers"] if b["name"] == "result")["shape"], [32])
                op = next(op for op in document["operations"] if op["id"] == "result")
                self.assertEqual(op["reads"], reads)
                self.assertEqual(op["parameters"], {"op": operation})
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowered = self.compiler.lower(assessment)
                self.assertIn("scale = tl.load(", lowered.source)
                direct = emit_triton.emit(Schedule.from_dict(document), Target.load(ROOT / "compiler/targets/sm_100a.json"))
                self.assertIn("scale = tl.load(", direct.source)

    def test_triton_refuses_direct_global_scalar_values_with_complete_access_maps(self):
        for expression, position in (("values / scalar[unit]", 1), ("scalar[unit] / values", 0)):
            source = self.scalar_source(expression)
            source = source.replace("    with compute:", "    unit = lm.program(scalar, axis=1, dimension=0, tile=1)\n    with compute:")
            source = source.replace("        scale = lm.load(scalar[:])\n", "")
            document = parse(source).document
            with self.subTest(expression=expression):
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.accepted)
                self.assertFalse(assessment.lowering_eligible)
                refusal = [f for f in assessment.findings if f.code == "TRITON_ELEMENTWISE_STORAGE"]
                self.assertEqual([f.path for f in refusal], [f"operations[1].reads[{position}]"])
                with self.assertRaisesRegex(CompilerError, "TRITON_ELEMENTWISE_STORAGE"):
                    self.compiler.lower(assessment)
                with self.assertRaisesRegex(emit_triton.EmitError, "requires register values"):
                    emit_triton.emit(Schedule.from_dict(document), Target.load(ROOT / "compiler/targets/sm_100a.json"))

    def test_triton_refuses_global_arithmetic_destinations(self):
        source = self.scalar_source()
        source = source.replace('out: cake.Tensor((2,32), "fp32", mode="output")',
                                'out: cake.Tensor((2,32), "fp32", mode="output"), direct: cake.Tensor((32,), "fp32", mode="output")')
        source = source.replace("        result = values * scale", '        lm.mul(values, scale, out=direct[:], id="direct_arithmetic")\n        result = values * scale')
        document = parse(source).document
        assessment = self.compiler.assess(document)
        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        refusal = [f for f in assessment.findings if f.code == "TRITON_ELEMENTWISE_STORAGE"]
        self.assertEqual([f.path for f in refusal], ["operations[2].writes[0]"])
        with self.assertRaisesRegex(CompilerError, "TRITON_ELEMENTWISE_STORAGE"):
            self.compiler.lower(assessment)
        with self.assertRaisesRegex(emit_triton.EmitError, "requires register values"):
            emit_triton.emit(Schedule.from_dict(document), Target.load(ROOT / "compiler/targets/sm_100a.json"))

    def test_two_scalars_cannot_invent_a_larger_result(self):
        document = parse(self.scalar_source("scale + scale", scalar_output=True)).document
        result = next(b for b in document["buffers"] if b["name"] == "result")
        self.assertEqual(result["shape"], [1])
        self.assertTrue(self.compiler.assess(document).lowering_eligible)
        result["shape"] = [32]
        self.assertIn("ELEMENTWISE_SHAPE_MISMATCH", [f.code for f in self.compiler.assess(document).findings])

    def test_scalar_rule_does_not_erase_tensor_rank_axis_or_fma_contract(self):
        tensor = parse(self.scalar_source(scalar_tensor=True)).document
        self.assertFalse(self.compiler.assess(tensor).accepted)
        self.assertEqual(next(b for b in tensor["buffers"] if b["name"] == "scale")["shape"], [1, 1])
        for axis in (0, 9):
            document = parse(self.scalar_source(f"lm.mul(values, lm.broadcast(scale, axis={axis}))")).document
            self.assertIn("ELEMENTWISE_BROADCAST", [f.code for f in self.compiler.assess(document).findings])
        fma = parse(self.scalar_source("lm.fma(values, values, scale)")).document
        findings = self.compiler.assess(fma).findings
        self.assertIn("ELEMENTWISE_SHAPE_MISMATCH", [f.code for f in findings])
        self.assertEqual(fma["operations"][2]["parameters"]["instruction"], {"contract": "ptx.fma.rn.f32"})

    def test_existing_explicit_singleton_axis_broadcast_remains_valid(self):
        source = self.scalar_source("lm.mul(values, lm.broadcast(scale, axis=0))")
        source = source.replace("(2,32)", "(2,1,32)").replace("x[row,:]", "x[row,:,:]").replace("out[row,:]", "out[row,:,:]")
        assessment = self.compiler.assess(parse(source).document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        self.compiler.lower(assessment)

    def test_writes_depend_on_preceding_reads_even_when_the_read_value_is_unused(self):
        source = FMA.replace('c: cake.Tensor((8, 128), "fp32")',
                             'c: cake.Tensor((8, 128), "fp32", mode="state")')
        source = source.replace('lm.store(y[batch, :], y_tile, id="store_y")',
                                'lm.store(c[batch, :], a_tile, id="update_c")\n'
                                '        lm.store(y[batch, :], y_tile, id="store_y")')
        doc = parse(source).document
        update = next(op for op in doc["operations"] if op["id"] == "update_c")
        self.assertIn("load_c", update["depends_on"])

    def test_foreign_code_and_unsupported_control_flow_are_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "executed"
            injected = f"__import__('pathlib').Path({str(marker)!r}).write_text('executed')\n"
            cases = [injected + FMA, FMA + injected, "import os\n" + FMA,
                     FMA.replace('        y_tile = ', '        if True:\n            y_tile = '),
                     FMA.replace('with compute:', 'while True:'),
                     FMA.replace('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'eval("1")')]
            for source in cases:
                with self.subTest(source=source[-80:]), self.assertRaises(FrontendError):
                    parse(source, filename="candidate.py")
            self.assertFalse(marker.exists())

    def test_supported_input_errors_have_locations_instead_of_python_exceptions(self):
        replacements = [
            ('warps=[0, 1, 2, 3]', 'warps="bad"'),
            ('tile=1)', 'tile="bad")'),
            ('a[batch, :]', 'a[0, :]'),
            ('a[batch, :]', 'a[batch, ::2]'),
            ('a[batch, :]', 'a[batch, 128:129]'),
            ('reuse="streamed", id="load_a"', 'out=[], id="load_a"'),
            ('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'lm.fma(a_tile, b_tile, 1.0)'),
            ('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'lm.unknown(a_tile)'),
            ('registers_per_thread": 64', 'registers_per_thread": 1e400'),
            ('name="fma-b8-smoke"', 'name="\\ud800"'),
            ('def fma(lm, a:', 'def fma(lm, lm:'),
        ]
        for old, new in replacements:
            with self.subTest(new=new), self.assertRaises(FrontendError) as caught:
                parse(FMA.replace(old, new), filename="candidate.py")
            self.assertEqual(caught.exception.location.filename, "candidate.py")
            self.assertGreater(caught.exception.location.line, 0)

    def test_single_assignment_and_explicit_out_do_not_silently_rebind_storage(self):
        for source in (
            FMA.replace('y_tile = lm.fma', 'a_tile = lm.fma'),
            FMA.replace('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'a_tile'),
            FMA.replace('lm.fma(a_tile, b_tile, c_tile, id="fma")', 'lm.fma(a_tile, b_tile, c_tile, out=a_tile)'),
        ):
            with self.subTest(source=source[-180:]), self.assertRaises(FrontendError):
                parse(source)

    def test_indexed_views_are_not_erased_at_buffer_identity_boundaries(self):
        softmax = (EXAMPLES / "softmax.py").read_text()
        pipeline = (EXAMPLES / "kmeans_pipeline.py").read_text()
        cases = (
            FMA.replace('a[batch, :]', 'a[batch, :64][batch, :]'),
            FMA.replace('a[batch, :]', 'a[:, :64][batch, :]'),
            FMA.replace('lm.program(a,', 'lm.program(a[:, :64],'),
            softmax.replace('lm.broadcast(rowmax,', 'lm.broadcast(rowmax[:1],'),
            pipeline.replace('lm.range(centroids,', 'lm.range(centroids[:, :64],'),
        )
        for source in cases:
            with self.subTest(source=source[-150:]), self.assertRaises(FrontendError) as caught:
                parse(source, filename="view.py")
            self.assertIn("indexed views cannot be used here", str(caught.exception))
            self.assertEqual(caught.exception.code, "PYTHON_SYNTAX")
        # Ordinary operation addresses retain the view for the existing Verifier.
        doc = parse(FMA.replace('lm.fma(a_tile,', 'lm.fma(a_tile[:64],')).document
        result = self.compiler.assess(doc)
        self.assertFalse(result.accepted)
        self.assertIn("ACCESS_BUFFER_LOCAL", [finding.code for finding in result.findings])

    def test_resource_ownership_fields_cannot_be_silently_overridden(self):
        original = (EXAMPLES / "kmeans_pipeline.py").read_text()
        for old, new in (
            ('lm.smem(98304)', 'lm.smem(98304, space="global")'),
            ('smem_operands.view(dtype=', 'smem_operands.view(space="global", dtype='),
            ('smem_operands.view(dtype=', 'smem_operands.view(allocation="other", dtype='),
        ):
            with self.subTest(new=new), self.assertRaises(FrontendError):
                parse(original.replace(old, new))

    def test_structural_errors_reuse_the_canonical_parser_and_source_path(self):
        with self.assertRaises(FrontendError) as caught:
            parse(FMA.replace('warps=[0, 1, 2, 3]', 'warps=[0, 1, 2, 3], imaginary=True'), filename="fma.py")
        self.assertEqual(caught.exception.code, "SCHEDULE_STRUCTURE")
        self.assertIn("schedule.roles[0]", str(caught.exception))
        self.assertEqual(caught.exception.location.line, 9)

    def test_utf8_source_columns_are_character_columns(self):
        source = FMA.replace('y_tile = lm.fma(a_tile, b_tile, c_tile, id="fma")', 'label = "中文"; lm.unknown()')
        line = next(line for line in source.splitlines() if 'lm.unknown' in line)
        with self.assertRaises(FrontendError) as caught:
            parse(source, filename="unicode.py")
        self.assertEqual(caught.exception.location.column, line.index('lm.unknown') + 1)

    def test_decorator_and_file_entry_points_share_the_same_elaboration(self):
        spec = importlib.util.spec_from_file_location("frontend_fma_example", EXAMPLES / "fma.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertIsInstance(module.fma, ScheduleSource)
        parsed = read_schedule(EXAMPLES / "fma.py")
        self.assertEqual(module.fma.document, parsed.document)
        self.assertEqual(module.fma.location_for("operations[3]"), parsed.location_for("operations[3]"))
        copied = parsed.document
        copied["operations"].clear()
        self.assertTrue(parsed.document["operations"])

    def test_json_assessment_preserves_decoded_input_behavior(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.json"
            for value in [[], None, 7]:
                path.write_text(json.dumps(value))
                with self.subTest(value=value), self.assertRaises(CompilerError):
                    self.compiler.assess_file(path)
            doc = parse(FMA).document
            doc["operations"][3]["parameters"]["scalar"] = float("inf")
            path.write_text(json.dumps(doc))
            # The existing mapping API rejects non-serializable values by raising.
            # Adding Python input must not introduce a different JSON decoding path.
            with self.assertRaises(ValueError) as direct:
                self.compiler.assess(doc)
            with self.assertRaises(type(direct.exception)) as file_input:
                self.compiler.assess_file(path)
            self.assertEqual(str(file_input.exception), str(direct.exception))

    def run_cli(self, path, command="assess", output=None, format="json"):
        argv = ["--project-root", str(ROOT), "compiler", command, "--format", format,
                "--revision", str(DRAFT), str(path)]
        if output is not None:
            argv.extend(["--output", str(output)])
        with redirect_stdout(StringIO()) as stdout, redirect_stderr(StringIO()) as stderr:
            code = main(argv)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_cli_assess_and_lower_python_use_existing_compiler_path(self):
        for format in ("json", "text"):
            with self.subTest(format=format), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "generated.py"
                code, stdout, stderr = self.run_cli(EXAMPLES / "fma.py", format=format)
                self.assertEqual((code, stderr), (0, ""))
                if format == "json":
                    self.assertTrue(json.loads(stdout)["lowering_eligible"])
                else:
                    self.assertIn("未运行 GPU", stdout)
                code, stdout, stderr = self.run_cli(EXAMPLES / "fma.py", "lower", output, format)
                self.assertEqual((code, stderr), (0, ""))
                self.assertEqual(output.read_text(), self.compiler.lower(self.compiler.assess_file(EXAMPLES / "fma.py")).source)
                ast.parse(output.read_text())

    def test_cli_reports_original_semantic_findings_at_python_source_lines(self):
        source = FMA.replace('b: cake.Tensor((8, 128), "fp32")', 'b: cake.Tensor((8, 128), "bf16")')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.py"
            path.write_text(source)
            expected = self.compiler.assess(parse(source).document)
            self.assertFalse(expected.accepted)
            code, stdout, stderr = self.run_cli(path)
            self.assertEqual((code, stderr), (2, ""))
            observed = json.loads(stdout)["findings"]
            self.assertEqual([{k: v for k, v in row.items() if k != "source"} for row in observed],
                             [asdict(finding) for finding in expected.findings])
            arithmetic = [row for row in observed if row["path"].startswith("operations[3]")]
            self.assertTrue(arithmetic)
            self.assertTrue(all('lm.fma' in source.splitlines()[row["source"]["line"] - 1] for row in arithmetic))
            code, stdout, stderr = self.run_cli(path, format="text")
            self.assertEqual(code, 2)
            self.assertIn(str(path) + ":", stdout)

    def test_cli_lower_does_not_write_when_python_candidate_is_not_lowerable(self):
        source = FMA.replace('    batch =', '    extra = lm.role(warps=[4])\n    batch =')
        with tempfile.TemporaryDirectory() as directory:
            path, output = Path(directory) / "bad.py", Path(directory) / "output.py"
            path.write_text(source)
            code, stdout, stderr = self.run_cli(path, "lower", output)
            self.assertEqual((code, stderr), (2, ""))
            report = json.loads(stdout)
            self.assertFalse(report["lowering_eligible"])
            self.assertIn("TRITON_ROLE_COUNT", [f["code"] for f in report["findings"]])
            self.assertFalse(output.exists())

    def test_cli_syntax_refusal_is_structured_and_located(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.py"
            for source in ("import os\n" + FMA, "def broken(:\n"):
                path.write_text(source)
                code, stdout, stderr = self.run_cli(path)
                self.assertEqual((code, stderr), (2, ""))
                finding = json.loads(stdout)["findings"][0]
                self.assertEqual(finding["code"], "PYTHON_SYNTAX")
                self.assertEqual(finding["source"]["filename"], str(path.resolve()))
                code, stdout, stderr = self.run_cli(path, format="text")
                self.assertEqual((code, stdout), (2, ""))
                self.assertIn(str(path.resolve()) + ":", stderr)


class AmdPythonFrontendTests(unittest.TestCase):
    SOURCE = 'from open_cake_ir.compiler import frontend as cake\n@cake.schedule(name="q8-python", target="gfx1151", backend="triton", entry_point="q8_python")\ndef candidate(lm, x: cake.Tensor((32,), "fp32"), q8: cake.Tensor((1,36), "uint8", mode="output", packed_block={"format":"ggml_q8_1_v1","record_axis":1})):\n    compute = lm.role(warps=[0])\n    tile = lm.program(x, axis=0, dimension=0, tile=32)\n    with compute:\n        flat = lm.load(x[tile])\n        blocks = lm.buffer(dtype="fp32", shape=(1,32))\n        lm.reshape(flat, out=blocks)\n        absolute = lm.abs(blocks)\n        amax = lm.reduce(absolute, op="max", axis=1, scope="cta", algorithm="xor_tree_32")\n        sums = lm.reduce(blocks, op="sum", axis=1, scope="cta", algorithm="xor_tree_32")\n        d = lm.divide_no_nan(amax, 127.0)\n        scaled = lm.divide_no_nan(blocks, lm.broadcast(d, axis=0))\n        rounded = lm.round(scaled, rounding="nearest_away_from_zero")\n        qs = lm.cast(rounded, to="int8", rounding="toward_zero", overflow="forbid")\n        half_d = lm.cast(d, to="fp16", rounding="nearest_even", overflow="ieee")\n        half_s = lm.cast(sums, to="fp16", rounding="nearest_even", overflow="ieee")\n        lm.store(q8[:], half_d, half_s, qs)\n'

    def test_python_reaches_the_canonical_packed_q8_path(self):
        source = parse(self.SOURCE)
        compiler = Compiler.load(ROOT, DRAFT)
        assessment = compiler.assess(source.document)
        self.assertTrue(assessment.lowering_eligible, assessment.findings)
        casts = [op for op in source.document["operations"] if op["kind"] == "cast"]
        self.assertEqual([op["parameters"]["to"] for op in casts], ["int8", "fp16", "fp16"])
        self.assertEqual(Schedule.from_dict(source.document).buffer("qs").dtype.value, "int8")
        lowering = compiler.lower(assessment)
        self.assertEqual(lowering.toolchain_requirements["triton_target"], {"backend": "hip", "arch": "gfx1151", "warp_size": 32})
        self.assertIn("record_ptrs", lowering.source)

    def test_python_rejects_unpaired_cast_policy_and_bad_packed_prefix(self):
        for value in (self.SOURCE.replace(', overflow="forbid"', ''), self.SOURCE.replace('q8[:]', 'q8[:, :, :]')):
            with self.subTest(source=value), self.assertRaises(FrontendError):
                parse(value)


if __name__ == "__main__":
    unittest.main()
