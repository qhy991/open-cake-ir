"""MLX host bridge contracts: signature translation, launch conversion and refusals.

Only the last test needs MLX or a GPU. The rest check the translation itself against
the real emitter, so a change to the emitted signature fails here rather than at the
first dispatch on a machine that happens to have MLX installed.
"""

from __future__ import annotations

from pathlib import Path
import unittest

from open_cake_ir.compiler import frontend
from open_cake_ir.compiler.backends import metal
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target
from tools.metal import mlx_adapter

ROOT = Path(__file__).resolve().parents[2]
TARGET = "apple_gpu_family7"


def emission(example: str):
    """Emit one released-vocabulary example at the exact target the MLX host admits."""
    path = ROOT / "examples/python" / example
    document = frontend.parse(path.read_text().replace("apple_gpu_family8", TARGET),
                              filename=str(path)).document
    target = Target.load(ROOT / f"compiler/targets/{TARGET}.json")
    return metal.emit(Schedule.from_dict(document), target)


def bridge_of(example: str):
    emitted = emission(example)
    return emitted, mlx_adapter.bridge(
        emitted.source, entry_point=emitted.entry_point,
        threadgroups_per_grid=emitted.toolchain["threadgroups_per_grid"],
        threads_per_threadgroup=emitted.toolchain["threads_per_threadgroup"])


class BridgeContracts(unittest.TestCase):
    def test_body_is_carried_verbatim_under_alias_declarations(self):
        emitted, bridged = bridge_of("metal_row_sum.py")
        self.assertEqual(bridged.body.splitlines()[:2],
                         ["    uint3 program = threadgroup_position_in_grid;",
                          "    uint lane = thread_index_in_simdgroup;"])
        # Everything the emitter wrote between its signature and its closing brace has to
        # reach MLX unchanged; the aliases are the only statements this host introduces.
        body = "\n".join(bridged.body.splitlines()[2:])
        self.assertIn(body, emitted.source)
        self.assertIn("// CAKE_KERNEL_END", body)
        self.assertNotIn("kernel void", bridged.body)

    def test_header_keeps_the_preamble_and_ends_a_comment_line(self):
        _, bridged = bridge_of("metal_row_sum.py")
        self.assertIn("#pragma METAL fp contract(off)", bridged.header)
        # MLX concatenates the header straight onto the signature it generates, so a
        # header ending inside a `//` comment would comment that signature out.
        self.assertTrue(bridged.header.endswith("\n"))

    def test_address_space_selects_input_and_output_names(self):
        _, bridged = bridge_of("metal_elementwise.py")
        self.assertEqual(bridged.input_names, ("v0", "v1"))
        self.assertEqual(bridged.output_names, ("v2",))
        self.assertTrue(bridged.buffer_indices_match_emission)

    def test_threadgroup_grid_becomes_a_thread_grid(self):
        emitted, bridged = bridge_of("metal_rmsnorm.py")
        rows = emitted.toolchain["threadgroups_per_grid"][0]
        self.assertEqual(bridged.threadgroup, (32, 1, 1))
        self.assertEqual(bridged.grid, (rows * 32, 1, 1))

    def test_refusals_name_what_differs(self):
        emitted, _ = bridge_of("metal_row_sum.py")
        grid = emitted.toolchain["threadgroups_per_grid"]
        for message, kwargs in (
            ("entry point differs", {"entry_point": "other_kernel"}),
            ("one SIMD group per threadgroup", {"threads_per_threadgroup": [64, 1, 1]}),
        ):
            with self.subTest(message=message):
                arguments = {"entry_point": emitted.entry_point, "threadgroups_per_grid": grid,
                             "threads_per_threadgroup": [32, 1, 1], **kwargs}
                with self.assertRaisesRegex(ValueError, message):
                    mlx_adapter.bridge(emitted.source, **arguments)
        for message, source in (
            ("declares no kernel", "// nothing to bridge\n"),
            ("unsupported emitted kernel parameter",
             emitted.source.replace("    uint lane [[thread_index_in_simdgroup]]) {",
                                    "    threadgroup float* share [[threadgroup(0)]]) {")),
            ("body is truncated", emitted.source.replace("    // CAKE_KERNEL_END\n", "")),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    mlx_adapter.bridge(source, entry_point=emitted.entry_point,
                                       threadgroups_per_grid=grid, threads_per_threadgroup=[32, 1, 1])


class HostSelectionContracts(unittest.TestCase):
    def test_the_swift_runner_stays_the_default_and_binds_late(self):
        from unittest.mock import patch

        from tools.metal import check_correctness

        self.assertIs(check_correctness.host_module("mlx"), mlx_adapter)
        swift = check_correctness.host_module("swift")
        with patch.object(check_correctness, "invoke") as patched:
            swift.invoke("binary", "directory")
        patched.assert_called_once_with("binary", "directory")

    def test_a_positional_caller_still_gets_the_swift_host(self):
        """benchmark.py calls evaluate_case positionally and must keep the Swift runner."""
        from unittest.mock import Mock, patch

        from tools.metal import check_correctness

        with patch.object(check_correctness, "prepare_case", return_value={"buffers": []}), \
             patch.object(check_correctness, "invoke",
                          return_value={"device": "Apple M1 Pro", "command_status": "completed"}) as invoke:
            case = {}
            check_correctness.evaluate_case(Mock(), "binary", {}, {}, {}, Path("."),
                                            ["Apple M1 Pro"], case)
        invoke.assert_called_once_with("binary", Path("."))
        self.assertEqual(case["gpu_correctness"], "passed")

    def test_host_mlx_reaches_the_mlx_host_and_never_builds_the_swift_one(self):
        """The real run needs a released Compiler and a clean tree; this covers the wiring."""
        import json
        import tempfile
        from unittest.mock import Mock, patch

        from tools.metal import check_correctness

        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            with patch.object(check_correctness, "fresh_receipt", return_value=directory), \
                 patch.object(check_correctness, "runtime_source",
                              return_value={"tracked": True, "clean": True}), \
                 patch.object(check_correctness, "released_compiler",
                              return_value=(Mock(), {"revision_id": "fixture"}, ["Apple M1 Pro"])), \
                 patch.object(mlx_adapter, "compile_runner",
                              side_effect=RuntimeError("mlx host selected")) as mlx_host, \
                 patch.object(check_correctness, "compile_runner") as swift_host, \
                 patch("sys.argv", ["check_correctness", "--output-root", str(directory),
                                    "--target", TARGET, "--host", "mlx"]):
                self.assertEqual(check_correctness.main(), 1)
            receipt = json.loads((directory / "receipt.json").read_text())
        mlx_host.assert_called_once()
        swift_host.assert_not_called()
        self.assertIn("mlx host selected", receipt["error"])
        self.assertEqual(receipt["status"], "failed")


class LiveHostContracts(unittest.TestCase):
    """The one test that needs the exact device; skipped everywhere else."""

    def setUp(self):
        try:
            import mlx.core as mx
        except ImportError:
            self.skipTest("MLX is not installed")
        name = mx.device_info()["device_name"]
        if (name,) != mlx_adapter.EXACT_DEVICE_NAMES[TARGET]:
            self.skipTest(f"{name} is not an exact {TARGET} device")
        self.mx = mx

    def test_bridged_row_sum_matches_an_independent_summation(self):
        import math

        mx = self.mx
        emitted, bridged = bridge_of("metal_row_sum.py")
        kernel = mx.fast.metal_kernel(name=emitted.entry_point, input_names=list(bridged.input_names),
                                      output_names=list(bridged.output_names), source=bridged.body,
                                      header=bridged.header,
                                      compile_options=dict(mlx_adapter.COMPILE_OPTIONS))
        rows, columns = 5, 65
        values = [float((index % 13) - 6) for index in range(rows * columns)]
        x = mx.reshape(mx.array(values, dtype=mx.float32), (rows, columns))
        out = kernel(inputs=[x], grid=bridged.grid, threadgroup=bridged.threadgroup,
                     output_shapes=[(rows,)], output_dtypes=[mx.float32],
                     init_value=mlx_adapter.OUTPUT_FILL)
        mx.eval(out)
        for row, actual in enumerate(out[0].tolist()):
            expected = math.fsum(values[row * columns:(row + 1) * columns])
            self.assertAlmostEqual(actual, expected, delta=1e-4)


if __name__ == "__main__":
    unittest.main()
