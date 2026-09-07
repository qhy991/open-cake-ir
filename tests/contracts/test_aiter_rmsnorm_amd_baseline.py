from __future__ import annotations

import ast
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "examples/gpu"))

import aiter_rmsnorm_amd_baseline as baseline  # noqa: E402
from open_cake_ir.evaluation import WorkloadContract  # noqa: E402
from tools.release_executor import _source_paths  # noqa: E402


WORKLOAD = ROOT / baseline.WORKLOAD


def _build_tools(root: Path) -> dict[str, dict[str, object]]:
    rocm_bin = root / "rocm/bin"
    tool_bin = root / "tools"
    rocm_bin.mkdir(parents=True)
    tool_bin.mkdir()
    paths = {
        "hipcc": rocm_bin / "hipcc",
        "hipconfig": rocm_bin / "hipconfig",
        "rocminfo": rocm_bin / "rocminfo",
        "cxx": tool_bin / "cxx",
        "git": tool_bin / "git",
        "ninja": tool_bin / "ninja",
    }
    for kind, path in paths.items():
        path.write_bytes(f"#!/bin/sh\n# {kind}\nexit 0\n".encode())
        path.chmod(0o755)
    paths["sh"] = Path("/bin/sh").resolve(strict=True)
    return {kind: {"kind": kind, "path": str(path)} for kind, path in paths.items()}


def _runtime_libraries(root: Path) -> dict[str, dict[str, object]]:
    path = root / "libxml2.so.2"
    path.write_bytes(b"libxml2 fixture\n")
    return {
        "libxml2.so.2": {
            "soname": "libxml2.so.2",
            "path": str(path),
        }
    }


def _executor(*, owns_runner: bool) -> SimpleNamespace:
    path = baseline.RUNNER_SOURCE if owns_runner else "other.py"
    build_tools = tuple(
        {"kind": kind, "path": f"/qualified/{kind}"}
        for kind in ("cxx", "git", "hipcc", "hipconfig", "ninja", "rocminfo", "sh")
    )
    runtime_libraries = (
        {"soname": "libxml2.so.2", "path": "/qualified/libxml2.so.2"},
    )
    return SimpleNamespace(
        executor_id="open-cake-ir-gfx1151-v1",
        reference={
            "path": "runtime/executors/open-cake-ir-gfx1151-v1.json",
            "executor_id": "open-cake-ir-gfx1151-v1",
            "canonical_sha256": "a" * 64,
        },
        document={
            "schema_version": 2,
            "sources": ({"path": path},),
            "host_environment": {
                "runtime_kind": "hip",
                "runtime_libraries": runtime_libraries,
                "tools": {"build_tools": build_tools},
            },
        },
    )


class AiterRmsnormPrepareTests(unittest.TestCase):
    def test_frozen_workload_is_the_only_semantic_authority(self) -> None:
        workload = baseline._load_workload(ROOT)

        self.assertIsInstance(workload, WorkloadContract)
        self.assertEqual(
            workload.workload_id,
            "llama-rmsnorm-mul-fp32-independent-v2",
        )
        self.assertEqual(
            workload.case_ids,
            ("seeded_random", "reduction_rsqrt_stress"),
        )
        self.assertEqual(workload.canonical_sha256, baseline.WORKLOAD_SHA256)

    def test_prepare_is_import_free_and_makes_no_execution_claim(self) -> None:
        aiter_source = {
            "repository": baseline.AITER_REPOSITORY,
            "tag": baseline.AITER_TAG,
            "revision": baseline.AITER_REVISION,
            "tree_clean": True,
            "submodules_used": [],
        }
        with (
            patch.object(
                baseline,
                "_project_git_state",
                return_value={"revision": "a" * 40, "tree_clean": True},
            ),
            patch.object(
                baseline,
                "_admit_aiter_checkout",
                return_value=aiter_source,
            ),
        ):
            workload, summary = baseline._prepare(ROOT, Path("/fixture/aiter"))

        self.assertEqual(workload.workload_id, summary["workload"]["workload_id"])
        self.assertEqual(summary["status"], "prepared")
        self.assertEqual(summary["aiter_source"], aiter_source)
        self.assertEqual(
            summary["baseline"]["entry_point"],
            "aiter.ops.rmsnorm.rms_norm_opus",
        )
        self.assertFalse(summary["evaluation"]["torch_imported"])
        self.assertFalse(summary["evaluation"]["aiter_imported"])
        self.assertFalse(summary["evaluation"]["jit_started"])
        self.assertFalse(summary["evaluation"]["gpu_submitted"])
        self.assertEqual(summary["evaluation"]["operator_calls"], 0)
        self.assertFalse(summary["scope"]["performance_measured"])
        self.assertFalse(summary["scope"]["promotion_authorized"])

    def test_source_checkout_rejects_wrong_revision_and_dirty_tree(self) -> None:
        root = Path("/fixture/aiter")

        def wrong_revision(_: Path, *arguments: str, **__: object) -> str:
            command = tuple(arguments)
            if command == ("rev-parse", "--show-toplevel"):
                return str(root)
            if command == ("rev-parse", "HEAD"):
                return "0" * 40
            raise AssertionError(command)

        with (
            patch.object(Path, "is_symlink", return_value=False),
            patch.object(Path, "resolve", return_value=root),
            patch.object(baseline, "_git", side_effect=wrong_revision),
            self.assertRaisesRegex(ValueError, "revision differs"),
        ):
            baseline._admit_aiter_checkout(root)

        def dirty(_: Path, *arguments: str, **__: object) -> str:
            command = tuple(arguments)
            values = {
                ("rev-parse", "--show-toplevel"): str(root),
                ("rev-parse", "HEAD"): baseline.AITER_REVISION,
                ("rev-parse", f"refs/tags/{baseline.AITER_TAG}^{{commit}}"): (
                    baseline.AITER_REVISION
                ),
                ("remote",): "origin\n",
                ("remote", "get-url", "origin"): baseline.AITER_REPOSITORY,
                ("status", "--porcelain", "--untracked-files=all"): " M aiter/ops/rmsnorm.py\n",
            }
            return values[command]

        with (
            patch.object(Path, "is_symlink", return_value=False),
            patch.object(Path, "resolve", return_value=root),
            patch.object(baseline, "_git", side_effect=dirty),
            self.assertRaisesRegex(ValueError, "tree must be clean"),
        ):
            baseline._admit_aiter_checkout(root)

    def test_running_checkout_cannot_be_relabelled_as_another_project_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "project root differs"):
                baseline._prepare(Path(directory), Path("/fixture/aiter"))

    def test_source_checkout_rejects_ignored_import_shadow(self) -> None:
        root = Path("/fixture/aiter")

        def ignored(_: Path, *arguments: str, **__: object) -> str:
            command = tuple(arguments)
            values = {
                ("rev-parse", "--show-toplevel"): str(root),
                ("rev-parse", "HEAD"): baseline.AITER_REVISION,
                ("rev-parse", f"refs/tags/{baseline.AITER_TAG}^{{commit}}"): (
                    baseline.AITER_REVISION
                ),
                ("remote",): "origin",
                ("remote", "get-url", "origin"): baseline.AITER_REPOSITORY,
                ("status", "--porcelain", "--untracked-files=all"): "",
                (
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "--",
                    "aiter",
                ): "aiter/__pycache__/__init__.cpython-312.pyc",
            }
            return values[command]

        with (
            patch.object(Path, "is_symlink", return_value=False),
            patch.object(Path, "resolve", return_value=root),
            patch.object(baseline, "_git", side_effect=ignored),
            self.assertRaisesRegex(ValueError, "ignored import artifacts"),
        ):
            baseline._admit_aiter_checkout(root)


class AiterRmsnormAuthorityTests(unittest.TestCase):
    def test_executor_must_own_runner_and_future_source_closure_includes_it(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not own the AITER runner"):
            baseline._admit_executor_contract(_executor(owns_runner=False))
        baseline._admit_executor_contract(_executor(owns_runner=True))

        relative_sources = {
            path.relative_to(ROOT).as_posix() for path in _source_paths(ROOT)
        }
        self.assertIn(baseline.RUNNER_SOURCE, relative_sources)

    def test_external_attempt_directories_must_be_new_disjoint_and_outside_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            external = Path(directory).resolve()
            aiter = external / "aiter"
            aiter.mkdir()
            evidence = external / "evidence"
            jit = external / "jit"

            observed_evidence, observed_jit = baseline._resolve_attempt_directories(
                ROOT, aiter, evidence, jit
            )
            self.assertEqual(observed_evidence, evidence)
            self.assertEqual(observed_jit, jit)

            with self.assertRaisesRegex(ValueError, "outside both source checkouts"):
                baseline._resolve_attempt_directories(
                    ROOT, aiter, ROOT / "forbidden-evidence", jit
                )
            with self.assertRaisesRegex(ValueError, "must not overlap"):
                baseline._resolve_attempt_directories(
                    ROOT, aiter, external / "attempt", external / "attempt"
                )

    def test_module_artifact_requires_elf_symbol_and_only_gfx1151(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            module = Path(directory) / "module_rmsnorm.so"
            module.write_bytes(b"\x7fELF fixture amdhsa--gfx1151 payload")
            library = SimpleNamespace(rms_norm_opus=object())
            with patch.object(baseline.ctypes, "CDLL", return_value=library):
                record = baseline._validate_module_artifact(module)
            self.assertEqual(record["code_objects"], ["gfx1151"])
            self.assertEqual(record["exported_symbol"], "rms_norm_opus")

            module.write_bytes(b"\x7fELF fixture amdhsa--gfx1100 payload")
            with self.assertRaisesRegex(RuntimeError, "code object set differs"):
                baseline._validate_module_artifact(module)

            module.write_bytes(b"not-elf amdhsa--gfx1151")
            with self.assertRaisesRegex(RuntimeError, "ELF shared object"):
                baseline._validate_module_artifact(module)

            module.write_bytes(b"\x7fELF fixture amdhsa--gfx1151 payload")
            with (
                patch.object(baseline.ctypes, "CDLL", return_value=object()),
                self.assertRaisesRegex(RuntimeError, "export rms_norm_opus"),
            ):
                baseline._validate_module_artifact(module)

            core = Path(directory) / "module_aiter_core.so"
            core.write_bytes(b"\x7fELF host-only fixture")
            core_library = SimpleNamespace(PyInit_module_aiter_core=object())
            with patch.object(
                baseline.ctypes, "CDLL", return_value=core_library
            ):
                core_record = baseline._validate_core_artifact(core)
            self.assertFalse(core_record["device_code_present"])
            self.assertEqual(core_record["code_objects"], [])

    def test_jit_environment_uses_only_executor_admitted_build_tools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            tools = _build_tools(root)
            runtime_libraries = _runtime_libraries(root)
            old_cxx = baseline.os.environ.get("CXX")
            inherited = {
                "PREBUILD_THREAD_NUM": "99",
                "TORCH_DONT_CHECK_COMPILER_ABI": "1",
            }
            with patch.dict(baseline.os.environ, inherited):
                with baseline._aiter_environment(
                    root, root / "jit", tools, runtime_libraries
                ) as authority:
                    self.assertEqual(
                        baseline.os.environ["CXX"], tools["cxx"]["path"]
                    )
                    self.assertEqual(
                        baseline.os.environ["ROCM_HOME"], str(root / "rocm")
                    )
                    self.assertEqual(
                        authority["build_tools"]["hipcc"],
                        tools["hipcc"]["path"],
                    )
                    self.assertEqual(baseline.os.environ["GPU_ARCHS"], "gfx1151")
                    self.assertEqual(
                        baseline.os.environ["AITER_SYMBOL_VISIBLE"], "0"
                    )
                    self.assertEqual(
                        authority["runtime_libraries"]["libxml2.so.2"],
                        runtime_libraries["libxml2.so.2"]["path"],
                    )
                    self.assertEqual(
                        baseline.os.environ["LD_LIBRARY_PATH"].split(
                            baseline.os.pathsep
                        )[0],
                        str(root),
                    )
                    self.assertNotIn("PREBUILD_THREAD_NUM", baseline.os.environ)
                    self.assertNotIn(
                        "TORCH_DONT_CHECK_COMPILER_ABI", baseline.os.environ
                    )
                self.assertEqual(baseline.os.environ["PREBUILD_THREAD_NUM"], "99")
            self.assertEqual(baseline.os.environ.get("CXX"), old_cxx)

            plan = root / "build.ninja"
            plan.write_text(
                f"cxx = {tools['cxx']['path']}\n"
                f"nvcc = {root / 'rocm/bin/hipcc'}\n"
                "cuda_post_cflags = --offload-arch=gfx1151\n",
                encoding="utf-8",
            )
            record = baseline._validate_build_plan(plan, tools)
            self.assertEqual(record["offload_architectures"], ["gfx1151"])

    def test_preexisting_torch_operator_cannot_capture_the_pinned_call(self) -> None:
        clean = SimpleNamespace(ops=SimpleNamespace(aiter=object()))
        baseline._reject_preexisting_aiter_operator(clean)

        collision = SimpleNamespace(
            ops=SimpleNamespace(
                aiter=SimpleNamespace(_rms_norm_opus_raw=object())
            )
        )
        with self.assertRaisesRegex(RuntimeError, "existed before pinned source"):
            baseline._reject_preexisting_aiter_operator(collision)

    def test_aiter_runtime_architecture_must_match_before_operator_use(self) -> None:
        with patch.object(
            baseline.importlib,
            "import_module",
            return_value=SimpleNamespace(get_gfx_runtime=lambda: "gfx1151"),
        ):
            self.assertEqual(
                baseline._admit_aiter_runtime_architecture(), "gfx1151"
            )
        with (
            patch.object(
                baseline.importlib,
                "import_module",
                return_value=SimpleNamespace(get_gfx_runtime=lambda: "gfx942"),
            ),
            self.assertRaisesRegex(RuntimeError, "architecture differs"),
        ):
            baseline._admit_aiter_runtime_architecture()

    def test_artifact_is_built_validated_and_retained_before_operator_call(self) -> None:
        order: list[str] = []
        executor = _executor(owns_runner=True)
        executor.admit_hip_host = Mock(
            return_value=SimpleNamespace(
                build_tools={"git": {"path": "/qualified/git"}},
                runtime_libraries={
                    "libxml2.so.2": {"path": "/qualified/libxml2.so.2"}
                },
                torch_hip_version="7.2.1",
            )
        )
        torch = SimpleNamespace(version=SimpleNamespace(hip="7.2.1"))
        properties = SimpleNamespace(
            name="fixture", gcnArchName="gfx1151", warp_size=32
        )
        source = {"revision": "a" * 40, "tree_clean": True}
        aiter_source = {"revision": baseline.AITER_REVISION, "tree_clean": True}
        summary = {
            "kind": baseline.RESULT_KIND,
            "workload": {"workload_id": "fixture"},
            "aiter_source": aiter_source,
        }

        def validate_module(_: Path) -> dict[str, object]:
            order.append("validate_module")
            return {"sha256": "b" * 64}

        def validate_plan(
            _: Path, __: object
        ) -> dict[str, object]:
            order.append("validate_plan")
            return {"sha256": "c" * 64}

        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory) / "evidence"
            evidence.mkdir()
            jit = Path(directory) / "jit"
            with (
                patch.object(baseline, "_admit_executor_contract"),
                patch.object(
                    baseline, "_tool_path", return_value=Path("/qualified/git")
                ),
                patch.object(baseline, "_project_git_state", return_value=source),
                patch.object(
                    baseline, "_admit_aiter_checkout", return_value=aiter_source
                ),
                patch.object(
                    baseline, "_admit_device", return_value=(torch, properties)
                ),
                patch.object(baseline, "_reject_preexisting_aiter_operator"),
                patch.object(
                    baseline,
                    "_aiter_environment",
                    return_value=nullcontext({"GPU_ARCHS": "gfx1151"}),
                ),
                patch.object(
                    baseline,
                    "_import_aiter_operator",
                    return_value=(object(), object(), ["aiter/ops/rmsnorm.py"]),
                ),
                patch.object(
                    baseline,
                    "_admit_aiter_runtime_architecture",
                    return_value="gfx1151",
                ),
                patch.object(
                    baseline,
                    "_build_aiter_rmsnorm_module",
                    side_effect=lambda _: order.append("build"),
                ),
                patch.object(
                    baseline,
                    "_validate_module_artifact",
                    side_effect=validate_module,
                ),
                patch.object(
                    baseline,
                    "_validate_core_artifact",
                    return_value={"sha256": "d" * 64},
                ),
                patch.object(
                    baseline, "_validate_build_plan", side_effect=validate_plan
                ),
                patch.object(baseline, "_admit_built_module_runtime_path"),
                patch.object(
                    baseline,
                    "_copy_new",
                    side_effect=lambda *_: order.append("retain"),
                ),
                patch.object(
                    baseline,
                    "_evaluate_cases",
                    side_effect=lambda *_: order.append("evaluate")
                    or [{"passed": True}],
                ),
                patch.object(
                    baseline.importlib.metadata, "version", return_value="fixture"
                ),
            ):
                result = baseline._run_live_impl(
                    ROOT,
                    Path("/fixture/aiter"),
                    executor,
                    Mock(),
                    summary,
                    evidence,
                    jit,
                    {"stage": "executor_admission"},
                )

        self.assertEqual(result, 0)
        self.assertLess(order.index("build"), order.index("evaluate"))
        self.assertLess(order.index("validate_module"), order.index("evaluate"))
        self.assertLess(order.index("retain"), order.index("evaluate"))

    def test_live_failure_is_retained_before_any_runtime_import(self) -> None:
        summary = {
            "kind": baseline.RESULT_KIND,
            "status": "prepared",
            "scope": {"performance_measured": False},
            "workload": {"workload_id": "fixture"},
            "aiter_source": {"revision": baseline.AITER_REVISION},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence = root / "evidence"
            jit = root / "jit"
            with (
                patch.object(
                    baseline,
                    "_admit_executor_contract",
                    side_effect=ValueError("does not own the AITER runner"),
                ),
                patch.object(
                    baseline.importlib,
                    "import_module",
                    side_effect=AssertionError("runtime import must not happen"),
                ),
                self.assertRaisesRegex(ValueError, "does not own the AITER runner"),
            ):
                baseline._run_live(
                    ROOT,
                    Path("/fixture/aiter"),
                    _executor(owns_runner=False),
                    Mock(),
                    summary,
                    evidence,
                    jit,
                )

            failure = baseline._read_json(evidence / "failure.json")
            self.assertEqual(failure["failed_stage"], "executor_admission")
            self.assertEqual(failure["failure_class"], "AUTHORITY_BLOCKED")
            self.assertFalse(failure["gpu_result_authorized"])
            self.assertFalse(failure["performance_conclusion_authorized"])
            self.assertTrue((evidence / "attempt-authority.json").is_file())
            self.assertTrue((evidence / "manifest.json").is_file())
            self.assertFalse(jit.exists())

    def test_prepare_cli_requires_no_torch_or_aiter_import_in_source(self) -> None:
        source = (ROOT / baseline.RUNNER_SOURCE).read_text(encoding="utf-8")
        self.assertNotIn("import torch", source)
        self.assertNotIn("import aiter", source)
        compiler_imports = [
            (node.module, tuple(alias.name for alias in node.names))
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("open_cake_ir.compiler")
        ]
        self.assertEqual(compiler_imports, [("open_cake_ir.compiler.target", ("Target",))])
        self.assertIn('importlib.import_module("torch")', source)
        self.assertIn('importlib.import_module("aiter.ops.rmsnorm")', source)


class AiterRmsnormDeviceContractTests(unittest.TestCase):
    def test_exact_target_is_read_from_compiler_and_runtime_drift_is_refused(self) -> None:
        for arch, width, count, cuda_version, accepted in (
            ("gfx1151", 32, 1, None, True),
            ("gfx942", 32, 1, None, False),
            ("gfx1151", 64, 1, None, False),
            ("gfx1151", "32", 1, None, False),
            ("gfx1151", True, 1, None, False),
            ("gfx1151", 32, True, None, False),
            ("gfx1151", 32, 1, "12.0", False),
        ):
            with self.subTest(arch=arch, width=width, count=count, cuda=cuda_version):
                properties = SimpleNamespace(gcnArchName=arch, warp_size=width)
                torch = SimpleNamespace(
                    version=SimpleNamespace(hip="7.2.1", cuda=cuda_version),
                    cuda=SimpleNamespace(is_available=lambda: True,
                        device_count=lambda: count,
                        get_device_properties=lambda _: properties),
                )
                with patch.object(baseline.importlib, "import_module", return_value=torch):
                    if accepted:
                        self.assertEqual(baseline._admit_device(ROOT), (torch, properties))
                    else:
                        with self.assertRaises(RuntimeError):
                            baseline._admit_device(ROOT)

    def test_byte_and_storage_mutations_reject_an_otherwise_correct_baseline(self) -> None:
        from tests.contracts.test_amd_rmsnorm_search import _FakeTensor, _FakeTorch
        workload = SimpleNamespace(
            document={"semantics": {"epsilon": 1e-6}}, case_ids=("case",),
        )
        for mutation in ("none", "signed_zero", "storage"):
            with self.subTest(mutation=mutation):
                x = _FakeTensor(0.0, 10, shape=(8, 512, 128), dtype=_FakeTorch.float32)
                gamma = _FakeTensor(1.0, 20, shape=(128,), dtype=_FakeTorch.float32)
                output = _FakeTensor(0.0, 30, shape=x.shape, dtype=x.dtype)
                torch = SimpleNamespace(
                    float32=_FakeTorch.float32, uint8=_FakeTorch.uint8,
                    cuda=_FakeTorch.cuda, equal=_FakeTorch.equal,
                    empty_like=lambda _: output,
                )

                def entry_point(out, value, weight, *args):
                    if mutation == "signed_zero":
                        value.value = -0.0
                    elif mutation == "storage":
                        weight.pointer += 1

                with (
                    patch.object(baseline, "generate_rmsnorm_case", return_value=(x, gamma)),
                    patch.object(baseline, "rmsnorm_oracle", return_value=object()),
                    patch.object(baseline, "rmsnorm_metrics", return_value={"passed": True}),
                ):
                    records = baseline._evaluate_cases(torch, entry_point, workload)
                self.assertEqual(records[0]["passed"], mutation == "none")
                self.assertEqual(records[0]["input_unchanged"], mutation == "none")


if __name__ == "__main__":
    unittest.main()
