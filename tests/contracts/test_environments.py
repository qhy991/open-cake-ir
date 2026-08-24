from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation import LaunchableCandidate  # noqa: E402
from open_cake_ir.evidence import EvidenceStore  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    BuildRequest,
    CandidateSubmission,
    DirectCudaEnvironment,
    EnvironmentResult,
    NvccToolchainBuilder,
)
from open_cake_ir.lab.environments import _ptxas_finding_rows  # noqa: E402


class EnvironmentContractTests(unittest.TestCase):
    def test_triton_qualification_externally_anchors_the_compile_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            triton = root / "triton"
            (triton / "backends").mkdir(parents=True)
            (triton / "compiler").mkdir()
            (root / "triton-3.4.0.dist-info").mkdir()
            (root / "triton-3.4.0.dist-info/METADATA").write_text(
                "Metadata-Version: 2.1\nName: triton\nVersion: 3.4.0\n",
                encoding="utf-8",
            )
            (triton / "__init__.py").write_text(
                "from . import language\n"
                "def jit(function):\n"
                "    return function\n",
                encoding="utf-8",
            )
            (triton / "language.py").write_text("constexpr = object()\n", encoding="utf-8")
            (triton / "backends/__init__.py").write_text("", encoding="utf-8")
            (triton / "backends/compiler.py").write_text(
                "class GPUTarget:\n"
                "    def __init__(self, backend, capability, warp_size):\n"
                "        self.backend = backend\n"
                "        self.capability = capability\n"
                "        self.warp_size = warp_size\n",
                encoding="utf-8",
            )
            (triton / "compiler/__init__.py").write_text(
                "from types import SimpleNamespace\n"
                "class ASTSource:\n"
                "    def __init__(self, kernel, signature, constants):\n"
                "        self.kernel = kernel\n"
                "        self.signature = signature\n"
                "        self.constants = constants\n"
                "def compile(source, *, target, options):\n"
                "    return SimpleNamespace(\n"
                "        asm={\n"
                "            'source': 'expanded fixture source',\n"
                "            'ttir': 'ttir fixture',\n"
                "            'ttgir': 'ttgir fixture',\n"
                "            'llir': 'llir fixture',\n"
                "            'ptx': '.target sm_100a',\n"
                "            'cubin': b'\\x7fELFfixture',\n"
                "        },\n"
                "        metadata=SimpleNamespace(shared=164880),\n"
                "    )\n",
                encoding="utf-8",
            )
            evidence_root = root / "evidence"
            anchor_path = root / "triton-qualification-anchor.json"
            command = [
                sys.executable,
                str(ROOT / "tools/qualify_triton_toolchain.py"),
                "--project-root",
                str(ROOT),
                "--revision",
                str(ROOT / "compiler/revision.lock.json"),
                "--schedule",
                str(ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json"),
                "--evidence-root",
                str(evidence_root),
                "--anchor-output",
                str(anchor_path),
                "--run-id",
                "triton-qualification-anchor-contract",
            ]

            completed = subprocess.run(
                command,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(root)},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr.decode())
            audit = EvidenceStore.open(evidence_root).audit_run(
                "triton-qualification-anchor-contract"
            )
            anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
            self.assertEqual(anchor_path.stat().st_mode & 0o444, 0o444)
            compile_event = next(
                event
                for event in EvidenceStore.open(evidence_root).replay_events(audit.run_id)
                if event["kind"] == "toolchain_compiled"
            )
            self.assertTrue(audit.integrity)
            self.assertEqual(anchor["authority_sha256"], audit.authority_sha256)
            self.assertEqual(anchor["terminal_seal_sha256"], audit.terminal_seal_sha256)
            self.assertEqual(
                anchor["qualification_artifact_sha256"],
                compile_event["payload"]["candidate_record_sha256"],
            )
            anchor_bytes = anchor_path.read_bytes()
            second_evidence_root = root / "second-evidence"
            blocked_command = list(command)
            blocked_command[blocked_command.index("--evidence-root") + 1] = str(
                second_evidence_root
            )
            blocked_command[blocked_command.index("--run-id") + 1] = (
                "triton-qualification-create-only-contract"
            )

            blocked = subprocess.run(
                blocked_command,
                cwd=ROOT,
                env={**os.environ, "PYTHONPATH": str(root)},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )

            self.assertNotEqual(blocked.returncode, 0)
            self.assertEqual(anchor_path.read_bytes(), anchor_bytes)
            self.assertFalse(second_evidence_root.exists())

    def test_nvcc_nonzero_is_bounded_compile_feedback_not_harness_terminal(self) -> None:
        source = (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nvcc = root / "nvcc"
            cuobjdump = root / "cuobjdump"
            nvcc.write_text("#!/bin/sh\necho 'invalid candidate' >&2\nexit 7\n")
            cuobjdump.write_text("#!/bin/sh\nexit 0\n")
            nvcc.chmod(0o700)
            cuobjdump.chmod(0o700)
            environment = DirectCudaEnvironment(
                NvccToolchainBuilder(nvcc=nvcc, cuobjdump=cuobjdump),
                toolchain_requirements={"compiler": "nvcc", "target": "sm_100a"},
                authority_document={"environment_kind": "direct_cuda"},
            )

            result = environment.build(CandidateSubmission.seal("text/x-cuda", source))

        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["stage"], "compile")
        self.assertIn("invalid candidate", result.feedback["diagnostic"])
        self.assertEqual(
            set(result.artifact_payloads), {"toolchain_stdout", "toolchain_stderr"}
        )

    def test_environment_cannot_replace_the_sealed_submission(self) -> None:
        with self.assertRaisesRegex(ValueError, "replaced"):
            EnvironmentResult(
                "launchable",
                "1" * 64,
                LaunchableCandidate(
                    candidate_sha256="2" * 64,
                    target="sm_100a",
                    entry_point="kernel",
                    artifact_roles={"cubin": "3" * 64},
                    launch_spec_sha256="4" * 64,
                ),
                {},
            )

    def test_nvcc_builder_retains_authored_ptx_cubin_sass_and_manifest_bytes(self) -> None:
        source = (ROOT / "contracts/scaffolds/direct-cuda-headline-v1.cu").read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nvcc = root / "nvcc"
            cuobjdump = root / "cuobjdump"
            # ptxas reports resources on the assembly pass only, and writes them to
            # stderr alongside its own wall clock.
            nvcc.write_text(
                "#!/usr/bin/env python3\n"
                "import pathlib,sys\n"
                "out=pathlib.Path(sys.argv[sys.argv.index('-o')+1])\n"
                "cubin='--cubin' in sys.argv\n"
                "out.write_bytes(b'\\x7fELFcubin' if cubin else b'.target sm_100a')\n"
                "if cubin:\n"
                "    sys.stderr.write('ptxas info    : Used 96 registers, 33792 bytes smem\\n'\n"
                "                     'ptxas info    : Compile time = 4.760 ms\\n')\n"
            )
            cuobjdump.write_text("#!/usr/bin/env python3\nprint('SASS')\n")
            nvcc.chmod(0o700)
            cuobjdump.chmod(0o700)
            candidate = NvccToolchainBuilder(
                nvcc=nvcc,
                cuobjdump=cuobjdump,
            ).build(
                BuildRequest(
                    candidate_sha256=sha256(source).hexdigest(),
                    source=source,
                    source_role="authored_source",
                    source_sha256=sha256(source).hexdigest(),
                    target="sm_100a",
                    entry_point="cake_flash_kmeans_assign",
                    toolchain_requirements={"compiler": "nvcc", "target": "sm_100a"},
                )
            )

        self.assertEqual(
            set(candidate.artifact_roles),
            {
                "authored_source",
                "ptx",
                "cubin",
                "sass",
                "toolchain_resource_report",
                "launch_manifest",
            },
        )
        # The arms are matched on one static channel each. Discarding what ptxas already
        # measured left this arm's author blind to the resource facts the Open Cake arm
        # reads off its Schedule before compiling at all.
        report = candidate.artifact_payloads["toolchain_resource_report"]
        self.assertIn(b"Used 96 registers", report)
        # Compile time is a fact about the machine, not the candidate; keeping it would
        # reseal the same source under a different digest on every build.
        self.assertNotIn(b"Compile time", report)
        rows = _ptxas_finding_rows(candidate)
        self.assertEqual([row["code"] for row in rows], ["TOOLCHAIN_RESOURCE_REPORT"])
        self.assertFalse(rows[0]["blocks_acceptance"])
        self.assertFalse(rows[0]["blocks_lowering"])
        self.assertIn("33792 bytes smem", rows[0]["message"])
        self.assertEqual(set(candidate.artifact_payloads), set(candidate.artifact_roles))
        self.assertTrue(candidate.artifact_payloads["cubin"].startswith(b"\x7fELF"))
        manifest = json.loads(candidate.artifact_payloads["launch_manifest"])
        self.assertEqual(manifest["kernel_name"], candidate.entry_point)
        self.assertEqual(
            sha256(candidate.artifact_payloads["launch_manifest"]).hexdigest(),
            candidate.launch_spec_sha256,
        )


if __name__ == "__main__":
    unittest.main()
