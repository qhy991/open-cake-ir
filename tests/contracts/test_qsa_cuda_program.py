from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.evaluation.qsa_cuda import QsaProgramArtifact  # noqa: E402


class QsaCudaProgramContractTests(unittest.TestCase):
    def _program(self, root: Path) -> Path:
        candidate = root / "candidate"
        candidate.mkdir()
        (candidate / "program.cubin").write_bytes(b"\x7fELFqsa")
        arguments = {
            "pool_layernorm": ["index_k", "k_norm_weight", "normalized_keys"],
            "score_topk": ["index_q", "normalized_keys", "block_indices"],
            "expand": ["block_indices", "token_indices"],
            "attention": ["q", "k", "v", "token_indices", "output"],
        }
        document = {
            "schema_version": 1,
            "abi": "qsa_prefill_task_geometry_v1",
            "arm": "direct_cuda",
            "kernels": [
                {
                    "id": kernel_id,
                    "cubin": "candidate/program.cubin",
                    "kernel_name": f"qsa_{kernel_id}",
                    "grid": [32768, 1, 1],
                    "block": [256, 1, 1],
                    "dynamic_shared_memory_bytes": 0,
                    "arguments": arguments[kernel_id],
                    "hidden_null_pointer_parameters": 0,
                }
                for kernel_id in (
                    "pool_layernorm",
                    "score_topk",
                    "expand",
                    "attention",
                )
            ],
        }
        source = candidate / "program.json"
        source.write_text(json.dumps(document), encoding="utf-8")
        return source

    def test_direct_program_loads_only_the_closed_qsa_launch_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = QsaProgramArtifact.load(root, self._program(root))
            self.assertEqual(artifact.arm, "direct_cuda")
            self.assertEqual(
                tuple(kernel.kernel_id for kernel in artifact.kernels),
                ("pool_layernorm", "score_topk", "expand", "attention"),
            )
            self.assertEqual(len({kernel.cubin_path for kernel in artifact.kernels}), 1)

    def test_program_rejects_an_argument_binding_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = self._program(root)
            document = json.loads(source.read_text(encoding="utf-8"))
            document["kernels"][1]["arguments"] = [
                "normalized_keys",
                "index_q",
                "block_indices",
            ]
            source.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "contract differs"):
                QsaProgramArtifact.load(root, source)


if __name__ == "__main__":
    unittest.main()
