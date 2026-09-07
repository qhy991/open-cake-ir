from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from open_cake_ir.tasks.qsa.launch import (  # noqa: E402
    _external_open_cake_candidates,
    _select_candidates,
    _task,
    build_parser,
)


class QsaLauncherContractTest(unittest.TestCase):
    def _runtime(self) -> dict[str, object]:
        return {
            "judge": {
                "python": "/runtime/python",
                "nvcc": "/cuda/nvcc",
                "cuobjdump": "/cuda/cuobjdump",
            }
        }

    def _candidate(self, root: Path, name: str, *, arm: str = "open_cake") -> Path:
        candidate = root / name
        candidate.mkdir()
        (candidate / "candidate.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "arm": arm,
                    "nodes": [],
                }
            ),
            encoding="utf-8",
        )
        return candidate

    def test_external_open_cake_batch_preserves_order_and_skips_seed_materialization(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._candidate(root, "first")
            second = self._candidate(root, "second")
            with patch(
                "open_cake_ir.tasks.qsa.launch._materialize_candidates"
            ) as materialize:
                selected = _select_candidates(
                    root / "run",
                    SimpleNamespace(),
                    arm="open_cake",
                    external_open_cake=[second, first],
                )

        self.assertEqual(selected, (second.resolve(), first.resolve()))
        materialize.assert_not_called()

    def test_external_open_cake_candidates_require_matching_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self._candidate(Path(directory), "candidate")
            with self.assertRaisesRegex(ValueError, "require --arm open_cake"):
                _external_open_cake_candidates([candidate], arm="both")

    def test_external_open_cake_candidates_reject_descriptor_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            candidate = self._candidate(
                Path(directory), "direct-candidate", arm="direct_cuda"
            )
            with self.assertRaisesRegex(ValueError, "descriptor differs"):
                _external_open_cake_candidates([candidate], arm="open_cake")

    def test_external_open_cake_candidates_reject_duplicates_and_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = self._candidate(root, "candidate")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _external_open_cake_candidates(
                    [candidate, candidate], arm="open_cake"
                )
            alias = root / "alias"
            alias.symlink_to(candidate, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _external_open_cake_candidates([alias], arm="open_cake")

    def test_default_candidate_selection_and_label_prefix_remain_compatible(
        self,
    ) -> None:
        cake = Path("/seed/open-cake")
        direct = Path("/seed/direct")
        with patch(
            "open_cake_ir.tasks.qsa.launch._materialize_candidates",
            return_value=(cake, direct),
        ) as materialize:
            selected = _select_candidates(
                Path("/run"),
                SimpleNamespace(),
                arm="both",
                external_open_cake=[],
            )
        arguments = build_parser().parse_args(
            [
                "--run-root",
                "/run",
                "--remote-project-root",
                "/remote/project",
                "--remote-state-root",
                "/remote/state",
                "--remote-socket",
                "/tmp/qsa.sock",
                "--remote-inbox",
                "/remote/inbox",
            ]
        )

        self.assertEqual(selected, (cake, direct))
        self.assertEqual(arguments.label_prefix, "qsa-seed-")
        materialize.assert_called_once()

    def test_parser_accepts_repeated_candidates_and_custom_label_prefix(self) -> None:
        arguments = build_parser().parse_args(
            [
                "--run-root",
                "/run",
                "--remote-project-root",
                "/remote/project",
                "--remote-state-root",
                "/remote/state",
                "--remote-socket",
                "/tmp/qsa.sock",
                "--remote-inbox",
                "/remote/inbox",
                "--arm",
                "open_cake",
                "--open-cake-candidate",
                "/candidate/one",
                "--open-cake-candidate",
                "/candidate/two",
                "--label-prefix",
                "qsa-r13-",
            ]
        )

        self.assertEqual(
            arguments.open_cake_candidate,
            [Path("/candidate/one"), Path("/candidate/two")],
        )
        self.assertEqual(arguments.label_prefix, "qsa-r13-")

    def test_component_timing_has_a_distinct_task_and_explicit_evaluator_flag(self) -> None:
        task = _task(
            remote_root="/remote/open-cake",
            runtime=self._runtime(),
            executor=SimpleNamespace(executor_id="executor-v1", canonical_sha256="a" * 64),
            protocol="seed",
            component_timing=True,
            profile_kernel="score_topk",
        )

        self.assertEqual(
            task["task_id"], "open-cake-qsa-prefill-t32768-seed-component-v1"
        )
        for stage in task["stages"]:
            self.assertIn("--component-timing", stage["judge"]["command"])

    def test_component_timing_cannot_masquerade_as_a_profile_task(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires the seed protocol"):
            _task(
                remote_root="/remote/open-cake",
                runtime=self._runtime(),
                executor=SimpleNamespace(
                    executor_id="executor-v1", canonical_sha256="a" * 64
                ),
                protocol="profile",
                component_timing=True,
                profile_kernel="score_topk",
            )

    def test_attention_profile_is_explicit_in_task_identity_and_command(self) -> None:
        task = _task(
            remote_root="/remote/open-cake",
            runtime=self._runtime(),
            executor=SimpleNamespace(executor_id="executor-v1", canonical_sha256="a" * 64),
            protocol="profile",
            component_timing=False,
            profile_kernel="attention",
        )

        self.assertEqual(
            task["task_id"],
            "open-cake-qsa-prefill-t32768-profile-attention-v1",
        )
        for stage in task["stages"]:
            self.assertEqual(
                stage["judge"]["command"][-2:],
                ["--profile-kernel", "attention"],
            )

    def test_nondefault_profile_target_requires_the_profile_protocol(self) -> None:
        with self.assertRaisesRegex(ValueError, "selection requires the profile protocol"):
            _task(
                remote_root="/remote/open-cake",
                runtime=self._runtime(),
                executor=SimpleNamespace(
                    executor_id="executor-v1", canonical_sha256="a" * 64
                ),
                protocol="seed",
                component_timing=False,
                profile_kernel="attention",
            )


if __name__ == "__main__":
    unittest.main()
