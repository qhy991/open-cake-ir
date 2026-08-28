from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from evaluate_qsa_candidate import _component_timing  # noqa: E402


@dataclass(frozen=True)
class _Kernel:
    kernel_id: str


@dataclass(frozen=True)
class _Artifact:
    kernels: tuple[_Kernel, ...]


class _Program:
    def __init__(self, *kernel_ids: str) -> None:
        self.artifact = _Artifact(tuple(_Kernel(value) for value in kernel_ids))

    def launch(self, _tensors, *, stream, boundary=None) -> None:
        del stream
        if boundary is not None:
            for kernel in self.artifact.kernels:
                boundary(kernel.kernel_id, "before")
                boundary(kernel.kernel_id, "after")


class _Event:
    def __init__(self, cuda) -> None:
        self.cuda = cuda
        self.at = 0.0

    def record(self) -> None:
        self.at = self.cuda.clock
        self.cuda.clock += 1.0

    def elapsed_time(self, stop) -> float:
        return stop.at - self.at


class _Cuda:
    def __init__(self) -> None:
        self.clock = 0.0

    def Event(self, *, enable_timing: bool):
        if not enable_timing:
            raise AssertionError("component events must enable timing")
        return _Event(self)

    def synchronize(self) -> None:
        return None


class _Torch:
    def __init__(self) -> None:
        self.cuda = _Cuda()


class ComponentTimingContractTest(unittest.TestCase):
    def test_balanced_trace_retains_raw_node_samples_and_bounded_summary(self) -> None:
        observed = _component_timing(
            _Program("pool", "score_topk"),
            _Program("pool_layernorm", "score_topk", "attention"),
            {},
            {},
            stream=0,
            torch=_Torch(),
        )

        self.assertTrue(observed["protocol"]["diagnostic_only"])
        self.assertFalse(observed["protocol"]["synchronization_between_nodes"])
        self.assertEqual(
            observed["programs"]["candidate"]["launch_order"],
            ["pool", "score_topk"],
        )
        self.assertEqual(
            len(observed["programs"]["candidate"]["whole_samples_ms"]),
            25,
        )
        self.assertEqual(
            len(
                observed["programs"]["baseline"]["kernel_samples_ms"]["attention"]
            ),
            25,
        )
        fractions = [
            row["fraction_of_summed_kernel_medians"]
            for row in observed["programs"]["candidate"]["summary"]["kernels"].values()
        ]
        self.assertAlmostEqual(sum(fractions), 1.0)


if __name__ == "__main__":
    unittest.main()
