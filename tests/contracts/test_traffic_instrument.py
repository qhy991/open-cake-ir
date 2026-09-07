"""Contract tests for the traffic half of the profiling instrument.

The residency half of `tools/profile_lowered_kernel.py` can only be exercised on a B200.
The two things that decide whether its traffic numbers mean anything cannot: reading
Nsight's units correctly, and dividing the right two byte counts. Both are pure, so both
are tested here rather than discovered on a device that has to be booked.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from open_cake_ir.compiler.ir import Schedule  # noqa: E402
from open_cake_ir.compiler.performance.work import work_bound  # noqa: E402
from profile_lowered_kernel import _LIMITS, _measured, traffic_comparison  # noqa: E402

GEMM = ROOT / "corpus" / "schedules" / "gemm-bias-b1-smoke.json"
GATHER = ROOT / "corpus" / "schedules" / "indexed-gather-b8-smoke.json"

_HEADER = '"ID","Kernel Name","Metric Name","Metric Unit","Metric Value"'


def _csv(rows: dict[str, tuple[str, str]]) -> str:
    lines = [_HEADER]
    for index, (metric, (unit, value)) in enumerate(rows.items()):
        lines.append(f'"{index}","cake_kernel","{metric}","{unit}","{value}"')
    return "\n".join(lines)


def _complete(**overrides: tuple[str, str]) -> dict[str, tuple[str, str]]:
    rows = {
        "launch__registers_per_thread": ("register", "81"),
        "launch__occupancy_limit_registers": ("block", "5"),
        "launch__occupancy_limit_shared_mem": ("block", "5"),
        "launch__occupancy_limit_blocks": ("block", "32"),
        "launch__occupancy_limit_warps": ("block", "16"),
        "dram__bytes_read.sum": ("byte", "1048576"),
        "dram__bytes_write.sum": ("byte", "524288"),
        "lts__t_bytes.sum": ("byte", "4194304"),
        "gpu__time_duration.sum": ("second", "0.000012"),
    }
    rows.update(overrides)
    return rows


def _bound(path: Path):
    bound = work_bound(Schedule.load(path))
    assert bound is not None
    return bound


class UnitTest(unittest.TestCase):
    """Nsight rescales for display, and a rescaled byte count is not a byte count."""

    def test_schema_two_names_the_measured_physical_register_limit(self) -> None:
        values = _measured(_csv(_complete()))
        limits = {name: values[metric] for metric, name in _LIMITS.items()}
        self.assertEqual(limits["registers"], 5.0)
        self.assertNotIn("logical_register_storage", limits)

    def test_base_units_are_read_as_written(self) -> None:
        values = _measured(_csv(_complete()))
        self.assertEqual(values["dram__bytes_read.sum"], 1048576.0)
        self.assertEqual(values["gpu__time_duration.sum"], 0.000012)

    def test_a_rescaled_byte_count_is_refused_rather_than_converted(self) -> None:
        """`--print-units base` should prevent this; the reader does not trust that it did.

        A "Mbyte" row carries a number a million times smaller than the one every
        quotient below assumes. Converting it here would mean guessing whether Nsight's
        prefix is decimal or binary, and the stored record would look identical either
        way -- so the run stops instead.
        """

        with self.assertRaises(SystemExit) as refusal:
            _measured(_csv(_complete(**{"dram__bytes_read.sum": ("Mbyte", "1.05")})))
        self.assertIn("Mbyte", str(refusal.exception))

    def test_a_rescaled_duration_is_refused_too(self) -> None:
        with self.assertRaises(SystemExit):
            _measured(_csv(_complete(**{"gpu__time_duration.sum": ("usecond", "12")})))

    def test_an_absent_metric_still_stops_the_run(self) -> None:
        rows = _complete()
        del rows["lts__t_bytes.sum"]
        with self.assertRaises(SystemExit) as refusal:
            _measured(_csv(rows))
        self.assertIn("lts__t_bytes.sum", str(refusal.exception))


class AmplificationTest(unittest.TestCase):
    def test_amplification_divides_measured_traffic_by_the_declared_bytes(self) -> None:
        """918,528 declared bytes against 1.5MB of DRAM and 4MB of L2 traffic."""

        comparison = traffic_comparison(_bound(GEMM), _measured(_csv(_complete())))
        self.assertEqual(comparison["compulsory_bytes"], 918528)
        self.assertEqual(comparison["dram_bytes"], 1572864.0)
        self.assertAlmostEqual(comparison["dram_amplification"], 1572864 / 918528)
        self.assertAlmostEqual(comparison["l2_amplification"], 4194304 / 918528)

    def test_the_achieved_rate_is_measured_over_measured(self) -> None:
        """Bytes this launch moved over the seconds it took, and no declared peak."""

        comparison = traffic_comparison(_bound(GEMM), _measured(_csv(_complete())))
        self.assertAlmostEqual(
            comparison["achieved_dram_bytes_per_second"], 1572864 / 0.000012
        )

    def test_dram_below_the_declared_count_is_recorded_and_not_judged(self) -> None:
        """A working set this small stays in L2, so DRAM sees less than must move.

        The Schedule did not move fewer bytes than it declared; the cache absorbed them.
        Treating that as a refutation of the work model would refute it on every Corpus
        Schedule, which is the reading this flag exists to prevent.
        """

        values = _measured(
            _csv(
                _complete(
                    **{
                        "dram__bytes_read.sum": ("byte", "16384"),
                        "dram__bytes_write.sum": ("byte", "16384"),
                    }
                )
            )
        )
        comparison = traffic_comparison(_bound(GEMM), values)
        self.assertTrue(comparison["dram_below_compulsory"])
        self.assertLess(comparison["dram_amplification"], 1)

    def test_an_inexact_byte_count_travels_with_its_quotient(self) -> None:
        """A gather's compulsory count is an upper bound, so its amplification is a floor."""

        comparison = traffic_comparison(_bound(GATHER), _measured(_csv(_complete())))
        self.assertFalse(comparison["compulsory_bytes_exact"])
        exact = traffic_comparison(_bound(GEMM), _measured(_csv(_complete())))
        self.assertTrue(exact["compulsory_bytes_exact"])


class NoDeclaredPeakTest(unittest.TestCase):
    def test_the_comparison_names_no_peak_bandwidth(self) -> None:
        """No Target declares one, so a utilisation here would be the invented number."""

        comparison = traffic_comparison(_bound(GEMM), _measured(_csv(_complete())))
        self.assertFalse(
            {key for key in comparison if "peak" in key or "utilization" in key}
        )


if __name__ == "__main__":
    unittest.main()
