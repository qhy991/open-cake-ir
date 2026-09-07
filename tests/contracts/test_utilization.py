"""Contract tests for declared peaks and the utilisations derived from them.

This is the only place in the Compiler where a measured number is an input, so it is the
only place a utilisation can be produced -- and the only place one can be invented. What
these tests pin is that it cannot be: without a declared peak there is no ratio, without a
declared rate for the instruction actually issued there is no arithmetic ratio, and a
ratio above one is reported as a refutation rather than as a good result.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import PeakSource, Target, TargetParseError
from open_cake_ir.compiler.performance.utilization import utilization
from open_cake_ir.compiler.performance.work import work_bound

ROOT = Path(__file__).resolve().parents[2]
TARGET_PATH = ROOT / "compiler" / "targets" / "sm_100a.json"
SCHEDULES = ROOT / "corpus" / "schedules"
GEMM = SCHEDULES / "gemm-bias-b1-smoke.json"
RMSNORM = SCHEDULES / "rmsnorm-b8-smoke.json"
BLOCK_SCALED = SCHEDULES / "block-scaled-gemm-b1-smoke.json"
QSA = SCHEDULES / "qsa-score-topk-t32768.json"
GATHER = SCHEDULES / "indexed-gather-b8-smoke.json"

OBSERVED = "2026-08-27T00:00:00Z"
DOT = "triton.dot.bf16_fp32"


def _document() -> dict:
    return json.loads(TARGET_PATH.read_text(encoding="utf-8"))


def _with_peak(**block) -> Target:
    document = _document()
    document["peak"] = block
    return Target.from_dict(document)


def _rate(value: float, field: str, source: str = "microbenchmark") -> dict:
    return {field: value, "source": source, "observed_at": OBSERVED}


def _full_peak(
    flops: float = 2.0e15,
    bytes_per_second: float = 8.0e12,
    source: str = "device_specification",
) -> Target:
    return _with_peak(
        memory_bandwidth=_rate(bytes_per_second, "bytes_per_second", source),
        arithmetic={DOT: _rate(flops, "flops_per_second", source)},
    )


def _bound(path: Path):
    bound = work_bound(Schedule.load(path))
    assert bound is not None
    return bound


class ReleasedTargetTest(unittest.TestCase):
    def test_the_released_target_declares_no_peak_and_stays_admissible(self) -> None:
        """A peak decides nothing structural, so a Target without one is complete.

        The released `sm_100a` has none because nobody has measured one on the device it
        names. That is a missing measurement, not a missing feature, and every gate,
        every lowering and every residency bound is unaffected by it.
        """

        target = Target.load(TARGET_PATH)
        self.assertIsNone(target.peak)
        self.assertIsNotNone(target.occupancy)


class PeakSchemaTest(unittest.TestCase):
    def test_a_rate_carries_its_source_and_when_it_was_taken(self) -> None:
        peak = _full_peak(source="microbenchmark").peak
        assert peak is not None
        self.assertEqual(peak.for_contract(DOT).source, PeakSource.MICROBENCHMARK)
        self.assertEqual(peak.memory_bandwidth.observed_at, OBSERVED)

    def test_an_arithmetic_key_must_be_a_declared_instruction_contract(self) -> None:
        """A peak keyed by a contract the Target does not admit outlives its instruction."""

        with self.assertRaises(TargetParseError) as refusal:
            _with_peak(arithmetic={"triton.dot.fp4_whatever": _rate(1e15, "flops_per_second")})
        self.assertIn("instruction contract", str(refusal.exception))

    def test_an_unsourced_rate_is_refused(self) -> None:
        with self.assertRaises(TargetParseError):
            _with_peak(memory_bandwidth={"bytes_per_second": 8.0e12})

    def test_an_unrecognised_source_is_refused(self) -> None:
        """The two kinds mean different things, so a third spelling cannot be admitted."""

        with self.assertRaises(TargetParseError) as refusal:
            _with_peak(
                memory_bandwidth=_rate(8.0e12, "bytes_per_second", source="datasheet")
            )
        self.assertIn("source", str(refusal.exception))

    def test_a_non_positive_rate_is_refused(self) -> None:
        for value in (0, -1.0, float("inf")):
            with self.subTest(value=value):
                with self.assertRaises(TargetParseError):
                    _with_peak(memory_bandwidth=_rate(value, "bytes_per_second"))

    def test_every_spelling_of_an_empty_peak_is_refused(self) -> None:
        """Two spellings of "no peak" would be two byte strings for one Target."""

        with self.assertRaises(TargetParseError):
            _with_peak()
        with self.assertRaises(TargetParseError):
            _with_peak(arithmetic={})
        with self.assertRaises(TargetParseError):
            _with_peak(memory_bandwidth=None)
        # A null beside a real rate is the spelling the arity check cannot catch, so
        # a present key is parsed rather than read through `get`.
        with self.assertRaises(TargetParseError):
            _with_peak(
                memory_bandwidth=None,
                arithmetic={DOT: _rate(2.0e15, "flops_per_second")},
            )


class UtilizationTest(unittest.TestCase):
    def test_without_a_peak_there_is_no_ratio_at_all(self) -> None:
        self.assertIsNone(
            utilization(_bound(GEMM), Target.load(TARGET_PATH), 1e-5)
        )

    def test_the_ratios_are_work_over_rate_times_time(self) -> None:
        """67,239,936 FLOPs and 918,528 bytes, at 2 PFLOP/s and 8 TB/s, in 10 microseconds."""

        bound = _bound(GEMM)
        derived = utilization(bound, _full_peak(), 1e-5)
        assert derived is not None
        self.assertAlmostEqual(derived.arithmetic, bound.flops / (2.0e15 * 1e-5))
        self.assertAlmostEqual(
            derived.bandwidth, bound.compulsory_bytes / (8.0e12 * 1e-5)
        )
        self.assertEqual(derived.arithmetic_contract, DOT)

    def test_the_roofline_is_the_longer_of_the_two_declared_floors(self) -> None:
        """This GEMM moves 918,528 bytes and does 67 MFLOP: at these rates it is memory bound."""

        bound = _bound(GEMM)
        derived = utilization(bound, _full_peak(), 1e-5)
        assert derived is not None
        self.assertAlmostEqual(derived.roofline_seconds, bound.compulsory_bytes / 8.0e12)
        self.assertGreater(derived.bandwidth, derived.arithmetic)
        self.assertAlmostEqual(derived.roofline_efficiency, derived.bandwidth)

    def test_a_ratio_above_one_reports_as_refuted(self) -> None:
        """No kernel beats a specified ceiling, so this is a defect in an input.

        The value is still returned. Hiding it would leave a caller unable to tell a
        wrong work count from a wrong peak, and both are worth knowing about.
        """

        derived = utilization(_bound(GEMM), _full_peak(), 1e-9)
        assert derived is not None
        self.assertTrue(derived.refuted)
        self.assertGreater(derived.arithmetic, 1)

    def test_a_microbenchmark_ratio_may_exceed_one_without_becoming_a_floor(self) -> None:
        derived = utilization(
            _bound(GEMM), _full_peak(source="microbenchmark"), 1e-9
        )
        assert derived is not None

        self.assertGreater(derived.arithmetic, 1)
        self.assertGreater(derived.bandwidth, 1)
        self.assertFalse(derived.refuted)
        self.assertIsNone(derived.roofline_seconds)
        self.assertIsNone(derived.roofline_efficiency)

    def test_exact_bytes_above_a_specified_bandwidth_ceiling_are_refuted(self) -> None:
        bound = _bound(GEMM)
        target = _with_peak(
            memory_bandwidth=_rate(
                8.0e12, "bytes_per_second", "device_specification"
            )
        )
        derived = utilization(bound, target, 1e-12)
        assert derived is not None

        self.assertTrue(bound.compulsory_bytes_exact)
        self.assertGreater(derived.bandwidth, 1)
        self.assertTrue(derived.refuted)
        self.assertAlmostEqual(
            derived.roofline_seconds, bound.compulsory_bytes / 8.0e12
        )

    def test_a_sound_pair_is_not_reported_as_refuted(self) -> None:
        derived = utilization(_bound(GEMM), _full_peak(), 1e-5)
        assert derived is not None
        self.assertFalse(derived.refuted)

    def test_qsa_arithmetic_ratio_uses_exact_dynamic_stop_repetition(self) -> None:
        contract = "triton.dot.fp32_ieee"
        target = _with_peak(
            arithmetic={contract: _rate(6.5e13, "flops_per_second")}
        )
        bound = _bound(QSA)
        derived = utilization(bound, target, 0.01)
        assert derived is not None

        self.assertEqual(bound.flops, 281_303_187_456)
        self.assertEqual(bound.mma_flops, 279_122_542_592)
        self.assertAlmostEqual(
            derived.arithmetic,
            281_303_187_456 / (6.5e13 * 0.01),
        )
        self.assertEqual(derived.arithmetic_contract, contract)
        self.assertFalse(derived.arithmetic_exact)


class AbstentionTest(unittest.TestCase):
    def test_a_schedule_with_no_contraction_gets_no_arithmetic_ratio(self) -> None:
        """An rmsnorm issues none of the contracts a peak is keyed by.

        Charging it against a tensor-core rate would report two percent and describe the
        metric rather than the kernel. Its bandwidth ratio is the one that means
        something, and that one is still derived.
        """

        derived = utilization(_bound(RMSNORM), _full_peak(), 1e-5)
        assert derived is not None
        self.assertIsNone(derived.arithmetic)
        self.assertIsNone(derived.arithmetic_contract)
        self.assertIsNotNone(derived.bandwidth)

    def test_an_unmeasured_contract_gets_no_arithmetic_ratio(self) -> None:
        """The block-scaled FP8 contraction has no probe, so it has no rate and no ratio.

        This is the case the instrument leaves out on purpose. Falling back to the BF16
        rate would report an FP8 kernel against the wrong instruction's ceiling.
        """

        bound = _bound(BLOCK_SCALED)
        self.assertEqual(bound.contended_contract, "triton.dot.fp8e4m3_block_scale_fp32")
        derived = utilization(bound, _full_peak(), 1e-5)
        assert derived is not None
        self.assertIsNone(derived.arithmetic)
        self.assertIsNotNone(derived.bandwidth)

    def test_a_bandwidth_only_peak_yields_a_bandwidth_only_roofline(self) -> None:
        """A missing rate is a term nobody measured, not a floor of zero."""

        target = _with_peak(
            memory_bandwidth=_rate(
                8.0e12, "bytes_per_second", "device_specification"
            )
        )
        bound = _bound(GEMM)
        derived = utilization(bound, target, 1e-5)
        assert derived is not None
        self.assertIsNone(derived.arithmetic)
        self.assertAlmostEqual(derived.roofline_seconds, bound.compulsory_bytes / 8.0e12)

    def test_the_error_direction_of_each_half_travels_with_it(self) -> None:
        """rmsnorm's rsqrt is unpriced, so its arithmetic count is a lower bound."""

        derived = utilization(_bound(RMSNORM), _full_peak(), 1e-5)
        assert derived is not None
        self.assertFalse(derived.arithmetic_exact)
        self.assertTrue(derived.bandwidth_exact)

    def test_an_upper_bound_byte_count_is_not_a_roofline_floor_or_refutation(self) -> None:
        """A runtime gather may touch less than the full Buffer charged by work."""

        bound = _bound(GATHER)
        target = _with_peak(
            memory_bandwidth=_rate(
                1.0, "bytes_per_second", "device_specification"
            )
        )
        derived = utilization(bound, target, 1.0)
        assert derived is not None and derived.bandwidth is not None

        self.assertFalse(bound.compulsory_bytes_exact)
        self.assertGreater(derived.bandwidth, 1)
        self.assertFalse(derived.bandwidth_exact)
        self.assertIsNone(derived.roofline_seconds)
        self.assertIsNone(derived.roofline_efficiency)
        self.assertFalse(derived.refuted)


class InstrumentOutputTest(unittest.TestCase):
    """What the peak instrument emits has to be what the Target admits."""

    def test_the_emitted_block_parses_as_a_peak(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "tools"))
        from observe_target_peak import build_peak_block  # noqa: E402

        block = build_peak_block(
            observed_at=OBSERVED,
            bandwidth_bytes_per_second=7.4e12,
            arithmetic_flops_per_second={DOT: 1.6e15},
        )
        document = _document()
        document["peak"] = block
        target = Target.from_dict(document)
        assert target.peak is not None
        self.assertEqual(
            target.peak.for_contract(DOT).source, PeakSource.MICROBENCHMARK
        )

    def test_one_peak_record_can_cover_bf16_and_ieee_fp32_contracts(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "tools"))
        from observe_target_peak import BF16_DOT, FP32_DOT, build_peak_block  # noqa: E402

        block = build_peak_block(
            observed_at=OBSERVED,
            bandwidth_bytes_per_second=7.4e12,
            arithmetic_flops_per_second={
                BF16_DOT: 1.6e15,
                FP32_DOT: 7.5e13,
            },
        )
        document = _document()
        document["peak"] = block
        peak = Target.from_dict(document).peak
        assert peak is not None
        self.assertEqual(peak.for_contract(BF16_DOT).value, 1.6e15)
        self.assertEqual(peak.for_contract(FP32_DOT).value, 7.5e13)

    def test_a_block_with_no_measured_contract_still_parses(self) -> None:
        """Every probe abstaining leaves a bandwidth-only peak, which is admissible."""

        import sys

        sys.path.insert(0, str(ROOT / "tools"))
        from observe_target_peak import build_peak_block  # noqa: E402

        block = build_peak_block(
            observed_at=OBSERVED,
            bandwidth_bytes_per_second=7.4e12,
            arithmetic_flops_per_second={},
        )
        document = _document()
        document["peak"] = block
        self.assertIsNotNone(Target.from_dict(document).peak)

    def test_an_unstable_cohort_is_refused_rather_than_averaged(self) -> None:
        import sys

        sys.path.insert(0, str(ROOT / "tools"))
        from observe_target_peak import MAXIMUM_CV, derive_rate  # noqa: E402

        steady = [1.0, 1.01, 0.99, 1.0, 1.0]
        rate, summary = derive_rate(1000.0, steady)
        self.assertLessEqual(summary["cv"], MAXIMUM_CV)
        self.assertAlmostEqual(rate, 1000.0 / 0.001)
        with self.assertRaises(SystemExit):
            derive_rate(1000.0, [1.0, 4.0, 0.5, 2.0, 1.0])


if __name__ == "__main__":
    unittest.main()
