"""The batched Metal paired assay: amortized samples and an outlier-robust gate.

No GPU runs here. The samples are explicit fixtures chosen to separate the two
dispersion statistics; they make no claim about any device.
"""
from __future__ import annotations

import copy
import unittest
from pathlib import Path

from open_cake_ir.evaluation.metal_observations import (
    amortized_dispatch_ms, command_buffer_ms, dispatch_count, validate_command_samples,
)
from open_cake_ir.evaluation.paired import (
    PAIRED_METAL_BATCHED_KIND, PAIRED_METAL_KIND, paired_protocol,
)
from open_cake_ir.evaluation.timing import (
    PairedTimingProtocol, derive_paired_timing, relative_iqr, summarize_cohort,
)
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.normalization.study import evaluation_policy
from open_cake_ir.tasks.workloads import create_task

PAIR_ORDER = [["candidate", "baseline"], ["baseline", "candidate"]] * 5


def policy(**changes):
    document, _ = create_task("rmsnorm", backend="metal-m2", rows=2, columns=7)
    value = evaluation_policy(WorkloadContract(document), **changes)
    return value


def command(index, *, timed, dispatches, seconds):
    return {"launch_index": index, "completed": True, "timed": timed,
            "dispatches": dispatches, "gpu_start_seconds": 10.0,
            "gpu_end_seconds": 10.0 + seconds}


class BatchedMetalAssayTests(unittest.TestCase):
    def test_successor_declares_batching_and_a_robust_gate_while_v1_forbids_both(self):
        value = policy()
        timing = value["paired_timing"]
        self.assertEqual(timing["kind"], PAIRED_METAL_BATCHED_KIND)
        protocol = paired_protocol(value)
        self.assertEqual(protocol.dispatches_per_sample, timing["dispatches_per_sample"])
        self.assertEqual(protocol.maximum_relative_iqr, timing["maximum_relative_iqr"])
        for field in ("dispatches_per_sample", "maximum_relative_iqr"):
            missing = copy.deepcopy(value)
            del missing["paired_timing"][field]
            with self.subTest(missing=field), self.assertRaises(ValueError):
                paired_protocol(missing)
        # The original assay keeps its exact single-dispatch, CV-gated meaning.
        original = copy.deepcopy(value)
        original["paired_timing"]["kind"] = PAIRED_METAL_KIND
        with self.assertRaises(ValueError):
            paired_protocol(original)
        for field in ("dispatches_per_sample", "maximum_relative_iqr"):
            del original["paired_timing"][field]
        retained = paired_protocol(original)
        self.assertEqual(retained.dispatches_per_sample, 1)
        self.assertIsNone(retained.maximum_relative_iqr)

    def test_declared_dispatch_count_is_required_and_bounded(self):
        for bad in (0, -1, 4097, 1.0, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                policy(dispatches_per_sample=bad)
        self.assertEqual(policy(dispatches_per_sample=1)["paired_timing"]["dispatches_per_sample"], 1)

    def test_amortization_divides_one_buffer_by_the_dispatches_it_encoded(self):
        buffer = command(0, timed=True, dispatches=64, seconds=0.000640)
        self.assertAlmostEqual(command_buffer_ms(buffer), 0.640)
        self.assertAlmostEqual(amortized_dispatch_ms(buffer), 0.010)
        # Evidence sealed before the successor carries no count and stays one dispatch.
        original = {k: v for k, v in buffer.items() if k != "dispatches"}
        self.assertEqual(dispatch_count(original), 1)
        self.assertAlmostEqual(amortized_dispatch_ms(original), command_buffer_ms(original))
        for bad in (0, 4097, 1.5, True):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                dispatch_count({**buffer, "dispatches": bad})

    def test_samples_must_match_the_declared_dispatch_count_and_amortized_times(self):
        commands = [command(i, timed=i >= 1, dispatches=8, seconds=0.000080) for i in range(4)]
        samples = [0.010, 0.010, 0.010]
        record = {"command_buffers": commands, "samples_ms": samples}
        validate_command_samples(record, route_calls=4, sample_count=3, dispatches_per_sample=8)
        with self.assertRaises(ValueError):
            validate_command_samples(record, route_calls=4, sample_count=3, dispatches_per_sample=16)
        # Unamortized samples no longer describe the raw buffers they came from.
        with self.assertRaises(ValueError):
            validate_command_samples({**record, "samples_ms": [0.080] * 3},
                                     route_calls=4, sample_count=3, dispatches_per_sample=8)

    def test_every_declared_paired_kind_reaches_the_paired_receipt_check(self):
        """A successor kind that is not routed here falls through to the unpaired branch.

        That is exactly how `fixed_baseline_paired_metal_v2` first reached a live
        Campaign and failed at turn 5 with "EvaluationReceipt timing samples differ":
        one module named the versions again instead of using the assay's vocabulary.
        """
        from open_cake_ir.evaluation import core
        from open_cake_ir.evaluation.paired import PAIRED_KINDS
        source = Path(core.__file__).read_text(encoding="utf-8")
        for kind in PAIRED_KINDS:
            with self.subTest(kind=kind):
                # The receipt must not carry its own copy of the vocabulary.
                self.assertNotIn(f"'{kind}'", source)
                self.assertNotIn(f'"{kind}"', source)
        self.assertIn("in PAIRED_KINDS", source)
        self.assertIn(PAIRED_METAL_BATCHED_KIND, PAIRED_KINDS)

    def test_relative_iqr_describes_the_bulk_where_cv_reports_one_disturbance(self):
        steady = [10.00 + 0.01 * (index % 3) for index in range(24)]
        disturbed = steady + [28.0]
        self.assertLess(relative_iqr(disturbed), 0.05)
        self.assertGreater(summarize_cohort(disturbed)["cv"], 0.05)
        # A genuinely wide cohort fails the robust gate too; it is not a blanket pass.
        wide = [10.0 + index * 0.5 for index in range(25)]
        self.assertGreater(relative_iqr(wide), 0.05)
        with self.assertRaises(ValueError):
            relative_iqr([10.0])

    def test_the_declared_gate_decides_measurement_quality(self):
        steady = [10.00 + 0.01 * (index % 3) for index in range(24)]
        cohorts = {"candidate": steady + [28.0], "baseline": [value * 1.10 for value in steady + [28.0]]}
        measurements = []
        for index, order in enumerate(PAIR_ORDER):
            row = {"pair_index": index, "order": list(order), "arms": {}}
            for position, arm in enumerate(order):
                row["arms"][arm] = {"position": position, "samples_ms": cohorts[arm],
                                    "summary": summarize_cohort(cohorts[arm]), "route_calls": 28}
            measurements.append(row)
        common = (("candidate", "baseline"), tuple(tuple(p) for p in PAIR_ORDER), 25, 28, 0.05, 1.05, 6)
        by_cv = derive_paired_timing(measurements, PairedTimingProtocol(*common, 64, None))
        by_iqr = derive_paired_timing(measurements, PairedTimingProtocol(*common, 64, 0.05))
        self.assertFalse(by_cv.measurement_quality_passed)
        self.assertEqual(by_cv.classification, "measurement_quality_failed")
        self.assertTrue(by_iqr.measurement_quality_passed)
        # The directional decision itself is untouched by which gate is declared.
        self.assertEqual(by_cv.pair_wins, by_iqr.pair_wins)
        self.assertEqual(by_cv.speedup, by_iqr.speedup)
        self.assertEqual(by_iqr.classification, "first_arm_faster")

    def test_robust_gate_bound_is_validated(self):
        common = (("candidate", "baseline"), tuple(tuple(p) for p in PAIR_ORDER), 25, 28, 0.05, 1.05, 6)
        for bad in (0.0, 1.0, -0.1, True, "0.05"):
            with self.subTest(value=bad), self.assertRaises(ValueError):
                PairedTimingProtocol(*common, 64, bad)


if __name__ == "__main__":
    unittest.main()
