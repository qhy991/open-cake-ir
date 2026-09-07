"""Performance package boundaries and single-input fact reuse."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from open_cake_ir import compiler
from open_cake_ir.compiler import performance
from open_cake_ir.compiler.performance import profile, residency
from open_cake_ir.compiler.ir import Schedule
from open_cake_ir.compiler.target import Target

ROOT = Path(__file__).resolve().parents[2]


class PerformanceArchitectureTests(unittest.TestCase):
    def test_supported_public_results_keep_one_definition_without_old_module_shims(self):
        for name in ("CompiledResources", "EmpiricalCostModel", "MetricEstimate", "ProfileEnvelope", "profile_envelope"):
            self.assertIs(getattr(compiler, name), getattr(performance, name))
        for module in ("analysis", "work", "profile_model", "compiled_resources", "empirical_cost", "ranking", "utilization"):
            with self.subTest(module=module):
                self.assertIsNone(importlib.util.find_spec("open_cake_ir.compiler." + module))

    def test_each_profile_reuses_residency_pressure_and_each_top_k_structure(self):
        schedule = Schedule.load(ROOT / "corpus/schedules/qsa-score-topk-t32768.json")
        target = Target.load(ROOT / "compiler/targets/sm_100a.json")
        with (
            patch.object(profile, "residency_upper_bound", wraps=profile.residency_upper_bound) as bounds,
            patch.object(profile, "logical_register_pressure_per_thread", wraps=profile.logical_register_pressure_per_thread) as pressure,
            patch.object(profile, "_top_k_features", wraps=profile._top_k_features) as features,
            patch.object(profile, "top_k_merge_structure", wraps=profile.top_k_merge_structure) as merge,
            patch.object(profile, "_runtime_indexed_buffers", wraps=profile._runtime_indexed_buffers) as indexed,
            patch.object(residency, "top_k_merge_structure", side_effect=AssertionError("merge structure recomputed")),
        ):
            observed = profile.profile_envelope(schedule, target).as_dict()
        for called in (bounds, pressure, features, merge, indexed):
            self.assertEqual(called.call_count, 1)
        self.assertEqual(observed["residency"]["logical_register_pressure_per_thread"], 72)
        self.assertEqual(observed["residency"]["ctas_per_sm_upper_bound"], 8)

    def test_target_without_occupancy_keeps_explicit_shared_bytes_in_reasons(self):
        document = json.loads((ROOT / "corpus/schedules/fma-b8-smoke.json").read_text())
        document["target"] = "sm_103a"
        document["allocations"] = [{"name": "shared", "space": "shared", "size_bytes": 1024}]
        target = Target.load(ROOT / "compiler/targets/sm_103a.json")
        self.assertIsNone(target.occupancy)
        observed = profile.profile_envelope(Schedule.from_dict(document), target).as_dict()
        self.assertIsNone(observed["residency"])
        shared = next(metric for metric in observed["ncu_metrics"] if metric["metric"] == "launch__occupancy_limit_shared_mem")
        self.assertIsNone(shared["value"])
        self.assertEqual(shared["estimate_kind"], "unknown")
        self.assertEqual(shared["reasons"], ["1024 shared bytes/CTA"])
