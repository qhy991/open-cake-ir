from __future__ import annotations

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation.amd_rmsnorm_search import (  # noqa: E402
    INCONCLUSIVE_MEASUREMENT_QUALITY,
    LEAF_TIMING_WIN,
    NUM_WARPS,
    ROUTE_BACKEND,
    ROW_TILES,
    STOP_BASELINE_FASTER,
    STOP_CLOSE_NULL,
    AmdRmsNormSearchContract,
    candidate_id,
    derive_confirmatory_decision,
    materialize_candidates,
)
from open_cake_ir.evaluation.timing import summarize_cohort  # noqa: E402


TEMPLATE = ROOT / "corpus/schedules/llama-rmsnorm-mul-b8-gfx1151-r1-w8.json"
WORKLOAD = ROOT / "contracts/workloads/llama-rmsnorm-mul-fp32-v2.json"
RUNNER = ROOT / "examples/gpu/rmsnorm_amd_search.py"
SEARCH_CONTRACT = (
    ROOT / "contracts/calibrations/llama-rmsnorm-mul-gfx1151-one-row-search-v2.json"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


class ContractFixture:
    def __init__(self, testcase: unittest.TestCase) -> None:
        temporary = tempfile.TemporaryDirectory()
        testcase.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compiler = {
            "schema_version": 1,
            "revision_id": "open-cake-ir-test-v29",
            "state": "released",
        }
        self.executor = {
            "schema_version": 1,
            "executor_id": "open-cake-ir-test-v30",
            "state": "released",
        }
        self.workload = json.loads(WORKLOAD.read_text(encoding="utf-8"))
        self.template = json.loads(TEMPLATE.read_text(encoding="utf-8"))
        _write_json(self.root / "compiler.json", self.compiler)
        _write_json(self.root / "executor.json", self.executor)
        _write_json(self.root / "workload.json", self.workload)
        _write_json(self.root / "template.json", self.template)
        self.document = {
            "schema_version": 1,
            "search_id": "gfx1151-llama-rmsnorm-mul-one-row-search-v2",
            "state": "frozen",
            "target": "gfx1151",
            "route": {"backend": ROUTE_BACKEND},
            "compiler": {
                "path": "compiler.json",
                "revision_id": self.compiler["revision_id"],
                "canonical_sha256": _canonical_sha256(self.compiler),
            },
            "executor": {
                "path": "executor.json",
                "executor_id": self.executor["executor_id"],
                "canonical_sha256": _canonical_sha256(self.executor),
            },
            "template": {
                "path": "template.json",
                "canonical_sha256": _canonical_sha256(self.template),
            },
            "workload": {
                "path": "workload.json",
                "canonical_sha256": _canonical_sha256(self.workload),
            },
            "geometry": {
                "row_tiles": list(ROW_TILES),
                "num_warps": list(NUM_WARPS),
                "candidate_count": len(ROW_TILES) * len(NUM_WARPS),
                "baseline_geometry": {"row_tile": 64, "num_warps": 4},
                "prior_screening_winner": {
                    "row_tile": 8,
                    "num_warps": 8,
                    "disposition": "STOP_CLOSE_NULL",
                },
            },
            "screening": {
                "case_id": "seeded_random",
                "rounds_per_candidate": 5,
                "warmup_launches": 5,
                "samples_per_round": 10,
                "launches_per_sample": 50,
                "l2_flush_bytes": 268435456,
                "maximum_cv": 0.05,
            },
            "confirmatory": {
                "arms": ["candidate", "baseline"],
                "pair_order": [
                    ["candidate", "baseline"],
                    ["baseline", "candidate"],
                    ["baseline", "candidate"],
                    ["candidate", "baseline"],
                ],
                "samples_per_cohort": 25,
                "warmup_launches_per_cohort": 5,
                "launches_per_sample": 50,
                "route_calls_per_cohort": 1255,
                "maximum_cv": 0.05,
                "materiality_ratio": 1.05,
                "required_pair_wins": 3,
            },
        }
        self.path = self.root / "search.json"
        self.write()

    def write(self) -> None:
        _write_json(self.path, self.document)

    def load(self) -> AmdRmsNormSearchContract:
        return AmdRmsNormSearchContract.load(self.root, self.path)


def _measurements(
    contract: AmdRmsNormSearchContract,
    candidate_samples: list[float],
    baseline_samples: list[float],
) -> list[dict[str, object]]:
    protocol = contract.confirmatory.timing
    result: list[dict[str, object]] = []
    for pair_index, order in enumerate(protocol.pair_order):
        arms: dict[str, object] = {}
        for position, arm in enumerate(order):
            samples = candidate_samples if arm == "candidate" else baseline_samples
            arms[arm] = {
                "position": position,
                "samples_ms": list(samples),
                "summary": summarize_cohort(samples),
                "route_calls": protocol.route_calls_per_cohort,
            }
        result.append(
            {
                "pair_index": pair_index,
                "order": list(order),
                "arms": arms,
            }
        )
    return result


class AmdRmsNormSearchContractTests(unittest.TestCase):
    @unittest.skipUnless(
        SEARCH_CONTRACT.exists(),
        "requires externally approved Compiler v29 search contract",
    )
    def test_released_contract_prepares_the_complete_domain_without_a_gpu(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--project-root",
                str(ROOT),
                "--contract",
                str(SEARCH_CONTRACT),
                "--prepare-only",
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "prepared")
        self.assertEqual(result["candidate_count"], 4)
        self.assertEqual(result["compiler"]["revision_id"], "open-cake-ir-v29")
        self.assertEqual(result["executor"]["executor_id"], "open-cake-ir-b200-v30")
        self.assertFalse(result["gpu_submitted"])

    @unittest.skipUnless(
        SEARCH_CONTRACT.exists(),
        "requires externally approved Compiler v29 search contract",
    )
    def test_runner_refuses_an_evidence_directory_inside_the_checkout(self) -> None:
        forbidden = ROOT / "amd-rmsnorm-search-evidence-forbidden"
        self.assertFalse(forbidden.exists())
        completed = subprocess.run(
            [
                sys.executable,
                str(RUNNER),
                "--project-root",
                str(ROOT),
                "--contract",
                str(SEARCH_CONTRACT),
                "--prepare-only",
                "--artifact-dir",
                str(forbidden),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("outside the checkout", completed.stderr)
        self.assertFalse(forbidden.exists())

    def test_loads_exact_frozen_contract_and_owned_json(self) -> None:
        fixture = ContractFixture(self)

        contract = fixture.load()

        self.assertEqual(contract.search_id, fixture.document["search_id"])
        self.assertEqual(contract.compiler_revision_id, "open-cake-ir-test-v29")
        self.assertEqual(contract.executor_id, "open-cake-ir-test-v30")
        self.assertEqual(contract.screening.case_id, "seeded_random")
        self.assertEqual(contract.screening.launches_per_sample, 50)
        self.assertEqual(contract.confirmatory.launches_per_sample, 50)
        self.assertEqual(
            contract.confirmatory.timing.route_calls_per_cohort,
            5 + 25 * 50,
        )
        self.assertEqual(
            contract.canonical_sha256,
            _canonical_sha256(fixture.document),
        )

    def test_contract_tampering_fails_closed(self) -> None:
        changes = {
            "draft": lambda fixture: fixture.document.__setitem__("state", "draft"),
            "extra_field": lambda fixture: fixture.document.__setitem__("extra", True),
            "geometry": lambda fixture: fixture.document["geometry"][
                "row_tiles"
            ].append(128),
            "route_count": lambda fixture: fixture.document["confirmatory"].__setitem__(
                "route_calls_per_cohort", 30
            ),
            "screening": lambda fixture: fixture.document["screening"].__setitem__(
                "rounds_per_candidate", 6
            ),
            "compiler_identity": lambda fixture: fixture.document["compiler"].__setitem__(
                "revision_id", "changed"
            ),
        }
        for label, mutate in changes.items():
            with self.subTest(label=label):
                fixture = ContractFixture(self)
                mutate(fixture)
                fixture.write()
                with self.assertRaises(ValueError):
                    fixture.load()

        fixture = ContractFixture(self)
        fixture.compiler["new_fact"] = True
        _write_json(fixture.root / "compiler.json", fixture.compiler)
        with self.assertRaisesRegex(ValueError, "canonical SHA-256"):
            fixture.load()

    def test_template_workload_binding_is_checked(self) -> None:
        fixture = ContractFixture(self)
        fixture.template["metadata"]["workload_contract_sha256"] = "0" * 64
        _write_json(fixture.root / "template.json", fixture.template)
        fixture.document["template"]["canonical_sha256"] = _canonical_sha256(
            fixture.template
        )
        fixture.write()

        with self.assertRaisesRegex(ValueError, "not bound"):
            fixture.load()


class AmdRmsNormCandidateTests(unittest.TestCase):
    def test_materializes_complete_domain_without_mutating_template(self) -> None:
        fixture = ContractFixture(self)
        contract = fixture.load()
        original = contract.template_path.read_bytes()

        candidates = materialize_candidates(contract)

        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            [item.candidate_id for item in candidates],
            [candidate_id(tile, warps) for tile in ROW_TILES for warps in NUM_WARPS],
        )
        self.assertEqual(len({item.canonical_sha256 for item in candidates}), 4)
        self.assertEqual(contract.template_path.read_bytes(), original)
        for candidate in candidates:
            with self.subTest(candidate=candidate.candidate_id):
                schedule = candidate.schedule
                self.assertEqual(schedule.target, "gfx1151")
                self.assertEqual(schedule.lowering.backend.value, ROUTE_BACKEND)
                self.assertEqual(schedule.roles[0].warps, tuple(range(candidate.num_warps)))
                self.assertEqual(schedule.program_map.axes[0].tile, candidate.row_tile)
                shapes = {buffer.name: buffer.shape for buffer in schedule.buffers}
                for name in ("x_tile", "sq", "normed", "y_tile"):
                    self.assertEqual(shapes[name], (candidate.row_tile, 128))
                for name in ("sumsq", "meansq", "shifted", "inv_rms"):
                    self.assertEqual(shapes[name], (candidate.row_tile,))

    def test_every_candidate_assesses_and_lowers_through_current_compiler(self) -> None:
        fixture = ContractFixture(self)
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.json")

        for candidate in materialize_candidates(fixture.load()):
            with self.subTest(candidate=candidate.candidate_id):
                assessment = compiler.assess(candidate.document)
                self.assertTrue(assessment.accepted, assessment.findings)
                self.assertTrue(assessment.lowering_eligible, assessment.findings)
                lowering = compiler.lower(assessment)
                self.assertEqual(lowering.route.backend.value, ROUTE_BACKEND)
                self.assertEqual(
                    lowering.toolchain_requirements["compile_options"],
                    {"num_warps": candidate.num_warps},
                )
                self.assertEqual(
                    lowering.toolchain_requirements["grid"],
                    [512 // candidate.row_tile, 8, 1],
                )

    def test_candidate_identity_rejects_geometry_outside_domain(self) -> None:
        for geometry in ((2, 1), (128, 4), (1, 16), (1, True)):
            with self.subTest(geometry=geometry), self.assertRaises(ValueError):
                candidate_id(*geometry)


class AmdRmsNormDecisionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.contract = ContractFixture(self).load()
        self.count = self.contract.confirmatory.timing.samples_per_cohort

    def test_maps_every_stable_direction_and_close_null(self) -> None:
        cases = (
            (0.90, 1.00, LEAF_TIMING_WIN),
            (0.99, 1.00, STOP_CLOSE_NULL),
            (1.20, 1.00, STOP_BASELINE_FASTER),
        )
        for candidate, baseline, status in cases:
            with self.subTest(status=status):
                decision = derive_confirmatory_decision(
                    self.contract,
                    _measurements(
                        self.contract,
                        [candidate] * self.count,
                        [baseline] * self.count,
                    ),
                )
                self.assertEqual(decision.status, status)
                self.assertTrue(decision.observation.measurement_quality_passed)

    def test_high_cv_is_inconclusive_not_stop(self) -> None:
        noisy = [0.5 if index % 2 == 0 else 1.5 for index in range(self.count)]
        decision = derive_confirmatory_decision(
            self.contract,
            _measurements(self.contract, noisy, [1.2] * self.count),
        )

        self.assertEqual(decision.status, INCONCLUSIVE_MEASUREMENT_QUALITY)
        self.assertFalse(decision.observation.measurement_quality_passed)

    def test_raw_measurement_tampering_is_rejected(self) -> None:
        base = _measurements(
            self.contract,
            [0.9] * self.count,
            [1.0] * self.count,
        )
        tampered = {
            "summary": lambda value: value[0]["arms"]["candidate"][
                "summary"
            ].__setitem__("median_ms", 0.1),
            "route_calls": lambda value: value[0]["arms"]["candidate"].__setitem__(
                "route_calls", 30
            ),
            "order": lambda value: value[0].__setitem__(
                "order", ["baseline", "candidate"]
            ),
            "sample_count": lambda value: value[0]["arms"]["candidate"][
                "samples_ms"
            ].pop(),
        }
        for label, mutate in tampered.items():
            with self.subTest(label=label):
                value = copy.deepcopy(base)
                mutate(value)
                with self.assertRaises(ValueError):
                    derive_confirmatory_decision(self.contract, value)


if __name__ == "__main__":
    unittest.main()
