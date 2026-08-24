from __future__ import annotations

import json
import sys
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler  # noqa: E402
from open_cake_ir.evaluation import LaunchableCandidate, WorkloadContract  # noqa: E402
from open_cake_ir.lab import (  # noqa: E402
    BuildRequest,
    CandidateSubmission,
    OpenCakeEnvironment,
)


class RecordingToolchain:
    """Minimal launchable seam that exposes only accepted build requests."""

    def __init__(self) -> None:
        self.requests: list[BuildRequest] = []

    def build(self, request: BuildRequest) -> LaunchableCandidate:
        self.requests.append(request)
        return LaunchableCandidate(
            candidate_sha256=request.candidate_sha256,
            target=request.target,
            entry_point=request.entry_point,
            artifact_roles={
                "lowered_source": request.source_sha256,
                "cubin": sha256(b"fixture cubin").hexdigest(),
            },
            launch_spec_sha256=sha256(b"fixture launch specification").hexdigest(),
        )


def _headline_schedule(workload: WorkloadContract) -> dict[str, object]:
    schedule = json.loads(
        (ROOT / "corpus/schedules/flash-kmeans-b32-smoke-v2.json").read_text(
            encoding="utf-8"
        )
    )
    shape = workload.case("headline_b32")["shape"]
    buffers = {item["name"]: item for item in schedule["buffers"]}
    buffers["tokens"]["shape"] = [shape["B"], shape["N"], shape["D"]]
    buffers["centroids"]["shape"] = [shape["B"], shape["K"], shape["D"]]
    buffers["centroid_sq"]["shape"] = [shape["B"], shape["K"]]
    buffers["assignments"]["shape"] = [shape["B"], shape["N"]]
    schedule["schedule_id"] = "headline-b32-r16-authoring-contract"
    schedule["metadata"]["workload_contract_sha256"] = workload.canonical_sha256
    return schedule


class OpenCakeAuthoringEnvironmentContractTests(unittest.TestCase):
    def test_complete_r16_skeleton_for_study_workload_builds_headline_candidate(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v24.json"
            ).read_text(encoding="utf-8")
        )
        toolchain = RecordingToolchain()
        environment = OpenCakeEnvironment(
            compiler,
            toolchain,
            authority_document=study["arms"]["open_cake"],
            workload=workload,
            case_id="headline_b32",
        )
        payload = json.dumps(
            _headline_schedule(workload), sort_keys=True, separators=(",", ":")
        ).encode()

        result = environment.build(
            CandidateSubmission.seal(
                "application/vnd.open-cake.schedule+json", payload
            )
        )

        self.assertEqual(result.disposition, "launchable")
        self.assertEqual(len(toolchain.requests), 1)
        request = toolchain.requests[0]
        self.assertEqual(request.source_role, "lowered_source")
        # The constexpr is named after the program axis that walks the dimension, so a
        # name in the emitted kernel points back at the declaration that produced it.
        constants = request.toolchain_requirements["compile_constants"]
        self.assertEqual(constants["N_TOKEN_BLOCK"], 65536)
        self.assertEqual(request.toolchain_requirements["grid"], [256, 32, 1])
        self.assertEqual(
            request.source_sha256,
            result.launchable.artifact_roles["lowered_source"],
        )

    def test_built_candidate_reports_what_bounds_its_residency(self) -> None:
        # The analysis attribution is derived for every candidate but only survives on
        # the accepted path, because a bound that blocks is a rejection instead. Emitting
        # an empty findings list here would compute the attribution and then discard it.
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v24.json"
            ).read_text(encoding="utf-8")
        )
        environment = OpenCakeEnvironment(
            compiler,
            RecordingToolchain(),
            authority_document=study["arms"]["open_cake"],
            workload=workload,
            case_id="headline_b32",
        )
        payload = json.dumps(
            _headline_schedule(workload), sort_keys=True, separators=(",", ":")
        ).encode()

        result = environment.build(
            CandidateSubmission.seal(
                "application/vnd.open-cake.schedule+json", payload
            )
        )

        self.assertEqual(result.disposition, "launchable")
        self.assertEqual(result.feedback["stage"], "built")
        findings = {item["code"]: item for item in result.feedback["findings"]}
        self.assertIn("RESIDENCY_BOUND", findings)
        self.assertIn("registers", findings["RESIDENCY_BOUND"]["message"])
        # Nothing blocking can reach this path; a blocking finding here would mean the
        # Environment accepted a candidate its own verifier refused.
        self.assertFalse(
            any(
                item["blocks_acceptance"] or item["blocks_lowering"]
                for item in findings.values()
            )
        )

    def test_r25_schedule_is_rejected_by_r16_study_before_toolchain(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v24.json"
            ).read_text(encoding="utf-8")
        )
        toolchain = RecordingToolchain()
        environment = OpenCakeEnvironment(
            compiler,
            toolchain,
            authority_document=study["arms"]["open_cake"],
            workload=workload,
            case_id="headline_b32",
        )
        schedule = json.loads(
            (
                ROOT / "corpus/schedules/flash-kmeans-assignment-full.json"
            ).read_text(encoding="utf-8")
        )
        schedule["metadata"]["workload_contract_sha256"] = workload.canonical_sha256
        payload = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()

        result = environment.build(
            CandidateSubmission.seal(
                "application/vnd.open-cake.schedule+json", payload
            )
        )

        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["stage"], "assessment")
        self.assertIn("profile", result.feedback["error"])
        self.assertEqual(toolchain.requests, [])
        # The fourth route on real feedback: a Schedule the gates refused is the
        # candidate's to fix, and it is the fallback, so nothing else must claim it.
        from open_cake_ir.lab.routing import CANDIDATE, route_rejection

        self.assertEqual(route_rejection(result.feedback).destination, CANDIDATE)

    def test_wrong_workload_binding_is_rejected_before_toolchain(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v24.json"
            ).read_text(encoding="utf-8")
        )
        toolchain = RecordingToolchain()
        environment = OpenCakeEnvironment(
            compiler,
            toolchain,
            authority_document=study["arms"]["open_cake"],
            workload=workload,
            case_id="headline_b32",
        )
        schedule = _headline_schedule(workload)
        schedule["metadata"]["workload_contract_sha256"] = "0" * 64
        payload = json.dumps(schedule, sort_keys=True, separators=(",", ":")).encode()

        result = environment.build(
            CandidateSubmission.seal(
                "application/vnd.open-cake.schedule+json", payload
            )
        )

        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["stage"], "assessment")
        self.assertIn("Workload binding", result.feedback["error"])
        self.assertEqual(toolchain.requests, [])


if __name__ == "__main__":
    unittest.main()


class VocabularyRejectionRoutesAcrossTheSeamTest(unittest.TestCase):
    """A Compiler refusal has to reach the router in the shape the router reads.

    The routing tests build feedback by hand, so they prove the classifier and not the
    seam. If `_finding_rows` and the router disagree about what a finding row carries,
    both suites pass and every vocabulary gap gets blamed on the candidate.

    The Schedule here is well formed -- the IR admits the dtype and so does the Target --
    and this backend cannot name it. That is the compiler's gap, not the author's.
    """

    def _rejection(self, mutate):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v24.json"
            ).read_text(encoding="utf-8")
        )
        document = _headline_schedule(workload)
        mutate(document)
        environment = OpenCakeEnvironment(
            compiler,
            RecordingToolchain(),
            authority_document=study["arms"]["open_cake"],
            workload=workload,
            case_id="headline_b32",
        )

        return environment.build(
            CandidateSubmission.seal(
                "application/vnd.open-cake.schedule+json",
                json.dumps(document, sort_keys=True, separators=(",", ":")).encode(),
            )
        )

    @staticmethod
    def _retype(document):
        for buffer in document["buffers"]:
            if buffer["name"] == "distance_tile":
                # fp8 is a dtype the IR admits and this profile's backend cannot name.
                buffer["dtype"] = "fp8_e4m3"

    @staticmethod
    def _rekind(document):
        for operation in document["operations"]:
            if operation["id"] == "store_assignment":
                # An epilogue is an operation kind the IR admits and Triton has no body
                # for. The Schedule stays well formed; the backend runs out of vocabulary.
                operation["kind"] = "epilogue"
                operation["parameters"] = {
                    "formula": "centroid_sq_minus_two_dot",
                    "coalesced": True,
                }

    @staticmethod
    def _omit_mma_instruction(document):
        for operation in document["operations"]:
            if operation["kind"] == "mma":
                operation["parameters"].pop("instruction")

    def test_a_gap_the_backend_has_reaches_the_router_as_the_vocabularys(self) -> None:
        from open_cake_ir.lab.routing import IR_VOCABULARY, route_rejection

        for name, mutate, code in (
            ("dtype", self._retype, "PROFILE_DTYPE_UNEMITTABLE"),
            ("operation kind", self._rekind, "PROFILE_OPERATION_UNEMITTABLE"),
        ):
            with self.subTest(gap=name):
                result = self._rejection(mutate)
                self.assertEqual(result.disposition, "rejected")
                codes = {
                    row["code"]
                    for row in result.feedback["findings"]
                    if row["blocks_lowering"]
                }
                self.assertIn(code, codes)
                self.assertEqual(
                    route_rejection(result.feedback).destination, IR_VOCABULARY
                )

    def test_an_emitted_mma_without_an_instruction_is_candidate_feedback(self) -> None:
        from open_cake_ir.lab.routing import CANDIDATE, route_rejection

        result = self._rejection(self._omit_mma_instruction)

        self.assertEqual(result.disposition, "rejected")
        codes = {
            row["code"]
            for row in result.feedback["findings"]
            if row["blocks_lowering"]
        }
        self.assertIn("PROFILE_MMA_INSTRUCTION_REQUIRED", codes)
        self.assertEqual(route_rejection(result.feedback).destination, CANDIDATE)
