from __future__ import annotations

import json
import re
import sys
import unittest
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, Schedule, ScheduleParseError  # noqa: E402
from open_cake_ir.evaluation import (  # noqa: E402
    LaunchableCandidate,
    WorkloadContract,
    parse_cuda_launch_manifest,
)
from open_cake_ir.lab import (  # noqa: E402
    BuildRequest,
    CandidateSubmission,
    Lab,
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
    def test_clean_start_starters_expose_contract_without_implementation(self) -> None:
        study_path = (
            ROOT
            / "contracts/studies/matched-search-clean-start-reference-v33.json"
        )
        lock = Lab(ROOT).preflight(study_path)
        study = json.loads(study_path.read_text(encoding="utf-8"))
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        shape = workload.case("headline_b32")["shape"]

        schedule_path = ROOT / study["arms"]["open_cake"]["schedule_skeleton"]["path"]
        starter = json.loads(schedule_path.read_text(encoding="utf-8"))
        self.assertEqual(
            set(starter),
            {
                "schema_version",
                "schedule_id",
                "target",
                "roles",
                "allocations",
                "buffers",
                "pipelines",
                "barriers",
                "operations",
                "outputs",
                "metadata",
            },
        )
        self.assertNotIn("grid", starter)
        self.assertNotIn("program_map", starter)
        for field in ("roles", "allocations", "pipelines", "barriers", "operations"):
            self.assertEqual(starter[field], [], field)
        self.assertEqual(starter["outputs"], ["assignments"])
        self.assertEqual(
            starter["metadata"],
            {
                "profile": "flash_kmeans_b32_smoke",
                "starter_scope": "authoring_contract_only",
                "workload_contract_sha256": workload.canonical_sha256,
            },
        )
        buffers = {item["name"]: item for item in starter["buffers"]}
        self.assertEqual(
            buffers,
            {
                "tokens": {
                    "name": "tokens",
                    "space": "global",
                    "dtype": "bf16",
                    "shape": [shape["B"], shape["N"], shape["D"]],
                    "mode": "input",
                },
                "centroids": {
                    "name": "centroids",
                    "space": "global",
                    "dtype": "bf16",
                    "shape": [shape["B"], shape["K"], shape["D"]],
                    "mode": "input",
                },
                "centroid_sq": {
                    "name": "centroid_sq",
                    "space": "global",
                    "dtype": "fp32",
                    "shape": [shape["B"], shape["K"]],
                    "mode": "input",
                },
                "assignments": {
                    "name": "assignments",
                    "space": "global",
                    "dtype": "int32",
                    "shape": [shape["B"], shape["N"]],
                    "mode": "output",
                },
            },
        )
        with self.assertRaisesRegex(
            ScheduleParseError, "exactly one of grid or program_map"
        ):
            Schedule.from_dict(starter)

        cuda_path = ROOT / study["arms"]["direct_cuda"]["candidate_skeleton"]["path"]
        cuda_source = cuda_path.read_bytes()
        manifest = parse_cuda_launch_manifest(cuda_source)
        self.assertEqual(manifest.abi, "flash_kmeans_assign_v1")
        self.assertEqual(manifest.target, "sm_100a")
        self.assertEqual(manifest.grid, (1, 1, 1))
        self.assertEqual(manifest.block, (1, 1, 1))
        self.assertEqual(manifest.dynamic_shared_memory_bytes, 0)
        text = cuda_source.decode("utf-8")
        function_start = text.index('extern "C" __global__')
        body_start = text.index("{", function_start)
        self.assertEqual(
            text[text.index("\n") + 1 : function_start],
            "#include <cuda_bf16.h>\n#include <stdint.h>\n\n",
        )
        self.assertEqual(
            text[function_start:body_start],
            'extern "C" __global__ void flash_kmeans_assign_candidate(\n'
            "    const __nv_bfloat16* tokens,\n"
            "    const __nv_bfloat16* centroids,\n"
            "    const float* centroid_sq,\n"
            "    int32_t* assignments) ",
        )
        self.assertEqual(text[function_start:].count("{"), 1)
        self.assertEqual(text[function_start:].count("}"), 1)
        body = text[body_start + 1 : text.rindex("}")]
        body_without_comments = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
        self.assertEqual(body_without_comments.strip(), "")
        self.assertEqual(text[text.rindex("}") + 1 :], "\n")
        self.assertEqual(lock.claim_scope, "scientific_matched_search")

    def test_complete_r16_skeleton_for_study_workload_builds_headline_candidate(self) -> None:
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v33.json"
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
                ROOT / "contracts/studies/matched-search-infrastructure-v33.json"
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
                ROOT / "contracts/studies/matched-search-infrastructure-v33.json"
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
                ROOT / "contracts/studies/matched-search-infrastructure-v33.json"
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

    A backend vocabulary gap and an operation-contract violation are both well-formed
    JSON, but they have different owners.  This seam must preserve that distinction.
    """

    def _rejection(self, mutate):
        compiler = Compiler.load(ROOT, ROOT / "compiler/revision.lock.json")
        workload = WorkloadContract.load(
            ROOT / "contracts/workloads/flash-kmeans-assign-v2.json"
        )
        study = json.loads(
            (
                ROOT / "contracts/studies/matched-search-infrastructure-v33.json"
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
                # The backend can name fp8, but reduce_argmin only compares fp32 values.
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

        result = self._rejection(self._rekind)
        self.assertEqual(result.disposition, "rejected")
        codes = {
            row["code"]
            for row in result.feedback["findings"]
            if row["blocks_lowering"]
        }
        self.assertIn("PROFILE_OPERATION_UNEMITTABLE", codes)
        self.assertEqual(route_rejection(result.feedback).destination, IR_VOCABULARY)

    def test_an_argmin_dtype_violation_is_candidate_feedback(self) -> None:
        from open_cake_ir.lab.routing import CANDIDATE, route_rejection

        result = self._rejection(self._retype)
        self.assertEqual(result.disposition, "rejected")
        codes = {
            row["code"]
            for row in result.feedback["findings"]
            if row["blocks_acceptance"]
        }
        self.assertIn("REDUCE_ARGMIN_VALUE_DTYPE", codes)
        self.assertEqual(route_rejection(result.feedback).destination, CANDIDATE)

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
