from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir import compiler as public
from open_cake_ir.compiler import core, corpus, diagnostics, errors, release, verifier
from open_cake_ir.compiler.target import TargetSource


class CompilerConvergenceContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compiler = public.Compiler.load(ROOT, ROOT / "compiler/revision.json")

    def document(self, name: str = "fma-b8-smoke") -> dict:
        return json.loads((ROOT / "corpus/schedules" / f"{name}.json").read_text())

    def cute(self) -> dict:
        return self.document("flash-kmeans-assignment-full")

    def test_public_diagnostic_error_and_report_types_have_one_owner(self) -> None:
        self.assertIs(public.CompilerError, errors.CompilerError)
        self.assertIs(core.CompilerError, errors.CompilerError)
        self.assertIs(release.CompilerError, errors.CompilerError)
        self.assertIs(public.CorpusCaseReport, corpus.CorpusCaseReport)
        self.assertIs(core.CorpusGateReport, corpus.CorpusGateReport)
        for name in ("Finding", "FindingCategory", "FindingSeverity"):
            self.assertIs(getattr(public, name), getattr(diagnostics, name))
            self.assertIs(getattr(verifier, name), getattr(diagnostics, name))

    def test_keyword_construction_keeps_target_binding_snapshot_semantics(self) -> None:
        revision = self.compiler._revision
        targets = dict(revision.targets)
        compiler = public.Compiler(
            project_root=revision.project_root,
            revision_id=revision.revision_id,
            revision_sha256=revision.canonical_sha256,
            state=revision.state,
            target_definitions=targets,
            corpus_path=revision.corpus_path,
            calibration_coverage=revision.calibration_coverage,
        )
        targets.clear()
        self.assertEqual(compiler.state, self.compiler.state)
        self.assertEqual(compiler.assess(self.document()), self.compiler.assess(self.document()))

    def test_assess_lower_profile_and_rank_reuse_the_admitted_target(self) -> None:
        document = self.document()
        initial = self.compiler.assess(document)
        proposal = json.loads((ROOT / "compiler/revision.json").read_text())
        proposal["calibration_coverage"] = [initial.analysis["semantic_sha256"]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "draft.json"
            path.write_text(json.dumps(proposal))
            compiler = public.Compiler.load(ROOT, path)
        target = compiler._revision.targets[document["target"]]
        with (
            mock.patch.object(public.Target, "from_dict", side_effect=AssertionError("Target reparsed")),
            mock.patch.object(TargetSource, "document", new_callable=mock.PropertyMock,
                              side_effect=AssertionError("provenance used as hardware")),
            mock.patch.object(core, "verify_contracts", wraps=core.verify_contracts) as verify_call,
            mock.patch.object(core.triton, "preflight", wraps=core.triton.preflight) as preflight,
            mock.patch.object(core.triton, "emit", wraps=core.triton.emit) as emit,
            mock.patch.object(core, "rank_candidates", wraps=core.rank_candidates) as rank,
        ):
            assessment = compiler.assess(document)
            self.assertTrue(assessment.lowering_eligible)
            self.assertTrue(compiler.lower(assessment).source)
            compiler.profile(assessment)
            compiler.rank([assessment])
        for calls in (verify_call, preflight, emit, rank):
            self.assertTrue(calls.call_args_list)
            for call in calls.call_args_list:
                self.assertIs(call.args[1], target)

    def test_assessment_replay_still_protects_all_consumers(self) -> None:
        assessment = self.compiler.assess(self.document())
        for changed, message in (
            (replace(assessment, compiler_revision_id="different-revision"), "different Compiler Revision"),
            (replace(assessment, accepted=False), "canonical Schedule replay"),
        ):
            for consume in (self.compiler.lower, self.compiler.profile,
                            lambda item: self.compiler.rank([item])):
                with self.subTest(changed=message, consume=consume):
                    with self.assertRaisesRegex(public.CompilerError, message):
                        consume(changed)

    def test_unknown_target_never_substitutes_hardware(self) -> None:
        document = self.document()
        document["target"] = "unbound_target"
        with mock.patch.object(core, "verify_contracts", side_effect=AssertionError("substitute Target")):
            assessment = self.compiler.assess(document)
        self.assertFalse(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertEqual([finding.code for finding in assessment.findings], ["TARGET_UNSUPPORTED"])

    def test_allocation_overflow_is_single_and_localized_in_public_assessment(self) -> None:
        document = self.cute()
        index = next(i for i, buffer in enumerate(document["buffers"])
                     if buffer.get("allocation") == "smem_operands")
        document["buffers"][index]["byte_offset"] = document["allocations"][0]["size_bytes"] + 16
        assessment = self.compiler.assess(document)
        overflow = [finding for finding in assessment.findings if finding.code == "BUFFER_ALLOCATION_OVERFLOW"]
        self.assertEqual(len(overflow), 1)
        self.assertEqual(overflow[0].path, f"buffers[{index}].byte_offset")
        self.assertIs(overflow[0].category, public.FindingCategory.DATA_CONSISTENCY)
        self.assertTrue(overflow[0].blocks_acceptance)
        self.assertFalse(assessment.lowering_eligible)

    def test_invalid_global_allocation_does_not_borrow_the_overflow_check(self) -> None:
        document = self.document()
        document["allocations"] = [{"name": "tiny", "space": "shared", "size_bytes": 1}]
        document["buffers"][0]["allocation"] = "tiny"
        assessment = self.compiler.assess(document)
        codes = [finding.code for finding in assessment.findings]
        self.assertEqual(codes.count("BUFFER_ALLOCATION_OVERFLOW"), 1)
        self.assertIn("BUFFER_GLOBAL_ALLOCATION", codes)
        self.assertFalse(assessment.accepted)

    def test_program_map_grid_limit_uses_the_same_projection_as_analysis(self) -> None:
        document = self.document()
        extent = self.compiler._revision.targets[document["target"]].resource_limits.maximum_grid[2] + 1
        document["program_map"]["axes"][0]["axis"] = 2
        document["buffers"][0]["shape"][0] = extent
        assessment = self.compiler.assess(document)
        findings = [finding for finding in assessment.findings if finding.code == "TARGET_GRID_LIMIT"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].path, "grid[2]")
        self.assertEqual(assessment.analysis["grid"], (1, 1, extent))
        self.assertFalse(assessment.accepted)

    def test_forward_dependency_without_data_flow_is_still_rejected(self) -> None:
        document = self.document()
        document["operations"][0]["depends_on"] = [document["operations"][1]["id"]]
        assessment = self.compiler.assess(document)
        findings = [finding for finding in assessment.findings if finding.code == "OPERATION_DEPENDENCY_ORDER"]
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].path, "operations[0].depends_on")
        self.assertFalse(assessment.accepted)

    def test_old_list_container_restrictions_remain_at_the_compiler_boundary(self) -> None:
        for field in ("reads", "writes", "waits", "signals", "depends_on"):
            document = self.document()
            document["operations"][0][field] = tuple(document["operations"][0].get(field, []))
            with self.subTest(field=field), self.assertRaises(public.CompilerError) as raised:
                self.compiler.assess(document)
            self.assertEqual(str(raised.exception), f"operations[0].{field} must be a list of non-empty strings")
        document = self.document()
        document["outputs"] = tuple(document["outputs"])
        with self.assertRaisesRegex(public.CompilerError, "^outputs must be a list"):
            self.compiler.assess(document)
        document = self.document("flash-kmeans-b32-smoke-v2")
        document["tile_loops"][0]["body"] = tuple(document["tile_loops"][0]["body"])
        with self.assertRaisesRegex(public.CompilerError, r"^tile_loops\[0\].body must be a list"):
            self.compiler.assess(document)

    def test_sequences_previously_admitted_by_the_compiler_remain_admitted(self) -> None:
        for owner, field in (("tile_loops", "body"), ("barriers", "producers"), ("barriers", "consumers")):
            document = self.cute()
            document[owner][0][field] = tuple(document[owner][0][field])
            with self.subTest(owner=owner, field=field):
                assessment = self.compiler.assess(document)
                self.assertTrue(assessment.accepted)
                self.assertTrue(assessment.lowering_eligible)

    def test_structural_rejections_precede_legacy_container_errors(self) -> None:
        for value in ("output", None, (123,)):
            document = self.document()
            document["outputs"] = value
            with self.subTest(value=value):
                assessment = self.compiler.assess(document)
                self.assertEqual(assessment.findings[0].code, "SCHEDULE_STRUCTURE")
                self.assertFalse(assessment.accepted)
        for operations in (None, [None]):
            document = self.document()
            document["outputs"] = tuple(document["outputs"])
            document["operations"] = operations
            assessment = self.compiler.assess(document)
            self.assertEqual(assessment.findings[0].code, "SCHEDULE_STRUCTURE")
        document = self.document()
        document["outputs"] = tuple(document["outputs"])
        document["buffers"][0]["dtype"] = "invalid_dtype"
        self.assertEqual(self.compiler.assess(document).findings[0].path, "schedule.buffers[0].dtype")
        document = self.document()
        document["outputs"] = b"output"
        with self.assertRaisesRegex(TypeError, "bytes is not JSON serializable"):
            self.compiler.assess(document)

    def test_duplicate_name_errors_survive_known_and_unknown_targets(self) -> None:
        for target in ("sm_100a", "unbound_target"):
            for collection in ("roles", "allocations", "buffers", "pipelines", "barriers", "operations"):
                document = self.cute()
                document["target"] = target
                index = len(document[collection])
                duplicate = copy.deepcopy(document[collection][0])
                document[collection].append(duplicate)
                field = "id" if collection == "operations" else "name"
                with self.subTest(target=target, collection=collection), self.assertRaises(public.CompilerError) as raised:
                    self.compiler.assess(document)
                self.assertEqual(str(raised.exception), f"{collection}[{index}].{field} duplicates {duplicate[field]!r}")

    def test_operation_name_and_container_errors_retain_their_relative_order(self) -> None:
        document = self.cute()
        document["operations"][0]["reads"] = tuple(document["operations"][0]["reads"])
        document["operations"].append(copy.deepcopy(document["operations"][1]))
        with self.assertRaises(public.CompilerError) as raised:
            self.compiler.assess(document)
        self.assertEqual(str(raised.exception), "operations[0].reads must be a list of non-empty strings")
        for tuple_index in (1, 3):
            document = self.cute()
            document["operations"].insert(1, copy.deepcopy(document["operations"][0]))
            document["operations"][tuple_index]["reads"] = tuple(document["operations"][tuple_index]["reads"])
            with self.assertRaises(public.CompilerError) as raised:
                self.compiler.assess(document)
            self.assertEqual(str(raised.exception), "operations[1].id duplicates 'load_tokens'")
        document = self.cute()
        document["roles"].append(copy.deepcopy(document["roles"][0]))
        document["operations"][0]["reads"] = tuple(document["operations"][0]["reads"])
        with self.assertRaises(public.CompilerError) as raised:
            self.compiler.assess(document)
        self.assertEqual(str(raised.exception), "roles[4].name duplicates 'epilogue'")

    def test_tile_loop_name_conflicts_remain_findings(self) -> None:
        document = self.cute()
        document["tile_loops"].append(copy.deepcopy(document["tile_loops"][0]))
        assessment = self.compiler.assess(document)
        self.assertTrue(any(finding.code == "NAME_DUPLICATE" and finding.path == "tile_loops"
                            for finding in assessment.findings))
        self.assertFalse(assessment.accepted)

    def test_backend_only_refusals_and_typed_guidance_keep_their_dispositions(self) -> None:
        assessment = self.compiler.assess(self.document("flash-kmeans-b32-warp-specialized-argmin"))
        finding = next(item for item in assessment.findings
                       if item.code == "TRITON_WARP_SPECIALIZED_ARGMIN_UNSUPPORTED")
        self.assertTrue(assessment.accepted)
        self.assertFalse(assessment.lowering_eligible)
        self.assertFalse(finding.blocks_acceptance)
        self.assertTrue(finding.blocks_lowering)
        assessment = self.compiler.assess(self.document("tinygemm2-stage4-split-k"))
        self.assertTrue(assessment.guidance)
        self.assertTrue(assessment.lowering_eligible)
        self.assertTrue(all(finding.severity is public.FindingSeverity.HINT and not finding.blocks_lowering
                            for finding in assessment.guidance))

    def test_same_codes_at_different_regions_are_not_globally_deduplicated(self) -> None:
        for name, code, count in (
            ("flash-kmeans-assignment-full-shape-drift", "CUTE_LOAD_DESCRIPTOR_REQUIRED", 2),
            ("fma-b8-smoke-vector-span-drift", "TRITON_ARANGE_RANGE_UNSUPPORTED", 4),
        ):
            findings = [finding for finding in self.compiler.assess(self.document(name)).findings
                        if finding.code == code]
            self.assertEqual(len(findings), count)
            self.assertEqual(len({finding.path for finding in findings}), count)

    def test_root_envelope_errors_keep_the_existing_exception_boundary(self) -> None:
        for mutation in (lambda item: item.pop("buffers"),
                         lambda item: item.update(unknown_field=True),
                         lambda item: item.update(grid=[1, 1, 1])):
            document = self.document()
            mutation(document)
            with self.assertRaises(public.CompilerError):
                self.compiler.assess(document)


if __name__ == "__main__":
    unittest.main()
