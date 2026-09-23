"""Controlled reference handoffs, using CPU semantic dependency fixtures only."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler, frontend
from open_cake_ir.lab._documents import _canonical_json_bytes
from open_cake_ir.lab.contracts import CampaignLock, StudyContract
from open_cake_ir.lab.reference_access import document_role, validate_reference_handoff
from open_cake_ir.lab.task_package import build_run_reference_documents, render_task_request
from open_cake_ir.tasks.authoring import prepare_schedule
from open_cake_ir.tasks.runtime import TaskLab
from open_cake_ir.tasks.workloads import load_workload
from tests.contracts._contexts import enter_context
from tests.contracts._executor_fixture import SemanticExecutorFixture

ROOT = Path(__file__).resolve().parents[2]
CLEAN = "matched-search-clean-start-reference-template.json"
KNOWN = "matched-search-infrastructure-template.json"


class ReferenceAccessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.external = Path(self.temporary.name).resolve()
        enter_context(self, SemanticExecutorFixture())
        self.lab = TaskLab(ROOT)

    def document(self, name=CLEAN):
        return json.loads((ROOT / "contracts/studies" / name).read_bytes())

    def preflight(self, document):
        path = self.external / "study.json"
        path.write_bytes(_canonical_json_bytes(document))
        return self.lab.preflight(path)

    def test_declaration_is_required_and_closed_for_study_and_lock(self):
        for access in (None, "all_python_is_high_level", {"kind": "clean_start"}):
            document = self.document()
            document["arms"]["open_cake"]["reference_access"] = access
            path = self.external / "study.json"
            path.write_bytes(_canonical_json_bytes(document))
            with self.assertRaisesRegex(ValueError, "reference_access"):
                StudyContract.load(path)
        lock = deepcopy(self.preflight(self.document()).document)
        arm = lock["resolved_inputs"]["arm_environments"]["open_cake"]
        del arm["reference_access"]
        with self.assertRaisesRegex(ValueError, "reference_access"):
            CampaignLock.from_dict(lock)

    def test_infrastructure_declares_known_kernel_scientific_treatment(self):
        document = self.document(KNOWN)
        self.assertEqual(document["claim_scope"], "scientific_matched_search")
        self.assertEqual({arm["reference_access"] for arm in document["arms"].values()}, {"known_kernel_reproduction"})
        lock = self.preflight(document)
        package = self.lab.task_package(lock, "direct_cuda-1")
        self.assertIn("known_kernel_reproduction", package.task_markdown)
        self.assertIn("wmma::mma_sync", package.task_markdown)
        self.assertIn("Reference role: target_reference", package.task_markdown)

    def test_bound_reproduction_rules_reach_every_request_once(self):
        source = ROOT / "contracts/scaffolds/kernel-reproduction/AGENTS.md"
        rules = source.read_bytes()
        external = self.external / "AGENTS.md"
        external.write_bytes(rules)
        document = self.document(KNOWN)
        reference = {"path": str(external), "sha256": sha256(rules).hexdigest()}
        # Matched comparisons require a shared scaffold. Preserve that existing
        # treatment rule while exercising the instruction delivery boundary.
        for arm in document["arms"].values():
            arm["scaffold"] = dict(reference)
        lock = self.preflight(document)
        package = self.lab.task_package(lock, "open_cake-1")
        self.assertNotIn(rules.decode(), package.task_markdown)
        self.assertEqual(package.agents_markdown.count(rules.decode()), 1)
        other = self.lab.task_package(lock, "direct_cuda-1")
        self.assertEqual(other.agents_markdown.count(rules.decode()), 1)
        for turn in (1, 2):
            prompt, bundle = render_task_request(package, {"turn": turn})
            delivered = json.loads(bundle)
            self.assertTrue(prompt.endswith(bundle.decode()))
            self.assertEqual(delivered["agents_markdown"], package.agents_markdown)
            self.assertIn("Write only `candidate-set.json`", delivered["agents_markdown"])
        external.write_text("Changed after preflight")
        with self.assertRaisesRegex(ValueError, "scaffold content differs from CampaignLock"):
            self.lab.task_package(lock, "open_cake-1")

    def test_reproduction_rules_cannot_be_injected_into_clean_start(self):
        document = self.document()
        source = ROOT / "contracts/scaffolds/kernel-reproduction/AGENTS.md"
        reference = {"path": str(source.relative_to(ROOT)), "sha256": sha256(source.read_bytes()).hexdigest()}
        for arm in document["arms"].values():
            arm["scaffold"] = dict(reference)
        with self.assertRaisesRegex(ValueError, "authoring_instructions.*vetted"):
            self.preflight(document)

    def test_clean_stubs_math_and_api_material_are_projected_without_complete_examples(self):
        lock = self.preflight(self.document())
        for run_id in ("open_cake-1", "direct_cuda-1"):
            package = self.lab.task_package(lock, run_id)
            self.assertIn("mathematical_specification_and_oracle", package.task_markdown)
            self.assertIn("incomplete_target_stub", package.task_markdown)
            self.assertNotIn("python-example.py", package.task_markdown)
            self.assertNotIn("candidate-baseline", package.task_markdown)
            self.assertNotIn("wmma::mma_sync", package.task_markdown)
        self.assertEqual(document_role("python-frontend.md", "clean_start"), "authoring_api")
        self.assertEqual(document_role("workload.json", "clean_start"), "mathematical_specification_and_oracle")
        with self.assertRaisesRegex(ValueError, "unclassified"):
            document_role("arbitrary-source.py", "clean_start")

    def test_full_target_references_refuse_before_compiler_for_restricted_access(self):
        for arm_name, access in (("open_cake", "clean_start"), ("direct_cuda", "clean_start"), ("direct_cuda", "direct_low_level")):
            document = self.document(KNOWN)
            document["arms"][arm_name]["reference_access"] = access
            with self.subTest(arm=arm_name, access=access), patch.object(Compiler, "load") as compiler:
                with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
                    self.preflight(document)
                compiler.assert_not_called()

    def test_external_identical_stubs_are_admitted_and_external_full_reference_is_refused(self):
        document = self.document()
        for arm_name, slot in (("open_cake", "schedule_skeleton"), ("direct_cuda", "candidate_skeleton")):
            reference = document["arms"][arm_name][slot]
            source = ROOT / reference["path"]
            destination = self.external / source.name
            destination.write_bytes(source.read_bytes())
            reference["path"] = str(destination)
        lock = self.preflight(document)
        self.assertIn("Intentionally empty", self.lab.task_package(lock, "direct_cuda-1").task_markdown)
        full = self.document(KNOWN)["arms"]["direct_cuda"]["candidate_skeleton"]
        destination = self.external / "foreign.cu"
        destination.write_bytes((ROOT / full["path"]).read_bytes())
        document["arms"]["direct_cuda"]["candidate_skeleton"] = {**full, "path": str(destination)}
        with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
            self.preflight(document)

    def test_complete_target_python_is_not_a_high_level_oracle_role(self):
        source = self.external / "complete.py"
        source.write_bytes((ROOT / "examples/python/fma.py").read_bytes())
        document = self.document()
        parsed = frontend.read_schedule(source).document
        document["arms"]["open_cake"]["schedule_skeleton"] = {
            "path": str(source), "canonical_sha256": sha256(_canonical_json_bytes(parsed)).hexdigest()}
        with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
            self.preflight(document)
        self.assertEqual(document_role("python-example.py", "known_kernel_reproduction"), "target_implementation")

    def test_unknown_environment_is_refused_without_claiming_inherited_code(self):
        for access in ("clean_start", "direct_low_level", "known_kernel_reproduction"):
            with self.subTest(access=access), self.assertRaisesRegex(
                ValueError, "unsupported Authoring Environment kind 'synthetic'"
            ):
                validate_reference_handoff(ROOT, {"arm": {
                    "environment_kind": "synthetic", "reference_access": access,
                }})

    def test_task_clean_start_delivers_only_the_workload_derived_stub(self):
        from types import SimpleNamespace
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.lab.reference_access import incomplete_schedule
        from open_cake_ir.tasks.workloads import create_task
        from open_cake_ir.tasks.normalization.study import task_run_inputs
        document, source = create_task('rmsnorm', backend='triton-b300', rows=7, columns=128)
        workload = WorkloadContract(document)
        route = frontend.parse(source).document['lowering']
        stub = incomplete_schedule(workload, 'primary', route)
        path, workload_path = self.external/'stub.json', self.external/'workload.json'
        path.write_bytes(_canonical_json_bytes(stub)); workload_path.write_bytes(_canonical_json_bytes(document))
        inputs = task_run_inputs(ROOT, workload, workload_path, path, harness='claude-code',
                                 model='fixture', effort='high', reference_access='clean_start')
        inputs['compiler_revision'] = {'path': 'compiler/revision.json', 'revision_id': 'fixture'}
        arm = inputs['authoring']
        delivered = build_run_reference_documents(ROOT, SimpleNamespace(document=inputs), arm,
                                                   workload_contract=workload, prepare_schedule=prepare_schedule)
        self.assertNotIn('schedule-starter.py', delivered)
        self.assertNotIn('python-example.py', delivered)
        self.assertNotIn('paired-triton-authoring.md', delivered)
        self.assertEqual(json.loads(delivered['schedule-skeleton.json']), stub)
        self.assertEqual(stub['operations'], [])
        self.assertEqual(stub['target'], 'sm_103a')
        for field, value in (('metadata', {'note': source}), ('operations', [{'id': 'hidden_implementation'}]),
                             ('buffers', [])):
            changed = deepcopy(stub); changed[field] = value
            path.write_bytes(_canonical_json_bytes(changed))
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'target_implementation.*forbidden'):
                validate_reference_handoff(ROOT, {'author': arm}, workload=workload, case_id='primary')

    def test_incomplete_python_starter_exposes_abi_without_an_implementation(self):
        from open_cake_ir.lab.reference_access import render_incomplete_python_starter
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.tasks.workloads import create_task
        document, complete = create_task('rmsnorm', backend='triton-b300', rows=7, columns=128)
        workload = WorkloadContract(document)
        route = frontend.parse(complete).document['lowering']
        starter = render_incomplete_python_starter(workload, 'primary', route)
        self.assertIn(b'@cake.schedule(', starter)
        self.assertIn(b'cake.Tensor((7, 128), "fp32", mode="input")', starter)
        self.assertIn(b'cake.Tensor((7, 128), "fp32", mode="output")', starter)
        self.assertTrue(starter.rstrip().endswith(b'...'))
        self.assertNotIn(b'lm.role(', starter)
        self.assertNotIn(b'lm.load(', starter)
        self.assertNotIn(b'lm.store(', starter)
        with self.assertRaises(frontend.FrontendError):
            frontend.parse(starter.decode())

    def test_python_clean_start_handoff_and_package_refuse_contaminated_source(self):
        from types import SimpleNamespace
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.lab.reference_access import render_incomplete_python_starter
        from open_cake_ir.tasks.normalization.study import task_run_inputs
        from open_cake_ir.tasks.workloads import create_task
        document, complete = create_task('rmsnorm', backend='triton-b300', rows=7, columns=128)
        workload = WorkloadContract(document)
        route = frontend.parse(complete).document['lowering']
        expected = render_incomplete_python_starter(workload, 'primary', route)
        starter = self.external/'starter.py'
        workload_path = self.external/'workload.json'
        starter.write_bytes(expected)
        workload_path.write_bytes(_canonical_json_bytes(document))
        inputs = task_run_inputs(ROOT, workload, workload_path, starter, harness='claude-code',
                                 model='fixture', effort='high', reference_access='clean_start',
                                 lowering_route=route)
        arm = inputs['authoring']
        self.assertEqual(arm['input_format'], 'python_source_v1')
        self.assertEqual(arm['tool_surface'], ['submit_python_source'])
        self.assertNotIn('schedule_skeleton', arm)
        validate_reference_handoff(ROOT, {'author': arm}, workload=workload, case_id='primary')
        delivered = build_run_reference_documents(ROOT, SimpleNamespace(document=inputs), arm,
                                                   workload_contract=workload,
                                                   prepare_schedule=prepare_schedule)
        self.assertEqual(delivered['schedule-starter.py'], expected)
        self.assertNotIn('schedule-skeleton.json', delivered)
        self.assertNotIn('python-example.py', delivered)
        starter.write_bytes(expected + b'\n# hidden implementation\n')
        with self.assertRaisesRegex(ValueError, 'unreviewed target reference'):
            validate_reference_handoff(ROOT, {'author': arm}, workload=workload, case_id='primary')

    def test_paired_python_clean_start_study_preserves_direct_cuda_treatment(self):
        from open_cake_ir.lab.reference_access import (
            PYTHON_CLEAN_START_SCAFFOLD, render_incomplete_python_starter,
        )
        document = self.document()
        workload = load_workload(ROOT / document['workload']['path'])
        cake = document['arms']['open_cake']
        route = cake['lowering_route']
        starter = self.external/'paired-starter.py'
        expected = render_incomplete_python_starter(workload,
            document['evaluation_protocol']['case_id'], route)
        starter.write_bytes(expected)
        del cake['schedule_skeleton']
        cake.update(input_format='python_source_v1', tool_surface=['submit_python_source'],
                    python_starter={'path': str(starter)})
        scaffold = {'path': PYTHON_CLEAN_START_SCAFFOLD,
                    'sha256': sha256((ROOT/PYTHON_CLEAN_START_SCAFFOLD).read_bytes()).hexdigest()}
        for arm in document['arms'].values():
            arm['scaffold'] = dict(scaffold)
        lock = self.preflight(document)
        cake_package = self.lab.task_package(lock, 'open_cake-1')
        cuda_package = self.lab.task_package(lock, 'direct_cuda-1')
        self.assertIn('schedule-starter.py', cake_package.task_markdown)
        self.assertNotIn('schedule-skeleton.json', cake_package.task_markdown)
        self.assertIn('candidate-skeleton.cu', cuda_package.task_markdown)
        self.assertNotIn('schedule-starter.py', cuda_package.task_markdown)
        starter.write_bytes(expected + b'\n# leaked implementation\n')
        with self.assertRaisesRegex(ValueError, 'unreviewed target reference'):
            self.preflight(document)

    def test_inherited_native_lowering_requires_known_kernel_reproduction(self):
        document = self.document("matched-search-triton-optimization-template.json")
        document["arms"]["native_triton"]["reference_access"] = "clean_start"
        with self.assertRaisesRegex(ValueError, "inherited target_implementation"):
            validate_reference_handoff(ROOT, document["arms"])

    def test_external_scaffold_cannot_smuggle_a_target_implementation(self):
        document = self.document()
        full = self.document(KNOWN)["arms"]["direct_cuda"]["candidate_skeleton"]
        path = self.external / "claimed-instructions.md"
        path.write_bytes((ROOT / full["path"]).read_bytes())
        for arm in document["arms"].values():
            arm["scaffold"] = {"path": str(path), "sha256": full["sha256"]}
        with self.assertRaisesRegex(ValueError, "authoring_instructions.*vetted"):
            self.preflight(document)

    def test_locked_package_construction_cannot_bypass_reference_policy(self):
        original = self.preflight(self.document(KNOWN))
        document = deepcopy(original.document)
        arm = document["resolved_inputs"]["arm_environments"]["direct_cuda"]
        arm["reference_access"] = "direct_low_level"
        document["resolved_inputs"]["arm_environment_sha256"]["direct_cuda"] = sha256(_canonical_json_bytes(arm)).hexdigest()
        lock = CampaignLock.from_dict(document)
        with self.assertRaisesRegex(ValueError, "target_implementation.*forbidden"):
            self.lab.task_package(lock, "direct_cuda-1")
        foreign_arm = deepcopy(original.document["resolved_inputs"]["arm_environments"]["open_cake"])
        foreign_arm["reference_access"] = "clean_start"
        with self.assertRaisesRegex(ValueError, "differs from the frozen arm"):
            build_run_reference_documents(ROOT, original.run_specification("open_cake-1"), foreign_arm,
                workload_contract=load_workload(ROOT / original.document["workload"]["path"]), prepare_schedule=prepare_schedule)
