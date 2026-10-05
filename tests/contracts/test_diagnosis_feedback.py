"""CPU diagnosis seams; explicit drafts/doubles never qualify a host or candidate."""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.compiler import Compiler, CompilerError, LoweringRefusedError
from open_cake_ir.lab import CandidateSubmission
from open_cake_ir.lab.routing import route_rejection
from open_cake_ir.tasks.environments import TaskOpenCakeEnvironment
from open_cake_ir.tasks.workloads import load_workload
from tests.contracts._contexts import enter_context
from tests.contracts.test_authoring_environment import RecordingToolchain, _headline_schedule


class DiagnosisSeamTests(unittest.TestCase):
    def test_backend_gap_is_retained_for_a_successor_compiler_tick(self):
        from open_cake_ir.lab.candidate_filter import record_candidate_rejections
        from open_cake_ir.lab.diagnoses import rejected_peer_feedback
        from open_cake_ir.lab.environments import EnvironmentResult
        submission = CandidateSubmission.seal('application/vnd.open-cake.schedule+json',
                                              b'{"candidate":"fixture"}')
        feedback = {'stage': 'assessment', 'findings': [
            {'code': 'BACKEND_ARITHMETIC_UNSUPPORTED', 'path': 'operations[2]',
             'category': 'hardware_conformance', 'severity': 'blocking',
             'message': 'this backend has no typed arithmetic body',
             'blocks_acceptance': False, 'blocks_lowering': True}]}
        result = EnvironmentResult('rejected', submission.sha256, None, feedback)
        retained = []
        ledger = Mock()
        ledger.append.side_effect = lambda kind, payload: retained.append((kind, payload))
        record_candidate_rejections(built=[(submission, result)], evidence=None,
                                    ledger=ledger, turn_number=1, arm='open_cake')
        self.assertEqual(retained[0][0], 'candidate_rejected')
        self.assertEqual(retained[0][1]['routed_to'], 'backend_lowering')
        self.assertEqual(retained[0][1]['feedback'], feedback)
        peer, = rejected_peer_feedback([(submission, result)], arm='open_cake')
        self.assertEqual(peer['routed_to'], 'backend_lowering')
        self.assertEqual(peer['findings'][0]['path'], 'operations[2]')

    def environment(self, *, python=False):
        # This is the explicit prospective source domain, not the stale released lock.
        draft = Compiler.load(ROOT, ROOT / "compiler/revision.json")
        # Explicit interface double permits exercising a prospective Environment seam;
        # no released descriptor or admission record is created or rewritten.
        compiler = Mock(wraps=draft)
        compiler.state = "released"
        workload = load_workload(ROOT / "contracts/workloads/flash-kmeans-assign-v2.json")
        study = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
        if python:
            study["arms"]["open_cake"]["input_format"] = "schedule_or_python_v1"
        toolchain = RecordingToolchain()
        environment = TaskOpenCakeEnvironment(compiler, toolchain,
            authority_document=study["arms"]["open_cake"], workload=workload, case_id="headline_b32")
        submission = CandidateSubmission.seal(environment.media_type, json.dumps(_headline_schedule(workload)).encode())
        return compiler, environment, submission, toolchain

    def test_controlled_lowering_refusal_reaches_router_without_build(self):
        compiler, environment, submission, toolchain = self.environment()
        with patch.object(compiler, "lower", side_effect=LoweringRefusedError("Schedule does not determine its source: missing body")):
            result = environment.build(submission)
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["stage"], "lowering")
        self.assertEqual(result.feedback["code"], "LOWERING_UNDETERMINED")
        self.assertEqual(route_rejection(result.feedback).destination, "backend_triage")
        self.assertEqual(toolchain.requests, [])

    def test_actual_python_refusal_preserves_top_level_location_for_peer(self):
        from open_cake_ir.lab.diagnoses import rejected_peer_feedback
        _, environment, _, toolchain = self.environment(python=True)
        submission = CandidateSubmission.seal(environment.media_type, json.dumps({"python_source": "def broken(:\n"}).encode())
        result = environment.build(submission)
        self.assertEqual(result.disposition, "rejected")
        self.assertEqual(result.feedback["code"], "PYTHON_SYNTAX")
        peer = rejected_peer_feedback([(submission, result)], arm="open_cake")[0]
        self.assertEqual(peer["source_location"], {key: value for key, value in result.feedback["source_location"].items() if key != "filename"})
        self.assertEqual(peer["source_location"]["line"], 1)
        self.assertNotIn("filename", peer["source_location"])
        self.assertEqual(toolchain.requests, [])

    def test_unrelated_compiler_fault_is_not_candidate_rejection(self):
        compiler, environment, submission, toolchain = self.environment()
        with patch.object(compiler, "lower", side_effect=CompilerError("Revision source binding differs")):
            with self.assertRaisesRegex(CompilerError, "Revision source"):
                environment.build(submission)
        self.assertEqual(toolchain.requests, [])

    def test_compile_route_uses_frozen_arm_not_candidate_feedback(self):
        for arm in ("direct_cuda", "native_triton", "native_cute_dsl"):
            with self.subTest(arm=arm):
                self.assertEqual(route_rejection({"stage": "compile", "source_role": "lowered_source"}, arm=arm).destination, "candidate")
        self.assertEqual(route_rejection({"stage": "compile"}, arm="open_cake").destination, "verifier")
        with self.assertRaises(ValueError):
            route_rejection({"stage": "compile"}, arm="unknown")


class DiagnosisProjectionTests(unittest.TestCase):
    def test_all_peers_bounded_without_source_or_artifact_leakage(self):
        from open_cake_ir.lab.diagnoses import rejected_peer_feedback
        from open_cake_ir.lab.environments import EnvironmentResult
        built = []
        for index in range(3):
            submission = CandidateSubmission.seal("text/x-cuda", f"candidate {index}".encode())
            result = EnvironmentResult("rejected", submission.sha256, None,
                {"stage": "assessment", "error": "x" * 4000, "source": "hidden implementation",
                 "findings": [{"code": f"BAD_{n}", "path": f"operations[{n}]", "message": "m" * 900,
                               "blocks_acceptance": True, "unbounded": {"source": "hidden"}}
                              for n in range(12)]}, {"source": b"hidden implementation"})
            built.append((submission, result))
        rows = rejected_peer_feedback(built, arm="direct_cuda")
        self.assertEqual([row["candidate_sha256"] for row in rows], [submission.sha256 for submission, _ in built])
        for row in rows:
            self.assertEqual(len(row["findings"]), 8)
            self.assertEqual(row["omitted_findings"], 4)
            self.assertTrue(row["text_truncated"])
            self.assertEqual(len(row["error"]), 512)
            self.assertEqual(row["findings"][0]["path"], "operations[0]")
        self.assertNotIn("hidden", json.dumps(rows))
        self.assertNotIn("unbounded", json.dumps(rows))

    def test_qsa_authored_compile_refusal_uses_common_router(self):
        from open_cake_ir.tasks.qsa.feedback import qsa_evaluation_feedback
        from tests.contracts.test_qsa_feedback import _completed
        value = _completed()
        value.update(outcome="rejected", validity="invalid")
        value["stages"] = [{"id": "compile", "status": "failed", "summary": "syntax error"}]
        for arm, destination in (("direct_cuda", "candidate"), ("open_cake", "verifier")):
            feedback = qsa_evaluation_feedback(value, arm=arm)
            self.assertEqual(feedback["routed_to"], destination)
            self.assertEqual(feedback["routing_reason"], route_rejection({"stage": "compile"}, arm=arm).reason)


class DiagnosisRunTests(unittest.TestCase):
    def setUp(self):
        from types import SimpleNamespace
        from tests.contracts._executor_fixture import SemanticExecutorFixture, compiler_reference
        enter_context(self, SemanticExecutorFixture())
        # Diagnosis consumer test only: Compiler admission has its own release tests.
        # Exact interface fixture binds the same declared input at preflight/replay;
        # it neither writes a release nor represents a successful Corpus Gate.
        reference = compiler_reference(ROOT)
        gate = SimpleNamespace(compiler_revision_id=reference["revision_id"])
        def resolve(root, value, context, *, template):
            if value != ({"binding": "current_release"} if template else reference):
                raise ValueError("diagnosis fixture Compiler reference differs")
            return gate, reference["path"], reference
        def load(root, value, context):
            if value != reference:
                raise ValueError("diagnosis fixture Compiler reference differs")
        enter_context(self, patch("open_cake_ir.lab.preflight._resolve_compiler_reference", resolve))
        enter_context(self, patch("open_cake_ir.lab.bindings.load_compiler_reference", load))

    def test_real_next_turn_and_replay_retain_all_rejected_peers(self):
        import dataclasses
        import tempfile
        from hashlib import sha256
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.lab.environments import EnvironmentResult
        from open_cake_ir.tasks.runtime import TaskLab
        from tests.contracts.test_lab import FakeProvider, FakeEnvironment, FakeEvaluator, _enable_candidate_set, _execute, _submission_envelope

        class ThreeCandidates(FakeProvider):
            def turn(self, request):
                turn = super().turn(request)
                candidates = tuple(json.dumps({"turn": request.turn, "variant": i}, sort_keys=True, separators=(",", ":")).encode() for i in range(3))
                return dataclasses.replace(turn, candidates=candidates,
                    candidate_sha256s=tuple(sha256(value).hexdigest() for value in candidates),
                    raw_submission=_submission_envelope(request.arm, candidates))

        for mixed in (False, True):
            class Rejections(FakeEnvironment):
                def build(self, submission, *, compilation=None):
                    variant = json.loads(submission.payload)["variant"]
                    if mixed and variant == 0:
                        return super().build(submission, compilation=compilation)
                    return EnvironmentResult("rejected", submission.sha256, None,
                        {"stage": "compile", "diagnostic": f"syntax variant {variant}"})
            with self.subTest(mixed=mixed), tempfile.TemporaryDirectory() as directory:
                document = json.loads((ROOT / "contracts/studies/matched-search-infrastructure-template.json").read_text())
                _enable_candidate_set(document, 3)
                document["budget"]["maximum_turns"] = 2
                study = Path(directory) / "study.json"
                study.write_text(json.dumps(document))
                lab = TaskLab(ROOT)
                lock = lab.preflight(study)
                arms = lock.document["resolved_inputs"]["arm_environments"]
                provider = ThreeCandidates()
                protocol = lock.document["evaluation_protocol"]
                evaluator = FakeEvaluator(protocol, sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), lock.document["workload"]["canonical_sha256"])
                campaign = _execute(lab, lock, Path(directory) / "evidence", provider=provider,
                    environments={arm: Rejections(arm, authority) for arm, authority in arms.items()}, evaluator=evaluator)
                next_turns = [request for request in provider.requests if request.turn == 2]
                self.assertEqual(len(next_turns), len(lock.run_order))
                for request in next_turns:
                    peers = request.feedback["rejected_candidates"]
                    self.assertEqual(len(peers), 2 if mixed else 3)
                    self.assertEqual([row["diagnostic"] for row in peers],
                                     [f"syntax variant {i}" for i in range(1 if mixed else 0, 3)])
                    self.assertTrue(all(row["routed_to"] == ("verifier" if request.arm == "open_cake" else "candidate") for row in peers))
                    if mixed:
                        self.assertEqual(request.feedback["kind"], "evaluation")
                report = lab.audit(campaign)
                self.assertTrue(report.semantic_replay_passed)
                store = EvidenceStore.open(campaign.evidence_root)
                run_id = next(run for run in lock.run_order if run.startswith("direct_cuda"))
                audit = store.audit_run(run_id)
                events = [dict(event) for event in store.replay_events(run_id)]
                rejected = next(event for event in events if event["kind"] == "candidate_rejected")
                rejected["payload"] = {**rejected["payload"], "routed_to": "verifier"}
                # In-memory tamper at the independent semantic boundary; archive untouched.
                with patch.object(store, "replay_events", return_value=tuple(events)):
                    self.assertFalse(lab._replay_matched_run(store, audit, lock))


class CompilerDiagnosisOwnershipTests(unittest.TestCase):
    def test_existing_storage_and_coordinate_repair_are_not_missing_ir(self):
        for code,path in (('TRITON_ELEMENTWISE_STORAGE','operations[1].reads[0]'),
                          ('TRITON_LOOP_STORE_OWNERSHIP','operations[16]')):
            route=route_rejection({'stage':'assessment','findings':[{'code':code,'path':path,
                'blocks_acceptance':False,'blocks_lowering':True}]})
            self.assertEqual(route.destination,'candidate')
            self.assertIn('declaration change',route.reason)

    def test_loop_topology_omission_is_backend_owned_and_route_qualification_is_explicit(self):
        route=route_rejection({'stage':'assessment','findings':[{'code':'TRITON_LOOP_NEST_UNSUPPORTED',
            'path':'tile_loops','blocks_acceptance':False,'blocks_lowering':True}]})
        self.assertEqual(route.destination,'backend_lowering')
        for code in ('MACA_REGISTER_BUDGET_UNQUALIFIED','MACA_WARP_COUNT_UNQUALIFIED'):
            route=route_rejection({'stage':'assessment','findings':[{'code':code,'path':'roles',
                'blocks_acceptance':False,'blocks_lowering':True}]})
            self.assertEqual(route.destination,'backend_triage')
            self.assertIn('qualification',route.reason)
            self.assertIn('expressible',route.reason)

class DiagnosisSummaryTests(unittest.TestCase):
    def test_missing_malformed_or_conflicting_policies_are_not_substituted(self):
        import os
        import tempfile
        from hashlib import sha256
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.serialization import canonical_json_bytes
        from tools.summarize_diagnoses import summarize
        policy = {"event_vocabulary": "current-fixture"}
        cases = [
            ({"evidence_policy": policy, "resolved_inputs": {"evidence_policy": {"event_vocabulary": "other"}}}, "conflicting"),
            ({"evidence_policy": None, "resolved_inputs": {"evidence_policy": policy}}, "conflicting"),
            ({"evidence_policy": policy, "resolved_inputs": None}, "authority differs"),
            ({}, "requires retained Executor and evidence policy"),
            ({"evidence_policy": []}, "requires retained Executor and evidence policy"),
        ]
        for values, expected in cases:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                with patch.dict(os.environ, {"OPEN_CAKE_CUSTODY_DIRECTORY": str(root / "registry")}):
                    store = EvidenceStore.create(root / "evidence")
                    authority = {"run_id": "fixture-run", "execution": {
                        "executor_revision": {"executor_id": "fixture-executor"}}, **values}
                    run = store.start_run("fixture-run", authority=authority,
                        authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest())
                    run.seal(protocol_adherence="adhered", endpoint_observation="observed",
                             endpoint={"kind": "archive-reader-fixture"})
                    with self.assertRaisesRegex(ValueError, expected):
                        summarize([store.root], compiler_gaps=True)

    def test_transform_refusals_keep_permissions_guards_and_source_locators_separate(self):
        import os
        import shutil
        import tempfile
        from hashlib import sha256
        from open_cake_ir.compiler.frontend import parse
        from open_cake_ir.compiler.ir import Program
        from open_cake_ir.evaluation.workload import WorkloadContract
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.lab.actions import ActionResolution, resolve_action
        from open_cake_ir.serialization import canonical_json_bytes
        from open_cake_ir.tasks.contraction.authoring import starter_source
        from open_cake_ir.tasks.contraction.workload import workload_document
        from tools.summarize_diagnoses import summarize

        compiler = Compiler.load(ROOT, ROOT / 'compiler/revision.json')
        workload = WorkloadContract(workload_document(
            'pairwise_sqdist', rows=64, depth=256, columns=32, backend='triton-metax'))
        program = Program.from_schedule(parse(starter_source(workload)).document)
        program_bytes = canonical_json_bytes(program.document)
        transform = 'tile_squared_difference'
        parameters = {'stage': program.stages[0].name, 'k_tile': 3,
                      'schedule_id': 'proposed', 'entry_point': 'proposed'}
        baseline = 'baseline:starter/a~b'
        context = {'environment_kind': 'open_cake', 'transformations': (transform,),
                   'candidates': {}, 'baselines': {'starter/a~b': program_bytes},
                   'compiler_factory': lambda: compiler}
        submitted = resolve_action(program_bytes, **context)
        parent = submitted.document['candidate_sha256']
        context['candidates'][parent] = program_bytes
        requests = [
            {'action': 'transform', 'parent': baseline, 'transformation': 'not_granted',
             'parameters': parameters},
            {'action': 'transform', 'parent': 'unavailable-parent', 'transformation': transform,
             'parameters': parameters},
            {'action': 'transform', 'parent': baseline, 'transformation': transform,
             'parameters': {**parameters, 'stage': 'not_a_stage'}},
            {'action': 'transform', 'parent': parent, 'transformation': transform,
             'parameters': parameters},
            {'action': 'transform', 'parent': baseline, 'transformation': transform,
             'parameters': {**parameters, 'k_tile': 64}},
        ]
        resolutions = [resolve_action(canonical_json_bytes(value), **context) for value in requests]
        self.assertEqual([row.reason for row in resolutions],
                         ['transform_not_granted', 'parent_not_authorized', 'stage_selection',
                          'tile_extent', 'applied'])
        # A future archived reason and a wrapped refusal require owner judgement.
        resolutions.extend(ActionResolution('f' * 64, 'transform', None, baseline,
                           transform, reason, 'Unresolved historical refusal')
                           for reason in ('future_reason', 'result_refused'))
        resolutions.append(ActionResolution('e' * 64, 'submit', None,
                                           reason='author_format', message='Not a transform'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {'OPEN_CAKE_CUSTODY_DIRECTORY': str(root / 'registry')}):
                store = EvidenceStore.create(root / 'evidence')
                authority = {'execution': {
                    'executor_revision': {'executor_id': 'fixture-source'}},
                    'reference_inputs': {'baseline_programs': {'starter/a~b': program.document}},
                    'run_id': 'open_cake-1', 'evidence_policy': {'event_vocabulary': 'fixture'}}
                run = store.start_run('open_cake-1', authority=authority,
                    authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest())
                obj = store.put(program_bytes, media_type='application/json')
                run.append('author_actions_resolved', {'turn': 1, 'actions': [
                    {'ordinal': 0, **submitted.document, 'objects': [obj.reference('resolved_candidate')]}]})
                references = [store.put(b'# retained raw author source\n', media_type='text/x-python')
                              .reference('provider_source_file')]
                for index, value in enumerate(requests):
                    references.append(store.put(canonical_json_bytes(value), media_type='application/json')
                                      .reference(f'candidate_submission_{index:04d}'))
                run.append('provider_turn_completed', {'turn': 2, 'objects': references})
                run.append('author_actions_resolved', {'turn': 2, 'actions': [
                    {'ordinal': index, **row.document, 'objects': (
                        [store.put(row.candidate, media_type='application/json').reference('resolved_candidate')]
                        if row.candidate is not None else [])}
                    for index, row in enumerate(resolutions)]})
                run.seal(protocol_adherence='adhered', endpoint_observation='observed',
                         endpoint={'kind': 'archive-reader-fixture'})
                mirror = root / 'mirror'
                shutil.copytree(store.root, mirror)
                before = {path: (path.read_bytes(), path.stat().st_mode)
                          for base in (store.root, mirror) for path in base.rglob('*') if path.is_file()}
                plain = summarize([store.root, mirror])
                result = summarize([store.root, mirror], compiler_gaps=True)
                self.assertNotIn('transform_refusals', plain)
                self.assertEqual(result['groups'], plain['groups'])
                self.assertEqual(result['runs'], plain['runs'])
                self.assertEqual(result['compiler_gaps'], [])
                self.assertEqual(result['groups'][0]['counts'], {})
                self.assertEqual(len(result['runs']), 1)
                leads = result['transform_refusals']
                self.assertEqual([row['action_index'] for row in leads], [0, 1, 2, 3, 5, 6])
                self.assertEqual([row['triage_owner'] for row in leads],
                    ['lab_permissions', 'lab_permissions', 'author_api', 'compiler_guard', 'unknown', 'unknown'])

                def dereference(document, pointer):
                    for part in pointer.split('/')[1:]:
                        part = part.replace('~1', '/').replace('~0', '~')
                        document = document[int(part)] if isinstance(document, list) else document[part]
                    return document

                for lead in leads:
                    retained = EvidenceStore.open(lead['evidence_root'])
                    events = {event['sequence']: event for event in retained.replay_events(lead['run_id'])}
                    locator = lead['action_locator']
                    action = dereference(events[locator['event_sequence']], locator['json_pointer'])
                    self.assertEqual(action['reason'], lead['reason'])
                    self.assertEqual(action['message'], lead['message'])
                    self.assertEqual(action['parent'], lead['parent'])
                    self.assertEqual(action['transformation'], lead['transformation'])
                    for locator in lead['submission_locators']:
                        ref = dereference(events[locator['event_sequence']], locator['json_pointer'])
                        self.assertEqual(ref['role'], locator['role'])
                        self.assertTrue(retained.read_object(ref))
                    locator = lead['parent_locator']
                    if lead['reason'] == 'parent_not_authorized':
                        self.assertIsNone(locator)
                    elif 'file' in locator:
                        document = json.loads((Path(lead['evidence_root']) / locator['file']).read_text())
                        self.assertEqual(dereference(document, locator['json_pointer']), program.document)
                    else:
                        refs = dereference(events[locator['event_sequence']], locator['json_pointer'])
                        self.assertEqual(retained.read_object(refs[0]), program_bytes)
                self.assertNotIn('sha256', json.dumps(leads))
                self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mode) for path in before})

    def test_cross_root_counts_are_read_only_deduplicated_and_policy_scoped(self):
        import os
        import shutil
        import subprocess
        import tempfile
        from hashlib import sha256
        from open_cake_ir.evidence import EvidenceStore
        from open_cake_ir.serialization import canonical_json_bytes
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch.dict(os.environ, {"OPEN_CAKE_CUSTODY_DIRECTORY": str(root / "registry")}):
                paths = []
                for index in range(2):
                    store = EvidenceStore.create(root / f"evidence-{index}")
                    authority = {"campaign_id": f"fixture-{index}", "execution": {
                        "executor_revision": {"executor_id": f"fixture-executor-{index}"}},
                        "resolved_inputs": {"evidence_policy": {"event_vocabulary": f"fixture-{index}"}}}
                    if index == 0:
                        authority.pop("campaign_id")
                        authority["run_id"] = "direct_cuda-1"
                        authority["evidence_policy"] = authority.pop("resolved_inputs")["evidence_policy"]
                    run = store.start_run("direct_cuda-1", authority=authority,
                        authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest())
                    run.append("candidate_rejected", {"turn": 1, "candidate_sha256": "a" * 64,
                        "routed_to": "candidate", "routing_reason": "retained prior routing", "feedback": {"diagnostic": "do not print source text"}})
                    if index == 0:
                        run.append("candidate_rejected", {"turn": 1, "candidate_sha256": "b" * 64,
                            "routed_to": "backend_lowering", "routing_reason": "selected backend lacks emission",
                            "feedback": {"findings": [{"code": "BACKEND_OPERATION_UNEMITTABLE", "path": "operations[3]",
                                                       "message": "private source must not be printed"}]}})
                        run.append("candidate_rejected", {"turn": 1, "candidate_sha256": "c" * 64,
                            "routed_to": "candidate", "routing_reason": "acceptance also refused",
                            "feedback": {"findings": [
                                {"code": "REDUCE_SHAPE_MISMATCH", "path": "operations[0]", "blocks_acceptance": True},
                                {"code": "BACKEND_OPERATION_UNEMITTABLE", "path": "operations[3]", "blocks_lowering": True}]}})
                        run.append("candidate_rejected", {"turn": 1, "candidate_sha256": "d" * 64,
                            "routed_to": "candidate", "routing_reason": "advisory is not a gap",
                            "feedback": {"findings": [
                                {"code": "REDUCE_SHAPE_MISMATCH", "path": "operations[0]", "blocks_acceptance": True},
                                {"code": "BACKEND_OPERATION_UNEMITTABLE", "path": "operations[3]", "blocks_lowering": False}]}})
                    run.append("diagnosis_routed", {"turn": 1, "routed_to": "cost_model", "routing_reason": "ranking inversion"})
                    run.seal(protocol_adherence="adhered", endpoint_observation="observed", endpoint={"kind": "fixture"})
                    paths.append(store.root)
                copied = root / "copied"
                shutil.copytree(paths[0], copied)
                paths.append(copied)
                before = {path: (path.read_bytes(), path.stat().st_mode) for evidence_root in paths for path in evidence_root.rglob("*") if path.is_file()}
                completed = subprocess.run([sys.executable, str(ROOT / "tools/summarize_diagnoses.py"), *map(str, paths)],
                    capture_output=True, text=True, check=True)
                result = json.loads(completed.stdout)
                self.assertEqual(len(result["groups"]), 2)
                self.assertEqual(len(result["runs"]), 2)
                self.assertEqual(sorted(group["counts"].get("candidate_rejected:backend_lowering", 0)
                                        for group in result["groups"]), [0, 1])
                for group in result["groups"]:
                    self.assertIn(group["counts"]["candidate_rejected:candidate"], (1, 3))
                    self.assertEqual(group["counts"]["diagnosis_routed:cost_model"], 1)
                self.assertTrue(any(not run["filesystem_custody_verified"] for run in result["runs"]))
                self.assertNotIn("do not print source text", completed.stdout)
                queue = subprocess.run([sys.executable, str(ROOT / "tools/summarize_diagnoses.py"),
                                        "--compiler-gaps", *map(str, paths)],
                                       capture_output=True, text=True, check=True)
                gaps = json.loads(queue.stdout)["compiler_gaps"]
                self.assertEqual(len(gaps), 2)
                self.assertEqual({gap["destination"] for gap in gaps}, {"backend_lowering", "candidate"})
                for gap in gaps:
                    self.assertEqual(gap["findings"], [{"code": "BACKEND_OPERATION_UNEMITTABLE", "path": "operations[3]"}])
                    self.assertEqual(gap["run_id"], "direct_cuda-1")
                    self.assertIsInstance(gap["event_sequence"], int)
                self.assertTrue(next(gap for gap in gaps if gap["destination"] == "candidate")["candidate_admission_blocked"])
                self.assertNotIn("private source must not be printed", queue.stdout)
                self.assertEqual(before, {path: (path.read_bytes(), path.stat().st_mode) for path in before})
                # An incomplete new root is refused, not silently omitted or repaired.
                broken = EvidenceStore.create(root / "incomplete")
                broken.start_run("direct_cuda-1", authority=authority,
                    authority_sha256=sha256(canonical_json_bytes(authority)).hexdigest())
                refused = subprocess.run([sys.executable, str(ROOT / "tools/summarize_diagnoses.py"), str(broken.root)], capture_output=True, text=True)
                self.assertEqual(refused.returncode, 1)
                self.assertIn("archive integrity failed", refused.stderr)
                self.assertEqual(refused.stdout, "")
