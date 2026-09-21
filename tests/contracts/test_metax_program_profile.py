"""Ordered native profile admission and ordinary receipt feedback, CPU fixtures."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import Compiler
from open_cake_ir.evaluation.core import EvaluationProtocol
from open_cake_ir.evaluation.metax_program_profile import (
    MACA_PROGRAM_PROFILE, capture_program_activity, program_profile_summary,
)
from open_cake_ir.evaluation.metax_observations import NOT_COLLECTED
from open_cake_ir.evaluation.paired import candidate_identity
from open_cake_ir.evaluation.program import program_components
from open_cake_ir.evaluation.triton_metax import MetaxDeviceAdmission
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.tasks.program_evaluation import PreparedProgramCase
from open_cake_ir.tasks.solx_fib import attention
from tests.contracts.test_metax_measurement import capture, kernel
from tests.contracts.test_native_program_tensors import build

ROOT = Path(__file__).resolve().parents[2]


class ProgramProfile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workload = WorkloadContract(attention.workload_document(next(iter(attention.TASKS)),
                                        variant='boundary', backend='triton-metax'))
        cls.candidate = build(attention.launch_plan(cls.workload), cls.workload,
                              Compiler.load(ROOT, ROOT / 'compiler/revision.json'))
        cls.manifest, cls.children, cls.manifests = program_components(cls.candidate)
        cls.admission = MetaxDeviceAdmission('maca-123456789abc', 'xcore1002', 'xcore1002',
            'MetaX C550', 64, '0000:0f:00', '/opt/maca-3.5.3/lib/libmcruntime.so')
        records = []
        for index, stage in enumerate(cls.manifest.program.stages):
            item = cls.manifests[stage.name]
            records.append({**kernel(item.kernel_name, index + 1, 1000 + index * 4000,
                                    grid=item.grid, block=item.block),
                            'dynamic_shared_bytes': item.dynamic_shared_memory_bytes})
        cls.raw = {'activity': capture(*records), 'manifest': cls.manifest.as_dict(),
            'stage_manifests': {name: spec.as_dict() for name, spec in cls.manifests.items()},
            'stage_candidates': {name: candidate_identity(child) for name, child in cls.children.items()},
            'device_admission': asdict(cls.admission), 'not_collected': list(NOT_COLLECTED)}

    def test_exact_order_exposes_every_stage_and_only_attribution_intervals(self):
        summary = program_profile_summary(self.raw)
        self.assertEqual([r['stage'] for r in summary['stages']], [s.name for s in self.manifest.program.stages])
        self.assertEqual(summary['program_span_us'], 14.048)
        self.assertEqual(summary['summed_stage_time_us'], 8.192)
        self.assertEqual(summary['stages'][1]['preceding_gap_us'], 1.952)
        self.assertEqual(summary['timing_use'], 'attribution_only')

    def test_missing_extra_reordered_or_other_stream_stage_is_refused(self):
        variants = []
        for key, value in [('name', 'another_kernel'), ('stream', 1), ('grid', [999, 1, 1]),
                           ('dynamic_shared_bytes', 4096), ('end_ns', 9001)]:
            raw = deepcopy(self.raw); raw['activity']['records'][1][key] = value; variants.append(raw)
        raw = deepcopy(self.raw); raw['activity']['records'].pop(1); variants.append(raw)
        raw = deepcopy(self.raw); raw['activity']['records'] += capture(kernel('extra', 99, 20000))['records']; variants.append(raw)
        raw = deepcopy(self.raw); raw['activity']['records'][0]['name'], raw['activity']['records'][1]['name'] = (
            raw['activity']['records'][1]['name'], raw['activity']['records'][0]['name']); variants.append(raw)
        for raw in variants:
            with self.subTest(raw=raw), self.assertRaises(ValueError): program_profile_summary(raw)

    def test_child_manifests_and_actual_module_api_are_bound(self):
        name = self.manifest.program.stages[0].name
        for key, value in [('kernel_name', 'other'), ('workload_sha256', 'a' * 64),
                           ('block', [64, 1, 1]), ('hidden_null_pointer_parameters', 1)]:
            raw = deepcopy(self.raw); raw['stage_manifests'][name][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): program_profile_summary(raw)
        raw = deepcopy(self.raw); raw['stage_candidates'].pop(name)
        with self.assertRaises(ValueError): program_profile_summary(raw)

    def test_incomplete_collector_retains_raw_rows_and_stays_rejected(self):
        import threading
        from open_cake_ir.evaluation.metax_activity import McptiActivity
        from open_cake_ir.evaluation.metax_benchmark import kernel_records
        collector = object.__new__(McptiActivity)
        collector.version = 18
        collector._rows = deepcopy(self.raw['activity']['records'])
        collector._errors = ['callback failed']; collector._dropped = 2
        collector._buffers = {}; collector._enabled = []; collector._active = True
        collector._owner_thread = threading.get_ident(); collector._session = threading.Lock()
        collector._session.acquire(); collector._call = lambda *args: None
        with self.assertRaises(ValueError) as caught: collector.finish()
        raw = caught.exception.activity_snapshot
        self.assertEqual(raw['records'], self.raw['activity']['records'])
        self.assertEqual(raw['dropped_records'], 2)
        self.assertIn('callback failed', raw['collection_errors'])
        self.assertFalse(collector._active)
        with self.assertRaises(ValueError): kernel_records(raw)
        with self.assertRaises(ValueError): kernel_records({**raw, 'dropped_records': 0})
        raw = deepcopy(self.raw); raw['activity']['records'][-1]['cbid'] = 56
        with self.assertRaises(ValueError): program_profile_summary(raw)

    def test_collector_snapshot_precedes_the_next_session_after_release(self):
        import threading
        from open_cake_ir.evaluation.metax_activity import McptiActivity
        for dropped in (0, 1):
            collector = object.__new__(McptiActivity)
            collector.version = 18; collector._rows = [{'old_session': True}]
            collector._errors = []; collector._dropped = dropped
            collector._buffers = {}; collector._enabled = []; collector._active = True
            collector._owner_thread = threading.get_ident(); collector._call = lambda *args: None
            def next_session():
                collector._rows = [{'next_session': True}]
                collector._dropped = 0; collector._errors = []
            collector._session = SimpleNamespace(release=next_session)
            if dropped:
                with self.assertRaises(ValueError) as caught: collector.finish()
                snapshot = caught.exception.activity_snapshot
            else:
                snapshot = collector.finish()
            self.assertEqual(snapshot['records'], [{'old_session': True}])
            self.assertEqual(snapshot['dropped_records'], dropped)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'requires CPU Torch')
    def test_real_profile_composer_checks_outputs_and_retains_failed_capture(self):
        import torch
        spec = importlib.util.spec_from_file_location('profile_program_tool', ROOT / 'tools/qualify_tensor_program.py')
        tool = importlib.util.module_from_spec(spec); spec.loader.exec_module(tool)
        prepared = PreparedProgramCase(self.workload, 'primary')
        protocol = EvaluationProtocol('profile-fixture', 'confirmatory', self.workload.canonical_sha256, 'primary', 'none')
        owner = self
        class Loaded:
            def __init__(self, *args):
                self.loaded = SimpleNamespace(launch_calls=0, closed=False, resources={
                    'kind': 'ordered_program', 'stages': {name: {'registers_per_thread': 16,
                        'dynamic_shared_bytes': item.dynamic_shared_memory_bytes, 'local_bytes': 492}
                        for name, item in owner.manifests.items()}})
            def launch(self): self.loaded.launch_calls += owner.manifest.kernels_per_call
            def snapshot(self): return dict(prepared.expected), {name: True for name in prepared.inputs}
            def close(self): self.loaded.closed = True
        collector = SimpleNamespace(begin=lambda: None, finish=lambda: self.raw['activity'])
        with patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded), patch.object(
                torch.cuda, 'synchronize'), patch('open_cake_ir.evaluation.metax_program_profile.activity_collector', return_value=collector):
            receipt = tool.profile_program(self.candidate, self.workload, protocol, self.admission, prepared,
                                           {'activity_library': 'CPU collector fixture'})
        self.assertEqual(receipt.purpose, 'attribution')
        self.assertIsNone(receipt.timing)
        self.assertEqual(receipt.kernel_calls, 4)
        self.assertEqual(len(receipt.attribution_feedback['stages']), 4)
        self.assertEqual(receipt.attribution_feedback['stages'][0]['function_local_bytes_per_thread'], 492)
        profile = json.loads(receipt.artifact_payloads['profile'])
        launch = json.loads(receipt.artifact_payloads['launch_receipt'])
        correctness = json.loads(receipt.artifact_payloads['correctness_output'])
        launch['resources']['stages'][self.manifest.program.stages[0].name]['registers_per_thread'] = 17
        with self.assertRaisesRegex(ValueError, 'resources'): MACA_PROGRAM_PROFILE.validate_launch(profile, launch, correctness)
        def fail(): raise RuntimeError('injected device failure')
        with patch.object(torch.cuda, 'synchronize'), patch(
                'open_cake_ir.evaluation.metax_program_profile.activity_collector', return_value=collector):
            with self.assertRaisesRegex(RuntimeError, 'injected device failure') as caught:
                capture_program_activity(fail, candidate=self.candidate, admission=self.admission, activity_library='fixture')
        self.assertEqual(json.loads(caught.exception.artifact_payloads['program_activity'])['activity'], self.raw['activity'])
        bad = deepcopy(self.raw['activity']); bad['records'][0]['name'] = 'unexpected'
        collector.finish = lambda: bad
        with patch('open_cake_ir.tasks.program_evaluation.LoadedTorchTensorInputs', Loaded), patch.object(
                torch.cuda, 'synchronize'), patch('open_cake_ir.evaluation.metax_program_profile.activity_collector', return_value=collector):
            with self.assertRaises(RunProtocolFault) as caught:
                tool.profile_program(self.candidate, self.workload, protocol, self.admission, prepared,
                                     {'activity_library': 'CPU collector fixture'})
        self.assertIn('instrumented_correctness_output', caught.exception.artifact_payloads)
        self.assertIn('preflight_correctness_output', caught.exception.artifact_payloads)
