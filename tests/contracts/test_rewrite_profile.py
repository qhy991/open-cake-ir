import csv
import io
from types import SimpleNamespace
import unittest

from tools.compare_flashinfer_reference import LoadedCallable
from tools.compare_rewrite_artifacts import reference_arguments
from tools.profile_rewrite_artifacts import METRICS, parse_metrics
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.workloads import create_task


class ExternalDtypeTests(unittest.TestCase):
    def test_external_input_snapshot_preserves_native_array_bit_checks(self):
        from array import array
        from open_cake_ir.evaluation.core import _same_tensor_inputs
        class HostTensor:
            def __init__(self, values): self.values = values
            def cpu(self): return self
            def flatten(self): return self
            def tolist(self): return list(self.values)
        loaded = object.__new__(LoadedCallable)
        for values, expected in [([1.0,-0.0],True), ([1.0,0.0],False), ([2.0,-0.0],False)]:
            observed, after = loaded.snapshot({'result':HostTensor([1.0]),
                'inputs':{'x':HostTensor(values)}})
            self.assertEqual(observed,{'out':[1.0]})
            self.assertIsInstance(after['x'],array)
            self.assertEqual(_same_tensor_inputs({'x':array('d',[1.0,-0.0])},after),expected)

    def test_fp16_gemm_and_bf16_norm_keep_their_abi(self):
        class Tensor:
            counter = 0
            def __init__(self, dtype):
                Tensor.counter += 1
                self.pointer = Tensor.counter
                self.dtype = dtype
                self.device = SimpleNamespace(type='cuda', index=0)
            def reshape(self, shape):
                self.shape = shape
                return self
            def clone(self):
                return Tensor(self.dtype).reshape(self.shape)
            def data_ptr(self):
                return self.pointer
            def is_contiguous(self):
                return True
        torch = SimpleNamespace(Tensor=Tensor, bfloat16='bf16', float16='fp16',
            tensor=lambda values, dtype, device: Tensor(dtype),
            full=lambda shape, value, dtype, device: Tensor(dtype).reshape(shape))
        for task, rows, columns, dtype, names in [
            ('fib_gemm_n128_k2048', 1, 128, 'fp16', ['a', 'b']),
            ('fib_rmsnorm_h2048', 79, 2048, 'bf16', ['x', 'weight'])]:
            with self.subTest(task=task):
                doc, _ = create_task(task, backend='triton-b300', rows=rows, columns=columns)
                workload = WorkloadContract(doc)
                self.assertEqual(reference_arguments(workload), names)
                loaded = LoadedCallable(workload, {n: [1] for n in names}, lambda v, out: out, torch)
                arguments = loaded.fresh_argument_sets(1)[0]
                self.assertTrue(all(v.dtype == dtype for v in arguments['inputs'].values()))
                loaded.launch(arguments)
                self.assertEqual(arguments['result'].dtype, dtype)
                loaded.launch_function = lambda v, out: Tensor('bf16' if dtype == 'fp16' else 'fp16').reshape(out.shape)
                with self.assertRaisesRegex(ValueError, 'output ABI'):
                    loaded.launch(arguments)
        doc, _ = create_task('rmsnorm', backend='triton-b300', rows=1, columns=128)
        with self.assertRaisesRegex(ValueError, 'ABI differs'):
            reference_arguments(WorkloadContract(doc))


class ProfileCsvTests(unittest.TestCase):
    def csv(self, rows):
        stream = io.StringIO()
        writer = csv.writer(stream)
        writer.writerow(['ID', 'Kernel Name', 'Metric Name', 'Metric Unit', 'Metric Value'])
        writer.writerows(rows)
        return '==PROF== capture\n' + stream.getvalue()

    def test_multiple_launches_remain_separate_and_incomplete_metrics_refuse(self):
        rows = [[str(i), 'kernel_' + str(i), metric,
                 '' if metric in {'launch__block_size', 'launch__grid_size'} else 'count', '1']
                for i in range(2) for metric in METRICS]
        self.assertEqual(len(parse_metrics(self.csv(rows))), 2)
        for broken in [rows[:-1], rows + [rows[0]],
                       [r[:-1] + ['nan'] if j == 0 else r for j, r in enumerate(rows)],
                       [r[:-2] + ['', '1'] if j == 0 else r for j, r in enumerate(rows)]]:
            with self.subTest(broken=broken[-1]), self.assertRaises(ValueError):
                parse_metrics(self.csv(broken))


class PhasedProfileTests(unittest.TestCase):
    def test_replay_checks_every_output_and_input_after_device_capture(self):
        from array import array
        import json
        from pathlib import Path
        import sys
        import tempfile
        from unittest.mock import patch
        from tools import profile_rewrite_artifacts as profile
        from tools.check_alignment_candidate import prepare_cases, case_data
        from tools.staged_rewrite_comparison import SnapshotEncoder, INPUT_REFERENCES
        document,_ = create_task('rmsnorm', backend='triton-b300', rows=1, columns=4)
        workload = WorkloadContract(document)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve();prepared = root/'prepare';captured = root/'profile'
            prepared.mkdir();captured.mkdir();prepare_cases(workload, prepared)
            metadata = {'reference':{'kind':'python'}}
            (prepared/'preparation.json').write_text(json.dumps(metadata))
            raw = {'capture_complete':True, 'source_commit':'frozen',
                   'workload_sha256':workload.canonical_sha256, 'reference':metadata['reference'],
                   'participants':{'optimized':'optimized','starter':'starter'},
                   'roles':{r:{'returncode':0} for r in profile.ROLES}, 'ncu_version':'fixture'}
            (captured/'capture.json').write_text(json.dumps(raw))
            plan = profile.snapshot_plan(workload)
            for role in profile.ROLES:
                d = captured/role;d.mkdir()
                child = {'capture_complete':True,'role':role,'workload_sha256':workload.canonical_sha256,
                         'byteorder':sys.byteorder,'snapshot_encoding':INPUT_REFERENCES,'observations':plan}
                (d/'child.json').write_text(json.dumps(child))
                metrics = [[str(i),'kernel_'+str(i),m,
                            '' if m in {'launch__block_size','launch__grid_size'} else 'count','1']
                           for i in range(2) for m in profile.METRICS]
                (d/'stdout.log').write_text(ProfileCsvTests().csv(metrics))
                for index,row in enumerate(plan):
                    case=row['case'];number=workload.case_ids.index(case)
                    values={**case_data(prepared,workload,number,'input'),**case_data(prepared,workload,number,'output')}
                    with (d/f'{index}.bin').open('wb') as stream:
                        encoder=SnapshotEncoder(stream,workload,profile.snapshot_bytes(workload,case))
                        encoder.append(case,[array('f',values[arg.name]).tobytes() for arg in workload.tensor_abi(case)])
            serial=0
            def verify():
                nonlocal serial
                out=root/str(serial);out.mkdir();serial+=1
                return profile.verify(root,out,prepared,captured,workload,'frozen')
            with patch.object(profile,'load_participants',return_value=({'optimized':'optimized','starter':'starter'},{})), \
                 patch.object(profile,'candidate_identity',side_effect=lambda c:c):
                report=verify();self.assertTrue(report['correctness_passed'])
                self.assertTrue(all(len(r['checks'])==6 and len(r['kernels'])==2 for r in report['roles'].values()))
                last=captured/'external'/f'{len(plan)-1}.bin';valid=last.read_bytes()
                # A changed input in the profiled call fails even with correct outputs.
                changed=bytearray(valid);changed[1:5]=array('f',[999.0]).tobytes();last.write_bytes(changed)
                report=verify();self.assertFalse(report['correctness_passed'])
                self.assertTrue(all('kernels' not in r for r in report['roles'].values()))
                # The final output of the final call is checked too.
                changed=bytearray(valid);changed[-4:]=array('f',[999.0]).tobytes();last.write_bytes(changed)
                self.assertFalse(verify()['correctness_passed'])
                for broken,diagnostic in [(valid[:-1],'truncated'),(valid+b'x','trailing')]:
                    last.write_bytes(broken)
                    with self.assertRaisesRegex(ValueError,diagnostic):verify()
                last.write_bytes(valid)
                child_file=captured/'external'/'child.json';child=json.loads(child_file.read_text());child['observations']=plan[:-1];child_file.write_text(json.dumps(child))
                with self.assertRaisesRegex(ValueError,'required observations'):verify()

    def test_profile_phases_do_not_allow_a_cpu_stage_to_hold_a_lease(self):
        from tools.check_alignment_candidate import phase_contract
        task={'stages':[{'id':name,'execution':mode,'judge':{'identity':'open-cake-ir@fixed'}}
                        for name,mode in [('prepare','local'),('profile','broker'),('verify','local')]]}
        phase_contract('fixed',task,gpu_stage='profile')
        task['stages'][2]['execution']='broker'
        with self.assertRaisesRegex(ValueError,'resource phase'):
            phase_contract('fixed',task,gpu_stage='profile')


if __name__ == '__main__':
    unittest.main()
