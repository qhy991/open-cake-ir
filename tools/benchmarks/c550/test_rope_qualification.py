"""Native return-value bridge controls; original Bench remains the oracle owner."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('rope_qualification', Path(__file__).with_name('qualify_rope.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class Tensor:
    def __init__(self, shape, data): self.shape, self.data = shape, list(data)
    def view(self, dtype): return self
    def cpu(self): return self
    def clone(self): return Tensor(self.shape, self.data)


class ReturnValueBridge(unittest.TestCase):
    def test_only_a_native_launch_supplies_output_and_every_path_closes(self):
        torch = SimpleNamespace(uint8='uint8', bfloat16='bf16',
            full=lambda shape, value, **kwargs: Tensor(shape, [value]),
            equal=lambda a, b: a.data == b.data,
            cuda=SimpleNamespace(synchronize=lambda: None, current_stream=lambda: SimpleNamespace(cuda_stream=7)))
        for mutate in (False, True):
            with self.subTest(mutate=mutate), tempfile.TemporaryDirectory() as directory:
                runtime = probe.NativeRope.__new__(probe.NativeRope)
                runtime.root, runtime.trace = Path(directory), Path(directory) / 'calls.jsonl'
                runtime.trace.touch()
                runtime.cases, runtime.admission = {(1, 2): 'original'}, object()
                positions, frequency = Tensor((1, 2), [3, 9]), Tensor((64,), [1, 2])
                seen = []
                class Loaded:
                    launch_calls = 0
                    closed = False
                    def launch(self, args, *, tensor_contract, stream):
                        self.launch_calls += 1
                        seen.append((args, stream))
                        args[-1].data[:] = ['native result']
                        if mutate: args[0].data[0] = -1
                    def close(self, *, synchronize, primary=None):
                        synchronize(); self.closed = True
                        if primary: raise primary
                loaded = Loaded()
                with patch.dict('sys.modules', {'torch': torch}), \
                     patch.object(probe, 'read_candidate', return_value=(object(), object())), \
                     patch('open_cake_ir.evaluation.metax_driver.LoadedMetaxCandidate.load', return_value=loaded):
                    if mutate:
                        with self.assertRaisesRegex(ValueError, 'changed an original input'):
                            runtime.run(positions, frequency, 1.0)
                    else:
                        out = runtime.run(positions, frequency, 1.0)
                        self.assertEqual(out.data, ['native result'])
                self.assertIs(seen[0][0][0], positions)
                self.assertIs(seen[0][0][1], frequency)
                self.assertEqual(seen[0][1], 7)
                record = json.loads(runtime.trace.read_text())
                self.assertTrue(record['module_closed'])
                self.assertEqual(record['kernel_calls'], 1)
                self.assertEqual(record['input_unchanged'], not mutate)

    def test_original_shape_and_scalar_cannot_select_another_variant(self):
        runtime = probe.NativeRope.__new__(probe.NativeRope)
        runtime.cases = {(1, 2): 'original'}
        with patch.dict('sys.modules', {'torch': SimpleNamespace()}), \
             patch.object(probe, 'read_candidate') as load:
            for shape, scalar in (((2, 1), 1.0), ((1, 2), 2.0), ((1, 2), True)):
                with self.assertRaisesRegex(ValueError, 'original scalar'):
                    runtime.run(Tensor(shape, []), Tensor((64,), []), scalar)
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
