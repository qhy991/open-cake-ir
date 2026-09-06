from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("empirical_instrument", ROOT / "tools/calibrate_empirical_cost.py")
instrument = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(instrument)


class TraceAttributionTest(unittest.TestCase):
    def fixture(self):
        rows = [{"id": "a", "grid": [2, 1, 1], "profile": {"compiled_resources": {"entry_point": "same_name", "threads_per_cta": 128}}},
                {"id": "b", "grid": [4, 1, 1], "profile": {"compiled_resources": {"entry_point": "same_name", "threads_per_cta": 256}}}]
        events = []
        for position, index in enumerate([0, 1, 1, 0]):
            common = {"device": 0, "context": 1, "stream": 7}
            events.extend([
                {"cat": "kernel", "name": "FillFunctor<unsigned char>", "ts": position * 10, "dur": 1, "args": common},
                {"cat": "kernel", "name": "same_name", "ts": position * 10 + 2, "dur": index + 2, "args": {**common, "correlation": position + 1, "grid": rows[index]["grid"], "block": [rows[index]["profile"]["compiled_resources"]["threads_per_cta"], 1, 1]}},
                {"cat": "cuda_driver", "name": "cuLaunchKernel", "args": {"correlation": position + 1}},
            ])
        return rows, events

    def parse(self, rows, events):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            (stage / "launch-order-0.json").write_text(json.dumps(["a", "b", "b", "a"]))
            (stage / "cupti-trace-0.json").write_text(json.dumps({"traceEvents": events}))
            return instrument._trace_samples(stage, 0, {"sampling": {"rounds": 2}}, rows)

    def test_shared_names_need_order_launch_and_driver_binding(self):
        rows, events = self.fixture()
        self.assertEqual(self.parse(rows, events), [[2, 2], [3, 3]])

    def test_malformed_or_misattributed_records_are_refused(self):
        mutations = [
            lambda events: events.pop(0),
            lambda events: events[0].update(dur=3),
            lambda events: events[1]["args"].update(grid=[4, 1, 1]),
            lambda events: events[1]["args"].update(block=[128, True, 1]),
            lambda events: events[1]["args"].update(correlation=99),
            lambda events: events[1]["args"].update(stream=8),
            lambda events: events[1].update(dur=-1),
            lambda events: events[1].update(ts=float("nan")),
            lambda events: events[2]["args"].update(correlation=2),
        ]
        rows, original = self.fixture()
        for mutation in mutations:
            events = copy.deepcopy(original)
            mutation(events)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):self.parse(rows, events)


if __name__ == "__main__":unittest.main()
