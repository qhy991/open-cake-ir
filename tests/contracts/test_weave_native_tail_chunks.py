"""The CUDA tail helper agrees with the Workload-derived uneven chunks."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

from experiments.weave.dispatch_ledger import dispatch_ledger
from experiments.weave.rank_plan import rank_plans


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / 'experiments/weave/native_b300/chunk_math.hpp'


class WeaveNativeTailChunks(unittest.TestCase):
    def test_host_compiled_chunk_math_matches_workload_rank_plan(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('host C++ compiler unavailable')
        source = r'''
#include "chunk_math.hpp"
#include <iostream>
template<int Tokens> void rows() {
    for (int chunks = 1; chunks <= Tokens; ++chunks) {
        std::cout << Tokens << ' ' << chunks << ' ';
        for (int token = 0; token < Tokens; ++token)
            std::cout << cake_weave::chunk_for_token<Tokens>(token, chunks) << ',';
        std::cout << ' ';
        for (int chunk = 0; chunk < chunks; ++chunk)
            std::cout << cake_weave::chunk_size<Tokens>(chunk, chunks) << ',';
        std::cout << '\n';
    }
}
int main() { rows<7>(); rows<8>(); }
'''
        with tempfile.TemporaryDirectory(prefix='cake-weave-chunks-') as directory:
            path = Path(directory)
            (path / 'test.cpp').write_text(source)
            binary = path / 'test'
            subprocess.run([compiler, '-std=c++17', '-I', str(HEADER.parent),
                            str(path / 'test.cpp'), '-o', str(binary)],
                           check=True, capture_output=True, text=True, timeout=20)
            output = subprocess.run([str(binary)], check=True, capture_output=True,
                                    text=True, timeout=10).stdout
        for line in output.splitlines():
            tokens_text, count_text, token_map, sizes = line.split()
            tokens, count = int(tokens_text), int(count_text)
            shape = {'R': 4, 'T': tokens, 'E': 8, 'K': 2}
            ids = [[[0, 1] for _ in range(tokens)] for _ in range(4)]
            ledger = dispatch_ledger(shape, ids)
            plans = rank_plans(shape, ledger, sm_count=148,
                               communication_ctas=(12,) * 4,
                               chunks=(count,) * 4,
                               steal_budgets=(0,) * 4)
            chunks = plans[0].chunks
            expected_map = [next(index for index, chunk in enumerate(chunks)
                                 if chunk.first_token <= token < chunk.stop_token)
                            for token in range(tokens)]
            expected_sizes = [chunk.stop_token - chunk.first_token for chunk in chunks]
            with self.subTest(tokens=tokens, chunks=count):
                self.assertEqual([int(value) for value in token_map.rstrip(',').split(',')],
                                 expected_map)
                self.assertEqual([int(value) for value in sizes.rstrip(',').split(',')],
                                 expected_sizes)


if __name__ == '__main__':
    unittest.main()
