"""The CUDA tail helper agrees with the Workload-derived uneven chunks."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / 'experiments/weave/native_b300/chunk_math.hpp'
WORKLOAD = ROOT / 'contracts/workloads/weave-ep4-bf16-moe-b300-v1.json'


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
        declared_tokens = {case['shape']['T'] for case in
                           json.loads(WORKLOAD.read_text())['cases']}
        self.assertEqual(declared_tokens, {7, 8})
        for line in output.splitlines():
            tokens_text, count_text, token_map, sizes = line.split()
            tokens, count = int(tokens_text), int(count_text)
            self.assertIn(tokens, declared_tokens)
            base, longer = divmod(tokens, count)
            expected_sizes = [base + 1] * longer + [base] * (count - longer)
            expected_map = [index for index, size in enumerate(expected_sizes)
                            for _ in range(size)]
            with self.subTest(tokens=tokens, chunks=count):
                self.assertEqual([int(value) for value in token_map.rstrip(',').split(',')],
                                 expected_map)
                self.assertEqual([int(value) for value in sizes.rstrip(',').split(',')],
                                 expected_sizes)


if __name__ == '__main__':
    unittest.main()
