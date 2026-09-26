"""CPU-only FP64 oracle for changed EP4 routes and unchanged model inputs.

`data.py` is copied from the pinned independent open-baseline task into the
create-only oracle root. This module calls that reference, not Cake lowering.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from data import reference


R, T, K, E, H = 4, 512, 8, 128, 2048


def prepare(root: Path, baseline: Path, bin_root: Path,
            bridge: Path, ids_path: Path) -> None:
    if os.environ.get('GPUQ_JOB_ID'):
        raise RuntimeError('FP64 oracle must be prepared outside a GPU lease')
    if (root / 'expected_output.npy').exists() or (root / 'oracle_report.json').exists():
        raise ValueError('changed-route oracle must be create-only')
    if not (root / 'data.py').is_file():
        raise ValueError('pinned independent reference source is missing')
    ids = np.fromfile(ids_path, dtype='<i4')
    if ids.size != R*T*K:
        raise ValueError('changed expert IDs differ from EP4/T512/K8')
    ids = ids.reshape(R, T, K)
    if (np.any(ids < 0) or np.any(ids >= E)
            or any(len(set(map(int, row))) != K for rank in ids for row in rank)):
        raise ValueError('changed route IDs violate the distinct expert domain')
    ranks = []
    for rank in range(R):
        with np.load(baseline / f'rank{rank}-input.npz') as saved:
            inputs = {name: saved[name] for name in saved.files}
        if (inputs['hidden'].shape != (T, H)
                or inputs['weights'].shape != (T, K)
                or inputs['gate'].shape != (E//R, 768, H)
                or inputs['up'].shape != (E//R, 768, H)
                or inputs['down'].shape != (E//R, H, 768)):
            raise ValueError('independent rank model extents differ')
        inputs['ids'] = ids[rank]
        ranks.append(inputs)
    hidden = np.stack([rank['hidden'] for rank in ranks]).astype('<f4')
    if np.any(hidden.view('<u4') & 0xffff):
        raise ValueError('reference hidden input is not exactly BF16')
    observed_hidden = np.fromfile(bin_root / 'hidden.bf16', dtype='<u2')
    if (observed_hidden.size != R*T*H
            or not np.array_equal((hidden.view('<u4') >> 16).astype('<u2').ravel(),
                                  observed_hidden)):
        raise ValueError('CPU reference and GPU hidden input differ')
    weights = np.stack([rank['weights'] for rank in ranks]).astype('<f4')
    observed_weights = np.fromfile(bridge / 'route_weights.fp32', dtype='<f4')
    if (observed_weights.size != R*T*K
            or not np.array_equal(weights.ravel().view('<u4'),
                                  observed_weights.view('<u4'))):
        raise ValueError('CPU reference and GPU route weights differ')
    began = time.perf_counter_ns()
    expected = reference({'geometry': {
        'tokens_per_rank': T, 'hidden': H, 'experts': E,
    }}, ranks)
    if (expected.shape != (R, T, H) or expected.dtype != np.float32
            or not np.all(np.isfinite(expected))):
        raise ValueError('changed-route FP64 oracle extent or values differ')
    tolerance = .01 + .01*np.abs(expected)
    if np.count_nonzero(np.abs(expected) > tolerance) <= expected.size//10:
        raise ValueError('all-zero negative control is too weak')
    np.save(root / 'expected_output.npy', expected)
    (root / 'oracle_report.json').write_text(json.dumps({
        'source': 'independent open-baseline data.py reference',
        'geometry': {'R': R, 'T': T, 'K': K, 'E': E, 'H': H},
        'ids_path': str(ids_path.resolve()),
        'baseline_input': str(baseline.resolve()),
        'hidden_input': str(bin_root.resolve()),
        'bridge_weights': str(bridge.resolve()),
        'all_finite': True,
        'nonzero_elements': int(np.count_nonzero(expected)),
        'cpu_oracle_wall_ns': time.perf_counter_ns()-began,
        'scope': 'CPU FP64 FFN and weighted combine; no GPU result or timing',
    }, indent=2) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--bin-root', type=Path, required=True)
    parser.add_argument('--bridge', type=Path, required=True)
    parser.add_argument('--ids-path', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.root.expanduser().resolve(strict=True),
            args.baseline.expanduser().resolve(strict=True),
            args.bin_root.expanduser().resolve(strict=True),
            args.bridge.expanduser().resolve(strict=True),
            args.ids_path.expanduser().resolve(strict=True))


if __name__ == '__main__':
    main()
