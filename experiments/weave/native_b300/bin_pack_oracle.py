"""Post-lease CPU oracle for BF16 expert-bin rows and route-key ownership."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


R, T, K, E, H = 4, 512, 8, 128, 2048
LOCAL_E, MAX_ROWS = E // R, R * T
ROUTES = R * T * K


def verify_rows(root: Path, *, observed_global_ids: bool = True,
                input_root: Path | None = None) -> list[dict]:
    source = root if input_root is None else input_root
    hidden_bytes = (source / 'hidden.bf16').read_bytes()
    ids_bytes = (source / 'expert_ids.i32').read_bytes()
    if len(hidden_bytes) != R*T*H*2 or len(ids_bytes) != ROUTES*4:
        raise ValueError('source BF16 row or expert-id extent differs')
    hidden = np.frombuffer(hidden_bytes, dtype='<u2').reshape(R*T, H)
    ids = np.frombuffer(ids_bytes, dtype='<i4').reshape(R*T, K)
    output = root / 'device_outputs'
    if ((output / 'observed_hidden.bf16').read_bytes() != hidden_bytes
            or (observed_global_ids and
                (output / 'observed_expert_ids.i32').read_bytes() != ids_bytes)):
        raise ValueError('device modified a declared source input')
    all_keys = []
    reports = []
    for owner in range(R):
        row = output / f'owner{owner}'
        status = json.loads((row / 'device.json').read_text())
        counts_bytes = (row / 'counts.u32').read_bytes()
        keys_bytes = (row / 'keys.i32').read_bytes()
        rows_bytes = (row / 'rows.bf16').read_bytes()
        if len(counts_bytes) != LOCAL_E*4 or len(keys_bytes) != LOCAL_E*MAX_ROWS*4:
            raise ValueError(f'owner {owner} count or key extent differs')
        counts = np.frombuffer(counts_bytes, dtype='<u4')
        keys = np.frombuffer(keys_bytes, dtype='<i4').reshape(LOCAL_E, MAX_ROWS)
        rows = np.frombuffer(rows_bytes, dtype='<u2')
        selected = ids.ravel()[(ids.ravel() // LOCAL_E) == owner]
        expected = np.bincount(selected - owner*LOCAL_E, minlength=LOCAL_E)
        if (status['owner_rank'] != owner or status['error_flag'] != 0
                or not np.array_equal(counts, expected)
                or status['used_rows'] != int(counts.sum())
                or rows.size != int(counts.sum())*H):
            raise ValueError(f'owner {owner} count, error or packed-row extent differs')
        cursor = 0
        for local_expert, raw_count in enumerate(counts):
            count = int(raw_count)
            if not np.all(keys[local_expert, count:] == -1):
                raise ValueError(f'owner {owner} expert {local_expert} wrote unused keys')
            for index in range(count):
                key = int(keys[local_expert, index])
                if not 0 <= key < ROUTES:
                    raise ValueError(f'owner {owner} invalid return key {key}')
                if int(ids[key // K, key % K]) != owner*LOCAL_E+local_expert:
                    raise ValueError(f'owner {owner} expert {local_expert} key {key} is misrouted')
                if not np.array_equal(rows[(cursor+index)*H:(cursor+index+1)*H],
                                      hidden[key // K]):
                    raise ValueError(f'owner {owner} expert {local_expert} key {key} BF16 row differs')
                all_keys.append(key)
            cursor += count
        reports.append({'owner_rank': owner, 'routes': int(counts.sum()),
                        'experts_nonempty': int(np.count_nonzero(counts)),
                        'min_expert_rows': int(counts.min()),
                        'max_expert_rows': int(counts.max())})
    if len(all_keys) != ROUTES or set(all_keys) != set(range(ROUTES)):
        raise ValueError('route keys are duplicated or missing across owners')
    return reports
