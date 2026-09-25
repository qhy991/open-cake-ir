"""CPU witness for expert-owned tile tasks under a temporal dispatch order.

This does not prescribe a GPU arrival order or implement the ranked mailbox.
It identifies the task/return ownership and padding a 128-row tensor-core
worker would need for one concrete set of EP routes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import ceil
from pathlib import Path

import numpy as np


def analyze(ids_by_rank: list[np.ndarray], *, experts: int,
            tile_rows: int, source_chunk_tokens: int) -> dict:
    if (not ids_by_rank or type(experts) is not int or experts < 1
            or experts % len(ids_by_rank)
            or type(tile_rows) is not int or tile_rows < 1
            or type(source_chunk_tokens) is not int or source_chunk_tokens < 1):
        raise ValueError('world, experts, tile rows or source chunk differs')
    shape = ids_by_rank[0].shape
    if len(shape) != 2 or min(shape) < 1 or shape[1] > experts:
        raise ValueError('route tensors need positive [tokens, top_k] extents')
    for rank, ids in enumerate(ids_by_rank):
        if ids.shape != shape or ids.dtype.kind not in 'iu':
            raise ValueError(f'rank {rank} route shape or integer dtype differs')
        if np.any(ids < 0) or np.any(ids >= experts):
            raise ValueError(f'rank {rank} expert id outside declared range')
        if any(len(set(map(int, row))) != shape[1] for row in ids):
            raise ValueError(f'rank {rank} repeats an expert within a token')

    world, (tokens, routes) = len(ids_by_rank), shape
    experts_per_rank = experts // world
    expected = {(rank, token, route)
                for rank in range(world) for token in range(tokens)
                for route in range(routes)}
    pending: list[list[tuple[int, int, int]]] = [[] for _ in range(experts)]
    tile_number = [0] * experts
    tasks = []
    full_by_step = []
    direct_tasks = 0
    for chunk_index, start in enumerate(range(0, tokens, source_chunk_tokens)):
        stop = min(start + source_chunk_tokens, tokens)
        for rank, ids in enumerate(ids_by_rank):
            direct_counts = Counter(map(int, ids[start:stop].flat))
            direct_tasks += sum(ceil(count / tile_rows)
                                for count in direct_counts.values())
            full = 0
            for token in range(start, stop):
                for route in range(routes):
                    expert = int(ids[token, route])
                    pending[expert].append((rank, token, route))
                    if len(pending[expert]) == tile_rows:
                        tasks.append({
                            'owner_rank': expert // experts_per_rank,
                            'expert_id': expert,
                            'tile_index': tile_number[expert],
                            'valid_rows': tile_rows,
                            'published_after': [rank, chunk_index],
                            'rows': pending[expert],
                        })
                        tile_number[expert] += 1
                        pending[expert] = []
                        full += 1
            full_by_step.append({'source_rank': rank,
                                 'source_chunk': chunk_index,
                                 'full_tiles_published': full})
    partial = 0
    for expert, rows in enumerate(pending):
        if rows:
            tasks.append({
                'owner_rank': expert // experts_per_rank,
                'expert_id': expert,
                'tile_index': tile_number[expert],
                'valid_rows': len(rows),
                'published_after': 'all_dispatch_done',
                'rows': rows,
            })
            partial += 1
    observed = [tuple(row) for task in tasks for row in task['rows']]
    if len(observed) != len(expected) or set(observed) != expected:
        raise AssertionError('tile tasks lost or duplicated a route')
    for task in tasks:
        if (task['valid_rows'] != len(task['rows'])
                or not 1 <= task['valid_rows'] <= tile_rows
                or any(int(ids_by_rank[r][t, k]) != task['expert_id']
                       for r, t, k in task['rows'])):
            raise AssertionError('tile owner or valid-row extent differs')
    owner_counts = [sum(task['owner_rank'] == rank for task in tasks)
                    for rank in range(world)]
    owner_routes = [sum(task['valid_rows'] for task in tasks
                        if task['owner_rank'] == rank)
                    for rank in range(world)]
    total_routes = len(expected)
    # For A routes spread over at most E local experts, sum ceil(n_e/M)
    # is at most min(A, ceil(A/M)+E-1). A cannot exceed all routes here.
    capacity_per_owner = min(total_routes,
                             ceil(total_routes / tile_rows) + experts_per_rank - 1)
    if any(count > capacity_per_owner for count in owner_counts):
        raise AssertionError('declared tile queue upper bound was violated')
    return {
        'schema_version': 1,
        'policy': 'append routes by source chunk then rank; publish full expert tiles immediately and partial tiles after all dispatch',
        'geometry': {'world_size': world, 'tokens_per_rank': tokens,
                     'routes_per_token': routes, 'experts': experts,
                     'experts_per_rank': experts_per_rank,
                     'tile_rows': tile_rows,
                     'source_chunk_tokens': source_chunk_tokens},
        'summary': {
            'routes': total_routes,
            'tile_tasks': len(tasks),
            'full_tiles_published_during_dispatch': len(tasks) - partial,
            'terminal_partial_tiles': partial,
            'tile_padding_rows': len(tasks) * tile_rows - total_routes,
            'owner_routes': owner_routes,
            'owner_tile_tasks': owner_counts,
            'safe_tile_queue_capacity_per_owner': capacity_per_owner,
            'direct_per_source_chunk_tile_tasks': direct_tasks,
            'direct_per_source_chunk_padding_rows': direct_tasks * tile_rows - total_routes,
        },
        'full_tile_publication': full_by_step,
        'tasks': tasks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--world-size', type=int, required=True)
    parser.add_argument('--experts', type=int, required=True)
    parser.add_argument('--tile-rows', type=int, required=True)
    parser.add_argument('--source-chunk-tokens', type=int, required=True)
    args = parser.parse_args()
    ids = []
    for rank in range(args.world_size):
        with np.load(args.input_root / f'rank{rank}-input.npz') as snapshot:
            ids.append(snapshot['ids'].copy())
    plan = analyze(ids, experts=args.experts, tile_rows=args.tile_rows,
                   source_chunk_tokens=args.source_chunk_tokens)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(plan, stream, separators=(',', ':'))
        stream.write('\n')
    print(json.dumps(plan['summary'], indent=2))


if __name__ == '__main__':
    main()
