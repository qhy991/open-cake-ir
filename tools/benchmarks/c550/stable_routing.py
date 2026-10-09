"""Untuned counted-rank CAKE baseline for the original stable-routing ABI.

Every writer owns its output coordinate. No input flattening, scatter, sorting
library, Torch arithmetic, or new IR operation is part of the candidate.
"""
from __future__ import annotations

TASK = 'L1/058_moe_expert_token_radix_sort_with_prefix_sum'
TARGET = 'xcore1002'
EXPERTS = 256
ASSIGNMENTS_PER_TOKEN = 8
SCAN_ROWS = 128


def _access(buffer, batch, sequence):
    batch_index = 'scan_batch' if batch > 1 else '0:1'
    row_index = 'scan_row' if sequence > SCAN_ROWS else ':'
    return f'{buffer}[{batch_index}, {row_index}, :]'


def _scan(buffer, batch, sequence, body, output, destination):
    """Place each coordinate/fold in its owner loop; omit static single trips."""
    rows = []
    outer = ''
    if batch > 1:
        rows += [f'for scan_batch in lm.range({buffer}, name="scan_batches", dimension=0, tile=1,',
                 '                          num_stages=1, loop_unroll_factor=1):']
        outer = '    '
    batch_coordinate = ('source="loop_tile", name="scan_batch"' if batch > 1 else
                        'source="range", start=0, extent=1')
    rows += [outer + 'with compute:', outer + '    batch_indices = lm.coordinate('
             + batch_coordinate + ', id="batch_indices")']
    inner = outer
    if sequence > SCAN_ROWS:
        rows += [outer + f'for scan_row in lm.range({buffer}, name="scan_rows", dimension=1, tile={SCAN_ROWS},',
                 outer + '                        num_stages=1, loop_unroll_factor=1):']
        inner += '    '
    row_coordinate = ('source="loop_tile", name="scan_row"' if sequence > SCAN_ROWS else
                      f'source="range", start=0, extent={sequence}')
    statements = [f'row_indices = lm.coordinate({row_coordinate}, id="row_indices")',
                  f'slot_indices = lm.coordinate(source="range", start=0, extent={ASSIGNMENTS_PER_TOKEN}, id="slot_indices")',
                  f'valid_rows = lm.compare(row_indices, {sequence}, op="lt", id="valid_rows")']
    statements += [line.strip() for line in body.strip().splitlines()]
    row_scope = '' if sequence > SCAN_ROWS else ', across_loop=False'
    statements += ['valid_contributions = contributions * lm.broadcast(valid_rows, axis=1)',
                   'per_row = lm.reduce(valid_contributions, op="sum", axis=2, across_loop=False, id="sum_slots")',
                   f'per_batch = lm.reduce(per_row, op="sum", axis=1{row_scope}, id="sum_rows")']
    rows += [inner + 'with compute:'] + [inner + '    ' + line for line in statements]
    batch_scope = '' if batch > 1 else ', across_loop=False'
    rows += [outer + 'with compute:',
             outer + f'    total = lm.reduce(per_batch, op="sum", axis=0{batch_scope}, id="sum_batches")',
             'with compute:',
             f'    lm.store({output}[{destination}], total, coalesced=False, id="store_result")']
    return '\n'.join('    ' + line for line in rows) + '\n'


def source_for(batch_size: int, seq_len: int) -> str:
    if any(type(value) is not int or value <= 0 for value in (batch_size, seq_len)):
        raise ValueError('Stable routing requires positive original batch and sequence dimensions')
    count = batch_size * seq_len * ASSIGNMENTS_PER_TOKEN
    if count > 2**31 - 1:
        raise ValueError('Stable routing assignment count exceeds its exact INT32 result domain')
    shape = (batch_size, seq_len, ASSIGNMENTS_PER_TOKEN)
    keys_access = _access('topk_idx', batch_size, seq_len)
    ranks_access = _access('ranks', batch_size, seq_len)
    rank = f'''@cake.schedule(name="stable_rank", target="xcore1002", backend="triton", entry_point="cake_stable_rank")
def stable_rank(lm, topk_idx: cake.Tensor({shape!r}, "int32"), ranks: cake.Tensor({shape!r}, "int32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    batch = lm.program(topk_idx, axis=0, dimension=0, tile=1)
    row = lm.program(topk_idx, axis=1, dimension=1, tile=1)
    slot = lm.program(topk_idx, axis=2, dimension=2, tile=1)
    with compute:
        wanted = lm.load(topk_idx[batch, row, slot], id="load_wanted")
        wanted_batch = lm.coordinate(source="program", name="batch", id="wanted_batch")
        wanted_row = lm.coordinate(source="program", name="row", id="wanted_row")
        wanted_slot = lm.coordinate(source="program", name="slot", id="wanted_slot")
''' + _scan('topk_idx', batch_size, seq_len, f'''
keys = lm.load({keys_access}, id="load_keys")
lower_key = lm.compare(keys, wanted, op="lt", id="lower_key")
equal_key = lm.compare(keys, wanted, op="eq", id="equal_key")
earlier_batch = lm.compare(batch_indices, wanted_batch, op="lt", id="earlier_batch")
same_batch = lm.compare(batch_indices, wanted_batch, op="eq", id="same_batch")
earlier_row = lm.compare(row_indices, wanted_row, op="lt", id="earlier_row")
same_row = lm.compare(row_indices, wanted_row, op="eq", id="same_row")
earlier_slot = lm.compare(slot_indices, wanted_slot, op="lt", id="earlier_slot")
zeros = keys * 0
prior_rows = zeros + lm.broadcast(earlier_row, axis=1)
equal_rows = zeros + lm.broadcast(same_row, axis=1)
prior_slots = zeros + lm.broadcast(earlier_slot, axis=2)
prior_in_batch = prior_rows + equal_rows * prior_slots
earlier_position = earlier_batch + prior_in_batch * same_batch
contributions = lower_key + equal_key * earlier_position
''', 'ranks', 'batch, row, slot')
    inverse = f'''@cake.schedule(name="inverse_rank", target="xcore1002", backend="triton", entry_point="cake_inverse_rank")
def inverse_rank(lm, ranks: cake.Tensor({shape!r}, "int32"), sorted_token_indices: cake.Tensor(({count},), "int32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    position = lm.program(sorted_token_indices, axis=0, dimension=0, tile=1)
    with compute:
        wanted_position = lm.coordinate(source="program", name="position", id="wanted_position")
''' + _scan('ranks', batch_size, seq_len, f'''
values = lm.load({ranks_access}, id="load_ranks")
matches = lm.compare(values, wanted_position, op="eq", id="matches")
zeros = values * 0
batch_base = batch_indices * {seq_len * ASSIGNMENTS_PER_TOKEN}
row_base = row_indices * {ASSIGNMENTS_PER_TOKEN}
within_batch = zeros + lm.broadcast(row_base, axis=1)
with_slots = within_batch + lm.broadcast(slot_indices, axis=2)
flat_indices = with_slots + batch_base
contributions = flat_indices * matches
''', 'sorted_token_indices', 'position')
    offsets = f'''@cake.schedule(name="expert_offsets", target="xcore1002", backend="triton", entry_point="cake_expert_offsets")
def expert_offsets(lm, topk_idx: cake.Tensor({shape!r}, "int32"), expert_offsets: cake.Tensor(({EXPERTS + 1},), "int32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    boundary = lm.program(expert_offsets, axis=0, dimension=0, tile=1)
    with compute:
        expert_boundary = lm.coordinate(source="program", name="boundary", id="expert_boundary")
''' + _scan('topk_idx', batch_size, seq_len, f'''
keys = lm.load({keys_access}, id="load_keys")
contributions = lm.compare(keys, expert_boundary, op="lt", id="count_below_boundary")
''', 'expert_offsets', 'boundary')
    return ('from open_cake_ir.compiler import frontend as cake\n\n' + rank + '\n' + inverse + '\n' + offsets
            + f'''
cake.program(program_id="c550_stable_routing_b{batch_size}_s{seq_len}", inputs=("topk_idx",),
    outputs=("sorted_token_indices", "expert_offsets"), stages=(
        cake.stage(name="stable_rank", schedule=stable_rank,
                   bindings={{"topk_idx": "topk_idx", "ranks": "ranks"}}),
        cake.stage(name="inverse_rank", schedule=inverse_rank,
                   bindings={{"ranks": "ranks", "sorted_token_indices": "sorted_token_indices"}}),
        cake.stage(name="expert_offsets", schedule=expert_offsets,
                   bindings={{"topk_idx": "topk_idx", "expert_offsets": "expert_offsets"}}),
))
''')


def program_for(batch_size: int, seq_len: int):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(batch_size, seq_len)).program


def source_for_workload(workload, case_id: str) -> str:
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 3 or len(abi[0].shape) != 3:
        raise ValueError('Stable-routing target or public argument count differs')
    batch, sequence, _ = abi[0].shape
    expected = (('topk_idx', (batch, sequence, ASSIGNMENTS_PER_TOKEN), 'int32', 'input'),
                ('sorted_token_indices', (batch * sequence * ASSIGNMENTS_PER_TOKEN,), 'int32', 'output'),
                ('expert_offsets', (EXPERTS + 1,), 'int32', 'output'))
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError('Stable-routing original ordered ABI differs')
    return source_for(batch, sequence)
