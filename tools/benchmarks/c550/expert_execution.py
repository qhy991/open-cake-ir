"""Untuned CAKE implementation of the original routed-expert tensor contract.

Each token/slot computes its own expert contribution. The final stage sums those
contributions in expert-index order, with repeated assignments kept separately.
All arithmetic before the final store is FP32, including the projection inputs.
"""
from __future__ import annotations

TASK = 'L2/024_moe_expert_parallel_execution'
TARGET = 'xcore1002'
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 256
SLOTS = 8


def _source(tokens, hidden, intermediate, experts, slots, *, tile=16, reduction_tile=128):
    """The small dimensions are only for software semantic tests."""
    projection = f'''@cake.schedule(name="expert_intermediate", target="{TARGET}", backend="triton", entry_point="cake_expert_intermediate")
def expert_intermediate(lm, hidden_states: cake.Tensor(({tokens}, {hidden}), "bf16"),
        topk_indices: cake.Tensor(({tokens}, {slots}), "int64"),
        gate_proj_weights: cake.Tensor(({experts}, {intermediate}, {hidden}), "bf16"),
        up_proj_weights: cake.Tensor(({experts}, {intermediate}, {hidden}), "bf16"),
        intermediate_values: cake.Tensor(({tokens}, {slots}, {intermediate}), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    token = lm.program(intermediate_values, axis=0, dimension=0, tile=1)
    slot = lm.program(intermediate_values, axis=1, dimension=1, tile=1)
    column = lm.program(intermediate_values, axis=2, dimension=2, tile={tile})
    with compute:
        expert = lm.load(topk_indices[token, slot], id="load_expert")
'''
    # Static single trips have no carried reduction. This is the same canonical
    # choice used for full-axis reductions elsewhere in the frontend.
    repeated = hidden > reduction_tile
    index = 'k' if repeated else ':'
    prefix = '    ' if repeated else ''
    if repeated:
        projection += f'''    for k in lm.range(hidden_states, name="hidden_loop", dimension=1, tile={reduction_tile}, num_stages=1, loop_unroll_factor=1):
'''
    projection += prefix + '    with compute:\n'
    fold = '' if repeated else ', across_loop=False'
    statements = f'''x_stored = lm.load(hidden_states[token, {index}], id="load_x")
x = lm.cast(x_stored, to="fp32", id="widen_x")
gate_stored = lm.load(gate_proj_weights[lm.scalar_index(expert), column, {index}], id="load_gate")
up_stored = lm.load(up_proj_weights[lm.scalar_index(expert), column, {index}], id="load_up")
gate_weights = lm.cast(gate_stored, to="fp32", id="widen_gate")
up_weights = lm.cast(up_stored, to="fp32", id="widen_up")
gate_products = gate_weights * lm.broadcast(x, axis=1)
up_products = up_weights * lm.broadcast(x, axis=1)
gate = lm.reduce(gate_products, op="sum", axis=1{fold}, id="gate_sum")
up = lm.reduce(up_products, op="sum", axis=1{fold}, id="up_sum")'''
    projection += '\n'.join(prefix + '        ' + line for line in statements.splitlines()) + '\n'
    projection += '''    with compute:
        exponential = lm.exp(gate * -1.0, id="negative_gate_exp")
        sigmoid_gate = gate / (exponential + 1.0)
        product = sigmoid_gate * up
        lm.store(intermediate_values[token, slot, column], product, id="store_intermediate")
'''
    down = f'''@cake.schedule(name="expert_output", target="{TARGET}", backend="triton", entry_point="cake_expert_output")
def expert_output(lm, intermediate_values: cake.Tensor(({tokens}, {slots}, {intermediate}), "fp32"),
        topk_indices: cake.Tensor(({tokens}, {slots}), "int64"),
        topk_weights: cake.Tensor(({tokens}, {slots}), "bf16"),
        down_proj_weights: cake.Tensor(({experts}, {hidden}, {intermediate}), "bf16"),
        contributions: cake.Tensor(({tokens}, {slots}, {hidden}), "fp32", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    token = lm.program(contributions, axis=0, dimension=0, tile=1)
    slot = lm.program(contributions, axis=1, dimension=1, tile=1)
    column = lm.program(contributions, axis=2, dimension=2, tile={tile})
    with compute:
        expert = lm.load(topk_indices[token, slot], id="load_expert")
        route_stored = lm.load(topk_weights[token, slot], id="load_route_weight")
        route_weight = lm.cast(route_stored, to="fp32", id="widen_route_weight")
'''
    repeated = intermediate > reduction_tile
    index = 'k' if repeated else ':'
    prefix = '    ' if repeated else ''
    if repeated:
        down += f'''    for k in lm.range(intermediate_values, name="intermediate_loop", dimension=2, tile={reduction_tile}, num_stages=1, loop_unroll_factor=1):
'''
    down += prefix + '    with compute:\n'
    fold = '' if repeated else ', across_loop=False'
    statements = f'''values = lm.load(intermediate_values[token, slot, {index}], id="load_intermediate")
weights_stored = lm.load(down_proj_weights[lm.scalar_index(expert), column, {index}], id="load_down")
weights = lm.cast(weights_stored, to="fp32", id="widen_down")
products = weights * lm.broadcast(values, axis=1)
projected = lm.reduce(products, op="sum", axis=1{fold}, id="projection_sum")'''
    down += '\n'.join(prefix + '        ' + line for line in statements.splitlines()) + '\n'
    down += '''    with compute:
        weighted = projected * route_weight
        lm.store(contributions[token, slot, column], weighted, id="store_contribution")
'''
    aggregate = f'''@cake.schedule(name="aggregate_experts", target="{TARGET}", backend="triton", entry_point="cake_aggregate_experts")
def aggregate_experts(lm, contributions: cake.Tensor(({tokens}, {slots}, {hidden}), "fp32"),
        topk_indices: cake.Tensor(({tokens}, {slots}), "int64"),
        output: cake.Tensor(({tokens}, {hidden}), "bf16", mode="output")):
    compute = lm.role(execution_groups=[0, 1, 2, 3])
    token = lm.program(output, axis=0, dimension=0, tile=1)
    column = lm.program(output, axis=1, dimension=1, tile={tile})
    with compute:
        expert_ids = lm.load(topk_indices[token, :], id="load_expert_ids")
        slot_ids = lm.coordinate(source="range", start=0, extent={slots}, id="slot_ids")
'''
    for slot in range(slots):
        aggregate += f'''        wanted_{slot} = lm.load(topk_indices[token, {slot}:{slot + 1}], id="wanted_{slot}")
        lower_{slot} = lm.compare(expert_ids, wanted_{slot}, op="lt", id="lower_{slot}")
        same_{slot} = lm.compare(expert_ids, wanted_{slot}, op="eq", id="same_{slot}")
        earlier_{slot} = lm.compare(slot_ids, {slot}, op="lt", id="earlier_{slot}")
        precedes_{slot} = lower_{slot} + same_{slot} * earlier_{slot}
        rank_{slot} = lm.reduce(precedes_{slot}, op="sum", axis=0, across_loop=False, id="rank_{slot}")
        stored_{slot} = lm.load(contributions[token, {slot}:{slot + 1}, column], id="load_contribution_{slot}")
        value_{slot} = lm.reduce(stored_{slot}, op="sum", axis=0, across_loop=False, id="slot_value_{slot}")
'''
    for rank in range(slots):
        for slot in range(slots):
            aggregate += f'''        selected_{rank}_{slot} = lm.compare(rank_{slot}, {rank}, op="eq", id="select_{rank}_{slot}")
        masked_{rank}_{slot} = lm.select(selected_{rank}_{slot}, value_{slot}, 0.0, id="mask_{rank}_{slot}")
'''
        aggregate += f'        contribution_{rank} = ' + ' + '.join(f'masked_{rank}_{slot}' for slot in range(slots)) + '\n'
        if rank:
            previous = 'contribution_0' if rank == 1 else f'accumulated_{rank - 1}'
            aggregate += f'        accumulated_{rank} = {previous} + contribution_{rank}\n'
    final = 'contribution_0' if slots == 1 else f'accumulated_{slots - 1}'
    aggregate += f'''        rounded = lm.cast({final}, to="bf16", id="final_rounding")
        lm.store(output[token, column], rounded, id="store_output")
'''
    inputs = ('hidden_states', 'topk_indices', 'topk_weights', 'gate_proj_weights', 'up_proj_weights', 'down_proj_weights')
    stage_args = (
        ('expert_intermediate', ('hidden_states', 'topk_indices', 'gate_proj_weights', 'up_proj_weights', 'intermediate_values')),
        ('expert_output', ('intermediate_values', 'topk_indices', 'topk_weights', 'down_proj_weights', 'contributions')),
        ('aggregate_experts', ('contributions', 'topk_indices', 'output')),
    )
    stages = ',\n'.join(f'    cake.stage(name={name!r}, schedule={name}, bindings={dict(zip(args, args))!r})'
                         for name, args in stage_args)
    return ('from open_cake_ir.compiler import frontend as cake\n\n' + projection + '\n' + down + '\n' + aggregate
            + f'\ncake.program(program_id="c550_expert_execution_t{tokens}", inputs={inputs!r}, outputs=("output",), stages=(\n{stages},\n))\n')


def source_for(num_tokens: int) -> str:
    if type(num_tokens) is not int or num_tokens <= 0:
        raise ValueError('Expert execution requires a positive original token count')
    return _source(num_tokens, HIDDEN, INTERMEDIATE, EXPERTS, SLOTS)


def program_for(num_tokens: int):
    from open_cake_ir.compiler.program_frontend import parse_program
    return parse_program(source_for(num_tokens)).program


def source_for_workload(workload, case_id: str) -> str:
    abi = workload.tensor_abi(case_id)
    if workload.target != TARGET or len(abi) != 7 or len(abi[0].shape) != 2:
        raise ValueError('Expert execution target or public argument count differs')
    tokens, _ = abi[0].shape
    expected = (
        ('hidden_states', (tokens, HIDDEN), 'bf16', 'input'),
        ('topk_indices', (tokens, SLOTS), 'int64', 'input'),
        ('topk_weights', (tokens, SLOTS), 'bf16', 'input'),
        ('gate_proj_weights', (EXPERTS, INTERMEDIATE, HIDDEN), 'bf16', 'input'),
        ('up_proj_weights', (EXPERTS, INTERMEDIATE, HIDDEN), 'bf16', 'input'),
        ('down_proj_weights', (EXPERTS, HIDDEN, INTERMEDIATE), 'bf16', 'input'),
        ('output', (tokens, HIDDEN), 'bf16', 'output'),
    )
    if tuple((arg.name, tuple(arg.shape), arg.dtype, arg.mode) for arg in abi) != expected:
        raise ValueError('Expert execution original ordered ABI differs')
    return source_for(tokens)
