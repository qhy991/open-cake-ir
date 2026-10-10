"""Explicit extension graphs bound to existing task Workloads and their oracles."""
from open_cake_ir.tasks.workloads import create_task

WIDTHS = (1, 2, 4, 8, 16)


def _fixed_loop(source):
    header, body = source.split('    with compute:\n', 1)
    body = body.replace('[row, :]', '[row, column]')
    body = '\n'.join('    ' + line if line else line for line in body.splitlines())
    return (header + '    with compute:\n'
            '        for column in lm.range(x, name="columns", dimension=1, tile=32, num_stages=1):\n'
            + body + '\n')


def _carried_reduction(source):
    old = ('        squares = lm.square(values, id="square")\n'
           '        square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=False, id="sum_square")')
    new = ('        for column in lm.range(x, name="sum_columns", dimension=1, tile=64, num_stages=1):\n'
           '            chunk = lm.load(x[row, column], id="load_chunk")\n'
           '            squares = lm.square(chunk, id="square")\n'
           '            square_sum = lm.reduce(squares, op="sum", axis=0, scope="cta", across_loop=True, id="sum_square")')
    if source.count(old) != 1:
        raise ValueError('The task-owned RMS reduction site differs')
    return source.replace(old, new)


def _mma(source, *, mode, rounded=False):
    header = source.split('    compute =', 1)[0]
    left, right, output = ('a', 'b', 'out') if rounded else ('input', 'weight_nt', 'output')
    depth = 2048 if rounded else 64
    contract = 'triton.dot.fp16_fp32' if rounded else 'triton.dot.fp32_ieee'
    lines = ['    compute = lm.role(execution_groups=[0])',
             f'    row = lm.program({left}, axis=0, dimension=0, tile=16)']
    if mode != 'output_loop':
        lines.append(f'    column = lm.program({right}, axis=1, dimension=0, tile=16)')
    lines.append('    with compute:')
    indent = '        '
    k_index = ':'
    k_tile = depth
    if mode == 'k_loop':
        lines.append(f'        for k in lm.range({left}, name="contract_k", dimension=1, tile=32, num_stages=1):')
        indent = '            '
        k_index = 'k'
        k_tile = 32
    elif mode == 'output_loop':
        lines.append(f'        for column in lm.range({right}, name="output_columns", dimension=0, tile=16, num_stages=1):')
        indent = '            '
    elif mode != 'resident':
        raise ValueError('Unknown MMA qualification graph')
    lines += [indent + f'left = lm.load({left}[row, {k_index}], id="load_left")',
              indent + f'right = lm.load({right}[column, {k_index}], id="load_right")',
              indent + f'acc = lm.mma(left, right, instruction={{"contract": "{contract}"}}, tile_shape=(16,16,{k_tile}), id="dot")']
    if mode == 'k_loop':
        indent = '        '
    value = 'acc'
    if rounded:
        lines.append(indent + 'rounded = lm.cast(acc, to="fp16", id="round_out")')
        value = 'rounded'
    else:
        lines += [indent + 'bias_values = lm.load(bias[column], id="load_bias")',
                  indent + 'shifted = acc + lm.broadcast(bias_values, axis=1)']
        value = 'shifted'
    lines.append(indent + f'lm.store({output}[row, column], {value}, id="store_output")')
    return header + '\n'.join(lines) + '\n'


def extension_cases():
    """Return a fixed successor set. Never include the eight sealed first-block cases."""
    rows = []
    def add(name, task, batch, columns, widths=WIDTHS, *, depth=None, transform=None):
        workload, source = create_task(task, backend='triton-metax', rows=batch, columns=columns, depth=depth)
        if transform is not None:
            source = transform(source)
        rows.append(dict(name=name, task=task, rows=batch, columns=columns,
                         widths=widths, workload=workload, source=source))
    for task, batch, columns in [('fib_rmsnorm_h4096',64,4096),
                                  ('fib_fused_add_rmsnorm_h4096',64,4096),
                                  ('fib_rmsnorm_h128',32,128)]:
        add(task,task,batch,columns,(8,16))
    add('silu_resident','silu',4,128)
    add('silu_fixed_loop','silu',4,128,transform=_fixed_loop)
    add('rms_carried_loop','rmsnorm',4,256,transform=_carried_reduction)
    for mode in ('resident','k_loop','output_loop'):
        add('ieee_mma_'+mode,'aka_gemm_nt_bias',24,32,depth=64,
            transform=lambda source, mode=mode:_mma(source,mode=mode))
    add('rounded_mma_k_loop','fib_gemm_n128_k2048',5,128,depth=2048,
        transform=lambda source:_mma(source,mode='k_loop',rounded=True))
    return rows


def output_loop_cases():
    """An existing FP16 GEMM oracle for the pass's pure rounded output-loop domain."""
    workload, source = create_task('fib_gemm_n128_k2048', backend='triton-metax',
                                  rows=5, columns=128, depth=2048)
    return [dict(name='rounded_mma_output_loop', task='fib_gemm_n128_k2048',
                 rows=5, columns=128, widths=WIDTHS, workload=workload,
                 source=_mma(source, mode='output_loop', rounded=True))]
