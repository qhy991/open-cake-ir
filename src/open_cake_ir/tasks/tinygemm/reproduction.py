"""NVIDIA TinyGEMM2 reproduction: independent math plus a mandatory bitwise peer.

The CPU oracle checks the peer. The frozen external peer supplies the expected BF16
bits to common Evaluation. No tolerance-only pass can qualify a Cake candidate.
"""
from __future__ import annotations

from array import array
from functools import cache
import json
import math
from pathlib import Path
import random

from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.tasks.tiles.workload import _checked_inputs, _round

TASK = 'cake_tinygemm2'
OPERATOR = 'cake_tinygemm2_bf16_bias'
TARGETS = {'triton-b200': 'sm_100a', 'triton-b300': 'sm_103a'}
CASES = ('primary', 'zeros', 'near_zero', 'alternating', 'mixed_magnitude')
PEER_COMMIT = '67f76379a145f19793896394974e29e610cda912'
PEER_DIRECTORY = 'experiments/flashinfer_rewrites/references/029_cake_tinygemm2/baseline/csrc'


def workload_document(*, backend='triton-b300', rows=1, columns=128, depth=720):
    if backend not in TARGETS:
        raise ValueError('TinyGEMM reproduction requires its exact B200 or B300 target')
    if (any(type(v) is not int or v <= 0 for v in (rows, columns, depth))
            or rows > 64 or columns % 16 or depth % 16
            or max(rows * depth, columns * depth, rows * columns) * 2 > 2**31 - 1):
        raise ValueError('TinyGEMM requires batch 1..64, N/K multiples of 16 and a bounded BF16 ABI')
    tensors = {'x': {'shape': ['B', 'K'], 'max_abs': 256.0},
               'weight': {'shape': ['N', 'K'], 'max_abs': 256.0},
               'bias': {'shape': ['N'], 'max_abs': 0.25}, 'out': {'shape': ['B', 'N']}}
    for spec in tensors.values():
        spec.update(dtype='bf16', layout='contiguous_row_major', finite_only=True)
    return {
        'schema_version': 1, 'workload_id': f'cake-tinygemm2-{backend}-b{rows}-n{columns}-k{depth}-v1',
        'revision': '1', 'state': 'frozen', 'operator': OPERATOR,
        'provenance': [{'kind': 'restricted_artifact', 'repository': 'https://github.com/flashinfer-ai/flashinfer',
                        'commit': PEER_COMMIT, 'path': PEER_DIRECTORY + '/tinygemm2.cu',
                        'scope': 'known_kernel_reproduction; fixed_external_peer_not_Cake_candidate'}],
        'cases': [{'case_id': case, 'shape': {'B': rows, 'N': columns, 'K': depth},
                   'seed': 427400 + i, 'mode': case} for i, case in enumerate(CASES)],
        'tensors': tensors,
        'semantics': {'definition': 'out = BF16(x @ weight.T + bias), with pinned TinyGEMM2 reduction order',
                      'target': TARGETS[backend], 'candidate_abi': {'inputs': ['x', 'weight', 'bias'], 'outputs': ['out']},
                      'input_effects': 'unchanged', 'output_storage': 'fresh_contiguous_nonaliasing',
                      'materialization': 'python Random(seed+10000*ABI_index); BF16 RNE; uniform[-.25,.25], zeros, uniform[-1e-4,1e-4], alternating +/-.5, signed powers2[-8,8]; bias uniform[-.25,.25]',
                      'exclusions': ['no_bias_free_path', 'no_PDL_claim', 'no_35_or_239_shape_coverage_claim',
                                     'no_generated_starter_mechanism_or_speed_claim']},
        'oracle': {'kind': 'fixed_tinygemm2_peer_checked_against_independent_cpu_math',
                   'callable': 'open_cake_ir.tasks.tinygemm.reproduction.reference_outputs',
                   'peer_commit': PEER_COMMIT, 'peer_entry': 'tinygemm2_op', 'peer_use_pdl': False,
                   'mathematical_atol': 1e-2, 'mathematical_rtol': 1e-2},
        'validation': {'primary_case': 'primary', 'all_cases_required': True, 'equal_nan': False,
                       'comparison': 'bitwise_bf16', 'atol': 0.0, 'rtol': 0.0,
                       'qualification': 'all_elements_all_cases_exact_peer_plus_independent_math; GPU qualification required'},
    }


def validate_contract(document):
    workload = WorkloadContract(document)
    backend = next((b for b, target in TARGETS.items() if target == workload.target), None)
    shape = workload.case('primary')['shape']
    if backend is None or set(shape) != {'B', 'N', 'K'}:
        raise ValueError('TinyGEMM reproduction target or dimensions differ')
    expected = workload_document(backend=backend, rows=shape['B'], columns=shape['N'], depth=shape['K'])
    if json.dumps(document, sort_keys=True, allow_nan=False) != json.dumps(expected, sort_keys=True):
        raise ValueError('TinyGEMM reproduction contract differs, including its mandatory bitwise gate')
    for case in CASES:
        workload.tensor_abi(case)


def materialize_case(workload, case_id):
    validate_contract(workload.document)
    case = workload.case(case_id)
    values = {}
    for index, arg in enumerate(a for a in workload.tensor_abi(case_id) if a.mode == 'input'):
        rng = random.Random(case['seed'] + index * 10000)
        data = array('f')
        for i in range(math.prod(arg.shape)):
            if case_id == 'zeros': value = 0.0
            elif arg.name == 'bias': value = rng.uniform(-0.25, 0.25)
            elif case_id == 'near_zero': value = rng.uniform(-1e-4, 1e-4)
            elif case_id == 'alternating': value = (-1.0 if (i + index) % 2 else 1.0) * 0.5
            elif case_id == 'mixed_magnitude': value = (-1.0 if i % 2 else 1.0) * 2.0**rng.randint(-8, 8)
            else: value = rng.uniform(-0.25, 0.25)
            data.append(_round(value, 'bf16'))
        values[arg.name] = data
    return values


def mathematical_reference(workload, case_id, inputs):
    """Independent FP64 CPU contraction then BF16 rounding, outside GPU timing."""
    import numpy as np
    validate_contract(workload.document)
    args = tuple(a for a in workload.tensor_abi(case_id) if a.mode == 'input')
    values = _checked_inputs(workload, args, inputs)
    x, weight, bias = [np.asarray(value, dtype=np.float64).reshape(arg.shape)
                       for value, arg in zip(values, args, strict=True)]
    return {'out': [_round(float(v), 'bf16') for v in (x @ weight.T + bias).reshape(-1)]}


@cache
def _peer_module(target):
    # Called only in the task evaluator's admitted GPU process; never by preparation.
    import torch
    from flashinfer.jit.core import gen_jit_spec, current_compilation_context
    from open_cake_ir.compiler.target import declared_target
    from open_cake_ir.source_identity import checkout_commit
    root = Path(__file__).resolve().parents[4]
    checkout_commit(root)
    declared = declared_target(target)
    if tuple(torch.cuda.get_device_capability()) != declared.compute_capability:
        raise ValueError('TinyGEMM external peer device does not match the exact target')
    directory = root / PEER_DIRECTORY
    return gen_jit_spec(
        f'open_cake_tinygemm2_peer_{PEER_COMMIT[:12]}_{target}', [directory / 'tinygemm2.cu'],
        extra_include_paths=[directory],
        extra_cuda_cflags=current_compilation_context.get_nvcc_flags_list(supported_major_versions=[10]),
    ).build_and_load()


def peer_reference(workload, case_id, inputs):
    import torch
    args = tuple(a for a in workload.tensor_abi(case_id) if a.mode == 'input')
    tensors = [torch.tensor(inputs[a.name], device='cuda', dtype=torch.bfloat16).reshape(a.shape) for a in args]
    before = [tensor.clone() for tensor in tensors]
    shape = workload.case(case_id)['shape']
    output = torch.full((shape['B'], shape['N']), float('nan'), device='cuda', dtype=torch.bfloat16)
    _peer_module(workload.target).tinygemm2_op(*tensors, output, False)
    torch.cuda.synchronize()
    if not all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(before, tensors, strict=True)):
        raise ValueError('external TinyGEMM peer mutated an input')
    return {'out': output.cpu().reshape(-1).tolist()}


def reference_outputs(workload, case_id, inputs):
    math_output = mathematical_reference(workload, case_id, inputs)['out']
    peer = peer_reference(workload, case_id, inputs)
    actual = peer.get('out')
    rule = workload.document['oracle']
    if (set(peer) != {'out'} or not isinstance(actual, list) or len(actual) != len(math_output)
            or any(type(x) not in (int, float) or not math.isfinite(x) or _round(x, 'bf16') != x
                   or abs(x-y) > rule['mathematical_atol'] + rule['mathematical_rtol'] * abs(y)
                   for x, y in zip(actual, math_output, strict=True))):
        raise ValueError('external TinyGEMM peer failed independent complete-output math check')
    return peer



def _source(workload, case_id, stages, partitioned):
    validate_contract(workload.document)
    if type(stages) is not int or stages not in (4, 8):
        raise ValueError('TinyGEMM starter stages must be 4 or 8')
    if partitioned and workload.target != 'sm_103a':
        raise ValueError('partitioned TinyGEMM is currently bounded to sm_103a')
    args = workload.tensor_abi(case_id)
    declarations = [f'{a.name}: cake.Tensor({a.shape!r}, "{a.dtype}"' +
                    (', mode="output")' if a.mode == 'output' else ')') for a in args]
    tile = 1024 if partitioned else 16
    iterative = workload.case(case_id)['shape']['K'] > tile
    lines = ['from open_cake_ir.compiler import frontend as cake', '',
             f'@cake.schedule(name="{workload.workload_id}-s{stages}", target="{workload.target}", backend="triton",',
             f'               entry_point="cake_tinygemm2", metadata={{"workload_contract_sha256": "{workload.canonical_sha256}"}})',
             f'def candidate(lm, {", ".join(declarations)}):',
             '    compute = lm.role(execution_groups=[0, 1, 2, 3])',
             '    row = lm.program(x, axis=0, dimension=0, tile=16)',
             '    column = lm.program(weight, axis=1, dimension=0, tile=16)']
    if iterative:
        lines.extend([f'    for k in lm.range(x, name="k_loop", dimension=1, tile={tile}, num_stages={stages}, disallow_acc_multi_buffer=True):',
                      '        with compute:'])
        pad = '            '
    else:
        # A unit K grid preserves masking without a fictitious single-trip loop.
        lines.extend([f'    k = lm.program(x, axis=2, dimension=1, tile={tile})', '    with compute:'])
        pad = '        '
    lines.extend([pad+'a = lm.load(x[row, k], id="load_x")',
                  pad+'b = lm.load(weight[column, k], id="load_weight")'])
    for i in range(4 if partitioned else 1):
        name = f'acc{i}' if partitioned else 'acc'
        ranges = f', k_ranges=[[{i*256}, {(i+1)*256}]]' if partitioned else ''
        lines.append(pad+f'{name} = lm.mma(a, b, instruction={{"contract": "triton.dot.bf16_fp32"}}, '
                     f'tile_shape=(16, 16, {tile}){ranges}, id="dot{i}")')
    lines.extend(['    with compute:', '        raw_bias = lm.load(bias[column], id="load_bias")',
                  '        bias32 = lm.cast(raw_bias, to="fp32", id="bias32")'])
    accumulator = 'acc'
    if partitioned:
        lines.append('        combined = ((acc0 + acc1) + acc2) + acc3')
        accumulator = 'combined'
    lines.extend([f'        result = {accumulator} + lm.broadcast(bias32, axis=1)',
                  '        rounded = lm.cast(result, to="bf16", id="round_out")',
                  '        lm.store(out[row, column], rounded, id="store_out")'])
    return '\n'.join(lines)+'\n'


def starter_source(workload, case_id='primary', *, stages=4):
    """Complete typed contraction; peer parity is a gate, not an assumption."""
    return _source(workload, case_id, stages, False)


def partitioned_source(workload, case_id='primary', *, stages=4):
    """Retain four K256 partials per K1024 cycle; no warp/TMA equivalence claim."""
    return _source(workload, case_id, stages, True)
