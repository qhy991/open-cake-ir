"""Task-owned admission of the declared squared-difference arithmetic.

This checks the operand formation, not index coverage or floating-point accuracy;
those remain Compiler analysis and the external device oracle respectively.
"""

def validate_candidate_arithmetic(document, workload):
    if workload.document['operator'] != 'contraction_pairwise_sqdist_fp32':
        return
    if 'program_id' in document:
        stages = document['stages']
        if len(stages) != 1:
            raise ValueError('Workload arithmetic: multi-stage squared-distance arithmetic is outside this admitted domain')
        document = stages[0]['schedule']
    operations = document['operations']
    buffers = {b['name']:b for b in document['buffers']}
    writers = {}
    for op in operations:
        for name in op.get('writes',[]):
            writers.setdefault(name,[]).append(op)
    squares = [op for op in operations if op['kind']=='elementwise' and op['parameters'].get('op')=='square']
    if not squares:
        raise ValueError('Workload arithmetic: form the FP32 difference before its square; formula expansion is not admitted')
    allowed = {'load','elementwise','reduce','store'}
    for op in operations:
        if (op['kind'] not in allowed or op['kind']=='elementwise' and op['parameters'].get('op') not in {'sub','square','add'}
            or op['kind']=='reduce' and op['parameters'].get('op') != 'sum'):
            raise ValueError('Workload arithmetic: only FP32 differences, squares and sums participate in this bounded task domain')
    if any(b['dtype'] != 'fp32' for b in buffers.values()):
        raise ValueError('Workload arithmetic: squared-distance values and intermediates must remain FP32')
    for square in squares:
        source = writers.get(square['reads'][0],[]) if len(square['reads'])==1 else []
        if len(source)!=1 or source[0]['kind']!='elementwise' or source[0]['parameters'].get('op')!='sub':
            raise ValueError('Workload arithmetic: square must read a rounded FP32 subtraction')
        operands = source[0]['reads']
        loads = [writers.get(name,[]) for name in operands]
        if (len(loads)!=2 or any(len(row)!=1 or row[0]['kind']!='load' for row in loads)
            or {row[0]['reads'][0] for row in loads} != {'x','c'}):
            raise ValueError('Workload arithmetic: each difference must use loaded x and c operands')
