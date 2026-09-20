"""Task-side assembly of complete Python-authored Cake Schedules into a launch plan."""
from open_cake_ir.compiler.frontend import parse
from open_cake_ir.compiler import Program
from open_cake_ir.evaluation.launch_plan import prepare_program


def plan_workload_target(backend, *, top_k=False, fp8=False):
    """Bind a successor to the existing device registry and declared operations.

    The original B300 documents remain revision 1. Other Triton targets describe
    new contracts; this structural admission never imports another device's results.
    """
    from open_cake_ir.tasks.devices import BACKENDS, admit_dtype, admit_operations
    if backend not in BACKENDS or BACKENDS[backend]['route'] != 'triton':
        raise ValueError('FlashInfer composed tasks require a registered Triton backend')
    if backend != 'triton-b300':
        kinds = ('load', 'cast', 'coordinate', 'compare', 'select', 'elementwise', 'reduce', 'store')
        admit_operations(backend, kinds + (('top_k',) if top_k else ()))
        for dtype in ('fp32', 'bf16', 'int32') + (('fp8_e4m3',) if fp8 else ()):
            admit_dtype(backend, dtype)
    return BACKENDS[backend]['target']


class PlanAuthor:
    def __init__(self, workload, case_id):
        self.workload = workload
        abi = workload.tensor_abi(case_id)
        self.document = {
            'schema_version':1, 'program_id':workload.workload_id+'-'+case_id,
            'target':workload.target,
            'inputs':[a.name for a in abi if a.mode=='input'],
            'outputs':[a.name for a in abi if a.mode=='output'],
            'tensors':{a.name:{'shape':list(a.shape),'dtype':a.dtype} for a in abi},
            'stages':[],
        }
        self.sources = {}

    def tensor(self, name, shape, dtype):
        if name in self.document['tensors']:
            raise ValueError(f'plan tensor {name!r} already declared')
        self.document['tensors'][name] = {'shape':list(shape),'dtype':dtype}

    def stage(self, name, inputs, outputs, axes, body):
        declarations = []
        for mode,names in [('input',inputs),('output',outputs)]:
            for field in names:
                spec = self.document['tensors'][field]
                declarations.append(f'{field}: cake.Tensor({tuple(spec["shape"])!r}, "{spec["dtype"]}", mode="{mode}")')
        source = ('from open_cake_ir.compiler import frontend as cake\n\n'
                  f'@cake.schedule(name="{name}", target="{self.workload.target}", backend="triton", entry_point="cake_{name}",\n'
                  f'               metadata={{"workload_contract_sha256": "{self.workload.canonical_sha256}"}})\n'
                  f'def candidate(lm, {", ".join(declarations)}):\n'
                  '    compute = lm.role(execution_groups=[0, 1, 2, 3])\n')
        for axis,(coord,buffer,dimension,tile) in enumerate(axes):
            source += f'    {coord} = lm.program({buffer}, axis={axis}, dimension={dimension}, tile={tile})\n'
        source += '\n'.join('    '+line for line in body)+'\n'
        schedule = parse(source, filename=name+'.py').document
        self.sources[name] = source
        self.document['stages'].append({'name':name,'schedule':schedule,
                                      'bindings':{field:field for field in (*inputs,*outputs)}})

    def finish(self):
        return Program.from_dict(self.document)
