"""Prepare one-variable AOT alignment comparisons from retained sealed sources."""
import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--runtime', type=Path, required=True)
parser.add_argument('--alignment', type=int, default=16)
args = parser.parse_args()
source = args.source.resolve(strict=True)
if source != Path(__file__).resolve().parents[1]:
    raise ValueError('prepare with the tool from the selected frozen checkout')
if args.output.exists():
    raise FileExistsError(args.output)
sys.path[:0] = [str(source), str(source / 'src')]
from open_cake_ir.compiler.target import declared_target
from open_cake_ir.compiler.backends.triton import pointer_type, target_route_facts
from open_cake_ir.compiler.ir import DType
from open_cake_ir.evaluation.core import TensorLaunchManifest
from open_cake_ir.evaluation.kernel_bundle import alignment_component
from open_cake_ir.evaluation.paired import candidate_identity, validate_pair_candidates
from open_cake_ir.evaluation.workload import WorkloadContract
from open_cake_ir.lab.bindings import load_baseline_bundle
from open_cake_ir.lab.build import BuildRequest, TritonToolchainBuilder
from open_cake_ir.lab.executor import ExecutorRevision
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
from open_cake_ir.source_identity import checkout_commit

commit = checkout_commit(source)
workload = WorkloadContract(json.loads((args.input / 'workload.json').read_text()))
old = load_baseline_bundle(source, args.input / 'optimized/candidate.json')
manifest = TensorLaunchManifest.from_dict(json.loads(old.artifact_payloads['launch_manifest']))
manifest.check_complete_domain()
manifest.check_workload(workload, 'primary')
original = old.artifact_payloads['lowered_source']
tree = ast.parse(original)
launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Subscript) and isinstance(n.func.value, ast.Name)
            and n.func.value.id == manifest.kernel_name]
if len(launches) != 1:
    raise ValueError('retained lowering must expose one unambiguous kernel launch')
constants = {kw.arg: ast.literal_eval(kw.value) for kw in launches[0].keywords}
options = {name: constants.pop(name) for name in ('num_warps', 'num_stages', 'maxnreg')
           if name in constants}
if 'num_warps' not in options:
    raise ValueError('retained lowering has no explicit CTA width')
requirements = {'compiler':'triton', 'source_language':'python', 'target':old.target,
    'kernel_entry_point':manifest.kernel_name,
    'signature':{n:pointer_type(DType(d)) for n, _, d, _ in manifest.tensor_abi},
    'compile_constants':constants, 'compile_options':options, 'grid':list(manifest.grid),
    **target_route_facts(declared_target(old.target))}
runtime = json.loads(args.runtime.read_text())['toolchain']
runtime['pointer_alignment'] = args.alignment
compiler = IsolatedTritonCompiler(**runtime)
compiler.check_executor(ExecutorRevision.for_target(source, old.target), author_workspace=args.output)
request = BuildRequest(old.candidate_sha256, original, 'lowered_source',
                       old.artifact_roles['lowered_source'], old.target, old.entry_point, requirements)
candidate = TritonToolchainBuilder(workload=workload, case_id='primary',
                                   isolated_compiler=compiler).build(request)
new_manifest = TensorLaunchManifest.from_dict(json.loads(candidate.artifact_payloads['launch_manifest']))
child, leaf = alignment_component(candidate,new_manifest)
validate_pair_candidates(candidate,old,workload,'primary')
args.output.mkdir(parents=True)
shutil.copyfile(args.input / 'workload.json', args.output / 'workload.json')
shutil.copytree(args.input / 'reference', args.output / 'reference')
# The fixed comparison baseline is the exact previously measured optimized binary.
# The original task starter stays in its original bundle and is referenced below.
shutil.copytree(args.input / 'optimized', args.output / 'starter')
out = args.output / 'optimized'
out.mkdir()
paths = {}
for role, payload in candidate.artifact_payloads.items():
    paths[role] = role + '.bin'
    (out / paths[role]).write_bytes(payload)
(out / 'candidate.json').write_text(json.dumps({'candidate':candidate_identity(candidate),
                                              'artifact_paths':paths})+'\n')
def instructions(item):
    return dict(Counter(re.findall(r'\b(?:ld|st)\.global[.a-zA-Z0-9_:]*',
                                  item.artifact_payloads['ptx'].decode())))
report = {'kind':'explicit_alignment_ablation', 'compiler_commit':commit,
          'source_input':str(args.input), 'original_task_starter':str(args.input / 'starter/candidate.json'),
          'comparison_baseline':'unchanged retained optimized binary, relabeled starter only for existing comparison input ABI',
          'treatment':'same source, constants and grid; runtime-guarded alignment variants',
          'pointer_alignment':args.alignment, 'compile_options':options,
          'ptx_global_instructions':{'old':instructions(old),'generic':instructions(candidate),'aligned':instructions(child)},
          'gpu_correctness':'not_run', 'performance':'not_measured'}
(args.output / 'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
