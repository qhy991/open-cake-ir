"""QSA's legacy input/inspection adapter over Compiler Program and common Evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from collections.abc import Mapping

from open_cake_ir.compiler import Program
from open_cake_ir.evaluation.program import program_tensor_abi, program_components, ProgramLaunchManifest
from open_cake_ir.evaluation.workload import TensorABI
from open_cake_ir.evaluation.core import load_torch_program, _load_cubin
from open_cake_ir.lab.bindings import load_baseline_bundle


def admit_candidate_descriptor(document):
    if not isinstance(document,Mapping) or type(document.get('schema_version')) is not int:
        raise ValueError('QSA candidate fields differ')
    version,arm = document['schema_version'],document.get('arm')
    if arm=='open_cake' and version==2:
        if set(document) != {'schema_version','arm','program'}:
            raise ValueError('QSA complete Program descriptor differs')
        Program.from_dict(document['program'])
    elif arm=='open_cake' and version==1:
        if set(document) != {'schema_version','arm','nodes'} or not isinstance(document['nodes'],list):
            raise ValueError('QSA legacy candidate descriptor differs')
    elif arm=='direct_cuda' and version==1:
        if set(document) != {'schema_version','arm','source','launch_manifest'}:
            raise ValueError('QSA native candidate descriptor differs')
    else:
        raise ValueError('QSA candidate arm or version differs')
    return document


@dataclass(frozen=True)
class QsaWorkloadBinding:
    """Expected public ABI/target come from the frozen reference, never the candidate.

    The original hardware-independent Workload bytes and identity stay unchanged.
    The legacy Program reference declares the target and ordered public interface.
    """
    reference: object

    @property
    def target(self): return self.reference.implementation.target
    @property
    def canonical_sha256(self): return self.reference.workload.canonical_sha256
    @property
    def document(self): return self.reference.workload.document

    def tensor_abi(self,case_id):
        if case_id!='target_t32768':
            raise ValueError('QSA reference Program binds only target_t32768')
        return tuple(TensorABI(name,shape,dtype,mode)
                     for name,shape,dtype,mode in program_tensor_abi(self.reference.implementation))


def candidate_program(candidate,candidate_root,reference,owned_file):
    """Version 1 node files are an external adapter; version 2 carries a complete Program."""
    admit_candidate_descriptor(candidate)
    if candidate.get('arm')!='open_cake':
        raise ValueError('Cake Program input requires the Cake authoring environment')
    if candidate['schema_version']==2:
        program = Program.from_dict(candidate['program'])
    else:
        nodes = candidate.get('nodes')
        if not isinstance(nodes,list):
            raise ValueError('QSA legacy node input differs')
        expected = [stage.name for stage in reference.implementation.stages]
        if [row.get('id') for row in nodes if isinstance(row,Mapping)] != expected:
            raise ValueError('legacy QSA node input differs from its frozen reference')
        document = reference.implementation.document
        for stage,row in zip(document['stages'],nodes,strict=True):
            if set(row) != {'id','schedule'}:
                raise ValueError('QSA legacy node reference fields differ')
            path = owned_file(candidate_root,row['schedule'],'QSA candidate Schedule')
            stage['schedule'] = json.loads(path.read_bytes())
        program = Program.from_dict(document)
    if (program.target != reference.implementation.target
        or program_tensor_abi(program) != program_tensor_abi(reference.implementation)):
        raise ValueError('QSA Program public ABI or target differs from the frozen reference')
    if any(stage.schedule.lowering.backend.value!='triton' for stage in program.stages):
        raise ValueError('QSA Cake execution requires its declared Triton backend')
    return program


@dataclass(frozen=True)
class KernelInspection:
    kernel_id: str
    kernel_name: str


@dataclass(frozen=True)
class ProgramInspection:
    """Profiler labels only; this projection cannot allocate storage or launch code."""
    kernels: tuple[KernelInspection,...]


def read_cake_artifact(project_root,path):
    candidate = load_baseline_bundle(project_root,str(Path(path).resolve(strict=True)))
    manifest,children,_ = program_components(candidate)
    inspection = ProgramInspection(tuple(KernelInspection(stage.name,children[stage.name].entry_point)
                                        for stage in manifest.program.stages))
    return candidate,manifest,inspection


class LoadedCakeProgram:
    """Keep the old assay call shape while common Evaluation owns the actual Program."""
    def __init__(self,candidate,manifest: ProgramLaunchManifest,inspection,inputs,output,admission):
        public = {**inputs,'output':output}
        self.arguments = [public[name] for name,_,_,_ in manifest.tensor_abi]
        self.loaded,self.tensors = load_torch_program(candidate,manifest,self.arguments,admission,_load_cubin)
        self.manifest,self.artifact = manifest,inspection

    def launch(self,tensors,*,stream,boundary=None):
        if tensors is not self.tensors:
            raise ValueError('QSA Program argument mapping differs from its prepared storage')
        self.loaded.launch(self.arguments,tensor_contract=self.manifest,stream=stream,boundary=boundary)

    def close(self,*,synchronize):
        try:
            self.loaded.close(synchronize=synchronize)
        finally:
            self.arguments.clear()
            self.tensors = {}
