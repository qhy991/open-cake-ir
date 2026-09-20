"""Frozen matched-search assignments; Run remains the sole execution authority.

The existing StudyContract/CampaignLock spelling is an external paired-study input.
This plan expresses same-Cake E/P assignments without teaching the Run engine a new mode.
"""
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import random
import re

from open_cake_ir.serialization import canonical_json_bytes
from .run_spec import RunSpecification
from .knowledge import validate_knowledge_access
from .endpoints import NORMAL_BUDGET_TERMINAL

_ASSIGNED = {'run_id','sequence','assignment','knowledge'}
_SCOPES = {'scientific_matched_search','system_qualification_only'}


def _closed(value, fields, name):
    if not isinstance(value,Mapping) or set(value) != set(fields):
        raise ValueError(f'{name} fields differ')
    return value


def _text(value, name):
    if not isinstance(value,str) or not value.strip() or value != value.strip():
        raise ValueError(f'{name} must be nonempty text')
    return value


def template_run(template):
    if not isinstance(template,Mapping) or _ASSIGNED & set(template):
        raise ValueError('Study Run template must leave identity, assignment and knowledge to the plan')
    return RunSpecification.from_dict({**template,'run_id':'study-template','sequence':1,
        'assignment':None,'knowledge':{'materials':[],'transformations':[]}})


@dataclass(frozen=True)
class StudyAllocation:
    run_id: str
    sequence: int
    task_id: str
    condition_id: str
    replicate: int


@dataclass(frozen=True)
class StudyPlan:
    _bytes: bytes

    @classmethod
    def from_dict(cls, document):
        fields = {'schema_version','study_id','kind','claim_scope','source_target','conditions',
                  'knowledge','discovery','adaptation','tasks','replicates','allocation_seed','analysis','upstream_costs'}
        _closed(document,fields,'Study plan')
        if type(document['schema_version']) is not int or document['schema_version'] != 1 or document['kind'] != 'matched_search':
            raise ValueError('Study plan version or kind differs')
        study_id = _text(document['study_id'],'Study id')
        if re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*',study_id) is None:
            raise ValueError('Study id must be a safe artifact name')
        if _text(document['claim_scope'],'Study claim scope') not in _SCOPES:
            raise ValueError('Study claim scope differs')
        _text(document['source_target'],'source target')
        conditions = document['conditions']
        if not isinstance(conditions,Mapping) or len(conditions) != 4:
            raise ValueError('E/P Study requires four named conditions')
        factors = set()
        for name,condition in conditions.items():
            _text(name,'condition id')
            _closed(condition,{'experience','passes'},'condition')
            if any(type(condition[field]) is not bool for field in condition):
                raise ValueError('E/P factors must be Boolean assignments')
            factors.add((condition['experience'],condition['passes']))
        if factors != {(False,False),(True,False),(False,True),(True,True)}:
            raise ValueError('each E/P combination must occur exactly once')
        validate_knowledge_access(document['knowledge'],environment_kind='open_cake')
        if not all(document['knowledge'][key] for key in ('materials','transformations')):
            raise ValueError('E/P Study requires nonempty material and transformation treatments')
        referenced = {name for unit in document['knowledge']['materials'] for name in unit['transformations']}
        if not set(document['knowledge']['transformations']) <= referenced:
            raise ValueError('granted transformations must belong to the frozen mechanism records')
        if any(ref['role']=='target_observation' for unit in document['knowledge']['materials'] for ref in unit['references']):
            raise ValueError('E treatment may not include target outcome observations')
        replicates = document['replicates']
        minimum = 5 if document['claim_scope']=='scientific_matched_search' else 1
        if type(replicates) is not int or replicates < minimum:
            raise ValueError(f'Study requires at least {minimum} predeclared replicates per cell')
        if type(document['allocation_seed']) is not int or document['allocation_seed'] < 0:
            raise ValueError('Study allocation seed must be a nonnegative integer')
        families = set()
        for name in ('discovery','adaptation'):
            entries = document[name]
            if not isinstance(entries,list) or not entries:
                raise ValueError(f'Study {name} split must be explicit and nonempty')
            for entry in entries:
                _closed(entry,{'family','workload','case_id','evidence'},name+' entry')
                families.add(_text(entry['family'],'operator family'))
                _text(entry['case_id'],'source case')
                _closed(entry['workload'],{'workload_id','path','canonical_sha256'},'source Workload reference')
                if not isinstance(entry['evidence'],list) or not entry['evidence']:
                    raise ValueError('discovery/adaptation entries must cite their evidence')
                for ref in entry['evidence']: _text(ref,'source evidence locator')
        tasks = document['tasks']
        if not isinstance(tasks,list) or not tasks:
            raise ValueError('Study requires target test tasks')
        ids,common,target = set(),None,None
        for task in tasks:
            _closed(task,{'task_id','family','generalization','run_template'},'test task')
            name = _text(task['task_id'],'task id')
            if name in ids: raise ValueError('Study task ids must be unique')
            ids.add(name)
            family = _text(task['family'],'test family')
            if _text(task['generalization'],'generalization') not in {'unseen_shape','unseen_family'}:
                raise ValueError('test generalization must distinguish unseen shapes and families')
            if (family in families) != (task['generalization']=='unseen_shape'):
                raise ValueError('test family membership differs from its declared generalization')
            run = template_run(task['run_template']).document
            author = run['authoring']
            if (author['environment_kind'] != 'open_cake' or author['provider'].get('harness') != 'responses'
                or run['execution']['sandbox'] != 'messages_only' or run['endpoint_policy'] != NORMAL_BUDGET_TERMINAL):
                raise ValueError('E/P Runs require confined message authoring and normal budget endpoints')
            shared = {key:run[key] for key in ('compiler_revision','budget','agent_interface','run_protocol','evidence_policy')}
            shared['authoring'] = {key:author.get(key) for key in
                ('provider','scaffold','reference_access','environment_kind','input_format','tool_surface','feedback','toolchain_sha256')}
            route = _closed(author.get('lowering_route'),{'backend','entry_point'},'Run lowering route')
            shared['backend'] = _text(route['backend'],'Run backend')
            shared['evaluation'] = {key:value for key,value in run['evaluation_protocol'].items()
                                    if key not in {'case_id','validation_case_ids'}}
            shared['execution'] = {key:value for key,value in run['execution'].items() if key != 'fixed_baseline'}
            if common is not None and canonical_json_bytes(shared) != common:
                raise ValueError('Study tasks differ in shared agent, Compiler, execution or budget controls')
            common = canonical_json_bytes(shared)
            target = run['execution']['target']
            if target == document['source_target']:
                raise ValueError('transfer Study source and target architectures must differ')
            if document['claim_scope']=='scientific_matched_search':
                from open_cake_ir.evaluation.paired import paired_protocol
                if paired_protocol(run['evaluation_protocol']) is None or 'fixed_baseline' not in run['execution']:
                    raise ValueError('scientific transfer needs a frozen local baseline and common paired confirmation')
        analysis = _closed(document['analysis'],{'endpoint','weighting','uncertainty'},'Study analysis')
        if analysis['endpoint'] != 'confirmed_material_gain' or analysis['weighting'] != 'equal_tasks':
            raise ValueError('Study endpoint or weighting differs')
        uncertainty = analysis['uncertainty']
        if uncertainty is not None:
            _closed(uncertainty,{'kind','draws','seed','minimum_tasks','pilot_evidence'},'Study uncertainty')
            if document['claim_scope'] != 'scientific_matched_search' or uncertainty['kind'] != 'paired_task_percentile_bootstrap':
                raise ValueError('task uncertainty is only available to scientific matched Studies')
            if (type(uncertainty['draws']) is not int or uncertainty['draws'] < 1000
                or type(uncertainty['seed']) is not int or uncertainty['seed'] < 0
                or type(uncertainty['minimum_tasks']) is not int or uncertainty['minimum_tasks'] < 2):
                raise ValueError('bootstrap requires explicit draws, seed and at least two independent tasks')
            _text(uncertainty['pilot_evidence'],'sample-size pilot evidence')
            for stratum in {task['generalization'] for task in tasks}:
                if sum(task['generalization']==stratum for task in tasks) < uncertainty['minimum_tasks']:
                    raise ValueError('a generalization stratum has fewer tasks than its preregistered minimum')
        costs = _closed(document['upstream_costs'],{'discovery','adaptation','maintenance'},'upstream costs')
        for refs in costs.values():
            if not isinstance(refs,list): raise ValueError('upstream costs are evidence locators, never assumed zero')
            for ref in refs: _text(ref,'upstream cost evidence')
        return cls(canonical_json_bytes(document))

    @classmethod
    def load(cls,path):
        return cls.from_dict(json.loads(Path(path).read_bytes()))

    @property
    def document(self): return json.loads(self._bytes)
    @property
    def study_id(self): return self.document['study_id']
    @property
    def canonical_sha256(self): return sha256(self._bytes).hexdigest()

    def allocations(self):
        document = self.document
        rng = random.Random(document['allocation_seed'])
        result = []
        for index,task in enumerate(document['tasks'],1):
            for replicate in range(1,document['replicates']+1):
                block = list(document['conditions'])
                rng.shuffle(block)
                for cell,condition in enumerate(block,1):
                    result.append(StudyAllocation(f'{document["study_id"]}-t{index}-r{replicate}-c{cell}',
                        len(result)+1,task['task_id'],condition,replicate))
        return tuple(result)

    def run_specification(self,allocation):
        if allocation not in self.allocations():
            raise ValueError('Run allocation is not part of this frozen Study')
        document = self.document
        task = next(task for task in document['tasks'] if task['task_id']==allocation.task_id)
        condition = document['conditions'][allocation.condition_id]
        return RunSpecification.from_dict({**task['run_template'],'run_id':allocation.run_id,
            'sequence':allocation.sequence,'assignment':{'study_id':document['study_id'],
            'study_sha256':self.canonical_sha256,'condition_id':allocation.condition_id},
            'knowledge':{'materials':document['knowledge']['materials'] if condition['experience'] else [],
                         'transformations':document['knowledge']['transformations'] if condition['passes'] else []}})


@dataclass(frozen=True)
class StudyRef:
    plan: StudyPlan
    root: Path
