"""Prepare frozen allocations and dispatch the existing independent Run engine."""
from collections.abc import Mapping
import json
from pathlib import Path

from open_cake_ir.serialization import canonical_json_bytes
from .bindings import source_reference_path, qualification_path
from .custody import admit_new_campaign_path
from .reference_access import require_qualified_clean_start_execution
from .message_provider import MessageQualification
from .study_plan import StudyPlan, StudyRef
from .run_spec import RunSpecification


def _workload(reference, *, root, loader):
    _,path = source_reference_path(root,reference['path'],'Study Workload')
    workload = loader(path)
    if workload.workload_id != reference['workload_id'] or workload.canonical_sha256 != reference['canonical_sha256']:
        raise ValueError('Study Workload differs from its frozen reference')
    return workload


def _case_key(workload,case_id):
    # This detects canonical renamed copies, not arbitrary mathematical equivalence.
    semantics = workload.document['semantics']
    semantics.pop('target',None)
    abi = [(row.name,list(row.shape),row.dtype,row.mode) for row in workload.tensor_abi(case_id)]
    return canonical_json_bytes({'operator':workload.document['operator'],'semantics':semantics,
                                 'shape':workload.case(case_id)['shape'],'abi':abi})


def _non_shape_contract(workload,case_id):
    """Shape transfer keeps operator semantics, tensor roles and numeric acceptance.

    Family names alone do not establish that relation. Target identity, case labels
    and explanatory prose do not change the contract; tensor dimensions may change.
    This compares declared contracts, not arbitrary mathematical equivalence.
    """
    document = workload.document
    semantics = document['semantics']
    semantics.pop('target',None)
    validation = document['validation']
    for field in ('primary_case','tolerance_rationale','qualification'):
        validation.pop(field,None)
    abi = [(row.name,row.dtype,row.mode) for row in workload.tensor_abi(case_id)]
    return canonical_json_bytes({'operator':document['operator'],'semantics':semantics,
                                 'abi':abi,'validation':validation})


def validate_study_inputs(plan, *, project_root, workload_loader, preflight_run, task_package):
    document = plan.document
    target = document['tasks'][0]['run_template']['execution']['target']
    seen = {}
    upstream_contracts = set()
    family_by_operator = {}
    def check_family(workload,family):
        operator = workload.document['operator']
        if operator in family_by_operator and family_by_operator[operator] != family:
            raise ValueError('one operator cannot change family labels across Study splits')
        family_by_operator[operator] = family
    for split in ('discovery','adaptation'):
        phase_keys = set()
        for entry in document[split]:
            workload = _workload(entry['workload'],root=project_root,loader=workload_loader)
            check_family(workload,entry['family'])
            expected_target = document['source_target'] if split=='discovery' else target
            if workload.target != expected_target:
                raise ValueError(f'{split} Workload target differs from its declared phase')
            key = _case_key(workload,entry['case_id'])
            if key in phase_keys:
                raise ValueError(f'{split} repeats a canonical workload case')
            phase_keys.add(key)
            seen[key] = split
            upstream_contracts.add(_non_shape_contract(workload,entry['case_id']))
    # One representative per task suffices for shared runtime admission. All four
    # actual material/API projections are nevertheless constructed before dispatch.
    by_task = {}
    for allocation in plan.allocations():
        by_task.setdefault(allocation.task_id,{})[allocation.condition_id] = allocation
    for task in document['tasks']:
        template = task['run_template']
        workload = _workload(template['workload'],root=project_root,loader=workload_loader)
        check_family(workload,task['family'])
        case_id = template['evaluation_protocol']['case_id']
        if (task['generalization']=='unseen_shape'
            and _non_shape_contract(workload,case_id) not in upstream_contracts):
            raise ValueError('unseen_shape requires an upstream operator with the same non-shape contract')
        key = _case_key(workload,case_id)
        if key in seen:
            raise ValueError('test task is a renamed duplicate or overlaps discovery/adaptation cases')
        seen[key] = 'test'
        allocations = list(by_task[task['task_id']].values())
        specification = preflight_run(plan.run_specification(allocations[0]))
        provider = specification.document['authoring']['provider']
        _,path = qualification_path(project_root,provider['qualification']['path'],'Study provider qualification')
        qualification = MessageQualification.load(path)
        if (document['claim_scope']=='scientific_matched_search'
            and qualification.scope != 'live_two_turn_message_provider'):
            raise ValueError('scientific Study cannot use a CPU fixture provider qualification')
        for allocation in allocations:
            run = plan.run_specification(allocation)
            task_package(run,run.run_id)


def prepare_study(plan,root, *, project_root, workload_loader, preflight_run, task_package):
    plan = StudyPlan.from_dict(plan.document) if isinstance(plan,StudyPlan) else StudyPlan.load(plan)
    root = admit_new_campaign_path(project_root,root,role='prepared Study')
    validate_study_inputs(plan,project_root=project_root,workload_loader=workload_loader,
                          preflight_run=preflight_run,task_package=task_package)
    root.mkdir(parents=True,exist_ok=False)
    with (root/'study.json').open('xb') as stream: stream.write(canonical_json_bytes(plan.document))
    for allocation in plan.allocations():
        directory = root/'runs'/allocation.run_id
        directory.mkdir(parents=True,exist_ok=False)
        with (directory/'run.json').open('xb') as stream:
            stream.write(canonical_json_bytes(plan.run_specification(allocation).document))
    return StudyRef(plan,root)


def read_study(root):
    root = Path(root).resolve(strict=True)
    return StudyRef(StudyPlan.load(root/'study.json'),root)


def validate_prepared_study(study):
    retained = StudyPlan.load(study.root/'study.json')
    if canonical_json_bytes(retained.document) != canonical_json_bytes(study.plan.document):
        raise ValueError('prepared Study changed after freezing')
    for allocation in study.plan.allocations():
        spec = RunSpecification.load(study.root/'runs'/allocation.run_id/'run.json')
        if canonical_json_bytes(spec.document) != canonical_json_bytes(study.plan.run_specification(allocation).document):
            raise ValueError('prepared Run differs from its predeclared Study allocation')


def execute_study(study, *, execute_run, runtime_factory):
    """A sequential allocation driver; each Run owns all search/fault/terminal state.

    Failures before a Run seals are retained separately without fabricating a Run
    result. There are no replacement allocations or automatic experiment retries.
    """
    validate_prepared_study(study)
    allocations = study.plan.allocations()
    require_qualified_clean_start_execution(
        study.plan.run_specification(allocation).document['authoring']
        for allocation in allocations)
    for allocation in allocations:
        directory = study.root/'runs'/allocation.run_id
        if (directory/'evidence').exists() or (directory/'failure.json').exists():
            raise ValueError('Study allocation was already attempted; never replace its evidence')
    for allocation in allocations:
        spec = study.plan.run_specification(allocation)
        directory = study.root/'runs'/allocation.run_id
        try:
            components = runtime_factory(spec,directory)
            if not isinstance(components,Mapping) or set(components) != {'provider','environment','evaluator'}:
                raise ValueError('Run runtime factory must bind provider, environment and evaluator')
            execute_run(spec,directory/'evidence',**components)
        except Exception as error:
            failure = {'schema_version':1,'run_id':spec.run_id,'run_specification_sha256':spec.canonical_sha256,
                       'exception_type':type(error).__name__,'message':str(error)[:2048]}
            with (directory/'failure.json').open('xb') as stream:
                stream.write(canonical_json_bytes(failure))
    return study
