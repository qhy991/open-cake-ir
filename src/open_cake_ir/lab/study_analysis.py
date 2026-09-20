"""Task-weighted E/P analysis from independently audited, preassigned Runs."""
from collections import defaultdict
import json
import math
import random
from statistics import mean

from open_cake_ir.evidence import EvidenceStore
from .run_spec import RunRef
from .study_execution import validate_prepared_study

_COEFFICIENTS = {
    'experience':(-.5,.5,-.5,.5),
    'passes':(-.5,-.5,.5,.5),
    'interaction':(1.,-1.,-1.,1.),
}
_CELLS = ((False,False),(True,False),(False,True),(True,True))


def factorial_effects(rates):
    if len(rates) != 4 or any(type(value) not in {int,float} or not math.isfinite(value) or not 0 <= value <= 1 for value in rates):
        raise ValueError('factorial effects require four finite cell probabilities')
    return {name:sum(c*p for c,p in zip(coefficients,rates)) for name,coefficients in _COEFFICIENTS.items()}


def _quantile(values,p):
    values = sorted(values)
    index = (len(values)-1)*p
    lo,hi = math.floor(index),math.ceil(index)
    return values[lo]+(values[hi]-values[lo])*(index-lo)


def task_bootstrap(vectors, *, draws, seed):
    if len(vectors) < 2:
        raise ValueError('one-task bootstrap cannot estimate cross-task uncertainty')
    if type(draws) is not int or draws <= 0 or type(seed) is not int or seed < 0:
        raise ValueError('bootstrap draws and seed differ')
    for row in vectors: factorial_effects(row)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(draws):
        chosen = [vectors[rng.randrange(len(vectors))] for _ in vectors]
        effects = factorial_effects([mean(row[cell] for row in chosen) for cell in range(4)])
        for name,value in effects.items(): samples[name].append(value)
    return {name:[_quantile(values,.025),_quantile(values,.975)] for name,values in samples.items()}


def summarize_cells(plan,rows, *, intervals=False):
    """Keep all allocated Runs; average task rates, never pool selected survivors."""
    document = plan.document
    labels = [next(name for name,value in document['conditions'].items()
                   if (value['experience'],value['passes'])==cell) for cell in _CELLS]
    by_task = {}
    for task in document['tasks']:
        cells = {}
        for label in labels:
            members = [row for row in rows if row['task_id']==task['task_id'] and row['condition_id']==label]
            if not members or any(row['status'] not in {'success','no_success','missing'} for row in members):
                raise ValueError('Study analysis requires every preallocated cell')
            success = sum(row['status']=='success' for row in members)
            missing = sum(row['status']=='missing' for row in members)
            speedups = [row['confirmed_speedup'] for row in members if row.get('performance_eligible')]
            cells[label] = {'allocated':len(members),'success':success,'no_success':len(members)-success-missing,
                'missing':missing,'rate':success/len(members),
                'missingness_bounds':[success/len(members),(success+missing)/len(members)],
                'conditional_speedup':math.exp(mean(math.log(value) for value in speedups)) if speedups else None,
                'conditional_performance_runs':len(speedups)}
        by_task[task['task_id']] = {'family':task['family'],'generalization':task['generalization'],'cells':cells,
                                  'effects':factorial_effects([cells[label]['rate'] for label in labels])}
    strata = {}
    for name in sorted({task['generalization'] for task in document['tasks']}):
        tasks = {key:value for key,value in by_task.items() if value['generalization']==name}
        vectors = [[value['cells'][label]['rate'] for label in labels] for value in tasks.values()]
        rates = [mean(row[i] for row in vectors) for i in range(4)]
        bounds = [[mean(value['cells'][label]['missingness_bounds'][side] for value in tasks.values())
                   for side in (0,1)] for label in labels]
        effect_bounds = {effect:[sum(coef*bounds[i][0 if coef>=0 else 1] for i,coef in enumerate(coefficients)),
                                sum(coef*bounds[i][1 if coef>=0 else 0] for i,coef in enumerate(coefficients))]
                         for effect,coefficients in _COEFFICIENTS.items()}
        observed = [value for value in tasks.values()
                    if all(value['cells'][label]['allocated'] > value['cells'][label]['missing'] for label in labels)]
        observed_rates = ([mean(value['cells'][label]['success'] /
                          (value['cells'][label]['allocated']-value['cells'][label]['missing']) for value in observed)
                           for label in labels] if observed else None)
        uncertainty = document['analysis']['uncertainty']
        interval = (task_bootstrap(vectors,draws=uncertainty['draws'],seed=uncertainty['seed'])
                    if intervals and uncertainty is not None else None)
        strata[name] = {'tasks':list(tasks),'cell_rates':dict(zip(labels,rates)),
            'effects':factorial_effects(rates),'missingness_rate_bounds':dict(zip(labels,bounds)),
            'missingness_effect_bounds':effect_bounds,
            'observed_subset_sensitivity':{'tasks':len(observed),
                'cell_rates':dict(zip(labels,observed_rates)) if observed_rates else None,
                'effects':factorial_effects(observed_rates) if observed_rates else None},
            'interval_95':interval,
            'interval_scope':'paired_task_bootstrap_of_verified_success' if interval else None}
    return {'conditions':document['conditions'],'weighting':'equal_tasks','per_task':by_task,'strata':strata}


def _read_outcome(study,allocation, *, audit_run):
    spec = study.plan.run_specification(allocation)
    directory = study.root/'runs'/allocation.run_id
    row = {'run_id':allocation.run_id,'task_id':allocation.task_id,'condition_id':allocation.condition_id,
           'replicate':allocation.replicate,'status':'missing','reason':'not_started','terminal':False,
           'pipeline_verified':False,'confirmed_speedup':None,'performance_eligible':False,
           'first_correct':None,'first_correct_status':'unknown',
           'transform_requests':0,'transforms_applied':0,'transforms_refused':0}
    failure_path = directory/'failure.json'
    failure = None
    if failure_path.exists():
        failure = json.loads(failure_path.read_bytes())
        if (set(failure) != {'schema_version','run_id','run_specification_sha256','exception_type','message'}
            or failure['schema_version'] != 1 or failure['run_id'] != allocation.run_id
            or failure['run_specification_sha256'] != spec.canonical_sha256):
            raise ValueError('allocation failure belongs to a different frozen Run')
        row.update(terminal=True,reason='pre_execution_or_unsealed_failure',failure=failure)
    if not (directory/'evidence').exists():
        return row
    try:
        audit,replay = audit_run(RunRef(spec,directory/'evidence'))
    except (OSError,ValueError,KeyError) as error:
        row.update(reason='unverified_run',diagnostic=str(error))
        return row
    row['terminal'] = True
    if not replay or not audit.filesystem_custody_verified or not audit.archive_integrity:
        row.update(reason='unverified_run')
        return row
    if failure is not None:
        raise ValueError('a sealed audited Run cannot also be a failed allocation')
    row.update(pipeline_verified=True,run_endpoint=audit.endpoint_observation,
               protocol_adherence=audit.protocol_adherence)
    evidence = EvidenceStore.open(directory/'evidence')
    events = evidence.replay_events(allocation.run_id)
    state = next(event['payload']['ralph'] for event in events if event['kind']=='checkpoints_projected')
    fault = next((event['payload'] for event in events if event['kind']=='run_fault'),{})
    row.update(provider_tokens=state['cumulative_provider_tokens'],
               provider_tokens_scope=fault.get('terminal_provider_tokens_scope','observed_total'),
               elapsed_wall_seconds=state['elapsed_wall_seconds'],compilation_count=state['compilation_count'],
               evaluation_counts=state['evaluation_counts'],budget_exceeded=state['budget_exceeded'])
    tokens,compilations,evaluations = {},0,0
    for event in events:
        payload = event['payload']
        if event['kind']=='provider_turn_completed': tokens[payload['turn']] = payload['cumulative_provider_tokens']
        if event['kind']=='compilation_started': compilations += 1
        if event['kind']=='evaluation_attempt_started': evaluations += 1
        if event['kind']=='candidate_evaluated' and row['first_correct'] is None:
            reference = next(ref for ref in payload['objects'] if ref['role']=='evaluation_receipt')
            receipt = json.loads(evidence.read_object(reference))
            if receipt['correctness_passed'] and receipt['kernel_calls'] > 0 and receipt['fallback_calls']==0:
                row['first_correct'] = {'candidate_sha256':payload['candidate_sha256'],
                    'provider_tokens':tokens[payload.get('source_turn',payload.get('turn'))],
                    'compilations':compilations,'evaluations':evaluations,
                    'elapsed_wall_seconds':payload['elapsed_wall_seconds']}
                row['first_correct_status'] = 'observed'
        if event['kind']=='author_actions_resolved':
            for action in event['payload']['actions']:
                if action['kind']=='transform':
                    row['transform_requests'] += 1
                    row['transforms_applied' if action['reason']=='applied' else 'transforms_refused'] += 1
    if audit.protocol_adherence != 'adhered':
        row['reason'] = 'protocol_fault'
        return row
    if row['first_correct'] is None:
        row['first_correct_status'] = 'not_reached'
    confirmations = [event['payload'] for event in events if event['kind']=='candidate_evaluated'
                     and event['payload']['purpose']=='confirmatory']
    if not confirmations:
        row.update(status='no_success',reason='no_confirmed_candidate')
        return row
    payload = confirmations[0]
    reference = next(ref for ref in payload['objects'] if ref['role']=='evaluation_receipt')
    receipt = json.loads(evidence.read_object(reference))
    timing = receipt.get('timing')
    speedup = timing.get('speedup') if isinstance(timing,dict) else None
    confirmed = (receipt['correctness_passed'] is True and receipt['kernel_calls'] > 0
                 and receipt['fallback_calls']==0 and timing is not None
                 and timing.get('measurement_quality_passed') is True)
    if not confirmed:
        row.update(status='no_success',reason='confirmation_rejected')
        return row
    if type(speedup) not in {int,float} or not math.isfinite(speedup) or speedup <= 0:
        row['reason'] = 'paired_confirmation_unavailable'
        return row
    row.update(confirmed_speedup=speedup,performance_eligible=not state['budget_exceeded'])
    if audit.endpoint_observation != 'qualified':
        row.update(status='no_success',reason='budget_or_endpoint_not_qualified')
    elif timing.get('classification') == 'first_arm_faster':
        row.update(status='success',reason='confirmed_material_gain')
    else:
        row.update(status='no_success',reason='no_material_gain')
    return row


def audit_study(study, *, audit_run, validate_inputs):
    """Never adopt an unrelated or post-hoc-labeled Run into a Study analysis."""
    validate_prepared_study(study)
    validate_inputs(study.plan)
    rows = []
    for allocation in study.plan.allocations():
        try:
            rows.append(_read_outcome(study,allocation,audit_run=audit_run))
        except (OSError,ValueError,KeyError,TypeError,StopIteration) as error:
            rows.append({'run_id':allocation.run_id,'task_id':allocation.task_id,'condition_id':allocation.condition_id,
                'replicate':allocation.replicate,'status':'missing','reason':'unverified_allocation',
                'terminal':False,'pipeline_verified':False,'diagnostic':str(error),
                'confirmed_speedup':None,'performance_eligible':False})
    complete = all(row['terminal'] for row in rows)
    scientific = study.plan.document['claim_scope']=='scientific_matched_search'
    summary = summarize_cells(study.plan,rows,intervals=complete and scientific)
    costs = study.plan.document['upstream_costs']
    return {'study_id':study.plan.study_id,'claim_scope':study.plan.document['claim_scope'],
            'complete':complete,'allocated':len(rows),'pipeline_verified':sum(row['pipeline_verified'] for row in rows),
            'estimand_available':complete and scientific,
            'primary':summary if complete and scientific else None,
            'descriptive':summary,'runs':rows,
            'upstream_costs':{phase:{'evidence':refs,'coverage':'referenced' if refs else 'unreported_not_zero'}
                              for phase,refs in costs.items()},
            'evidence_scope':'scientific_preassigned_target_runs' if scientific else 'software_protocol_only'}
