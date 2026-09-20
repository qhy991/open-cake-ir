"""Reconstruct native compilation permits separately from candidate counts."""
import math

from .refusals import refuse


def replay_compilations(events, *, candidates_by_turn, maximum, target):
    count, active, last_time = 0, None, 0.0
    filtered = set()
    closed = False
    identity_fields = {'turn','candidate_sha256','compiler','target','entry_point','variant'}
    for event in events:
        kind,payload = event['kind'],event['payload']
        if kind == 'candidate_set_filtered':
            filtered.add(payload['turn'])
        if kind in {'search_completed','checkpoints_projected'}:
            state = payload.get('state') if kind == 'search_completed' else payload.get('ralph')
            if not isinstance(state,dict) or (type(state.get('compilation_count')) is not int
                or state.get('compilation_count') != count
                or type(state.get('remaining',{}).get('compilations')) is not int
                or state.get('remaining',{}).get('compilations') != maximum-count):
                refuse(kind, 'compilation budget does not derive from native invocation starts')
            if type(state.get('elapsed_wall_seconds')) not in {int,float} or state['elapsed_wall_seconds'] < last_time:
                refuse(kind, 'Run clock predates its native compilation observations')
            closed = True
        if not kind.startswith('compilation_'):
            if active is not None:
                refuse('compilation_started', 'native compilation did not complete before the next Run event')
            continue
        if closed:
            refuse(kind,'native compilation continued after search closure')
        if kind in {'compilation_started','compilation_refused'}:
            expected = identity_fields | ({'number','elapsed_wall_seconds'} if kind=='compilation_started' else {'reason'})
            if (set(payload) != expected or type(payload.get('turn')) is not int
                or payload['candidate_sha256'] not in candidates_by_turn.get(payload['turn'],())
                or payload['turn'] in filtered or payload.get('target') != target
                or any(not isinstance(payload.get(name),str) or not payload[name] for name in ('compiler','entry_point','variant'))):
                refuse(kind,'native compilation identity differs from the admitted candidate build')
            if active is not None:
                refuse(kind,'native compiler invocations overlap')
            if kind == 'compilation_refused':
                if count != maximum or payload['reason'] != 'compilation_budget':
                    refuse(kind,'a native compilation permit was refused before exhaustion')
                continue
            if count >= maximum or type(payload['number']) is not int or payload['number'] != count+1:
                refuse(kind,'native compilation start exceeds its quota or ordinal')
            count += 1
            active = (payload['turn'],payload['candidate_sha256'],payload['number'])
        elif kind == 'compilation_completed':
            if (set(payload) != {'turn','candidate_sha256','number','outcome','elapsed_wall_seconds'}
                or type(payload.get('turn')) is not int or type(payload.get('number')) is not int
                or (payload['turn'],payload['candidate_sha256'],payload['number']) != active
                or payload.get('outcome') not in {'returned','raised'}):
                refuse(kind,'native compilation completion differs from its unique start')
            active = None
        elapsed = payload.get('elapsed_wall_seconds')
        if type(elapsed) not in {int,float} or not math.isfinite(elapsed) or elapsed < last_time:
            refuse(kind,'native compilation clock is nonfinite or runs backwards')
        last_time = elapsed
    if active is not None:
        refuse('compilation_started','native compiler invocation is unfinished')
    return count
