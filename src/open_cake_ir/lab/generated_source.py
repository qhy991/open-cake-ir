"""Bounded views of a Run's sealed Compiler output, never a second artifact store."""
from __future__ import annotations

import json
from collections.abc import Mapping

from open_cake_ir.compiler.backends import BACKENDS
from open_cake_ir.compiler.ir import LoweringBackend, Program

GENERATED_SOURCE_V1 = 'generated_source_v1'
MAX_CANDIDATE_SOURCE_BYTES = 32768
MAX_TURN_SOURCE_BYTES = 65536
MAX_SOURCE_STAGES = 32
MAX_CANDIDATE_VIEW_BYTES = 65536


def generated_source_permission(authoring):
    """The authoring treatment grants inspection, not additional author tools."""
    feedback = authoring.get('feedback')
    if feedback is None:
        return False
    if not isinstance(feedback, list):
        raise ValueError('authoring feedback must be a list')
    if GENERATED_SOURCE_V1 not in feedback:
        return False
    if (feedback.count(GENERATED_SOURCE_V1) != 1
        or authoring.get('environment_kind') != 'open_cake'
        or authoring.get('reference_access') != 'known_kernel_reproduction'):
        raise ValueError('generated source feedback requires explicit Cake known-kernel authoring')
    return True


def authored_document(payload):
    """Use the existing non-executing frontend for the retained candidate envelope."""
    document = json.loads(payload)
    if isinstance(document, Mapping) and set(document) == {'python_source'}:
        from open_cake_ir.compiler.frontend import parse
        return parse(document['python_source'], filename='candidate.ir.py').document
    if isinstance(document, Mapping) and set(document) == {'python_program_source', 'program_id'}:
        from open_cake_ir.compiler.program_frontend import parse_program
        return parse_program(document['python_program_source'], filename='projected-program.ir.py',
                             program_id=document['program_id']).document
    if not isinstance(document, Mapping):
        raise ValueError('generated source candidate must be a Cake document')
    return document


def sealed_sources(candidate, authored_bytes):
    """Enumerate existing stage identities through their artifact owner.

    The owner still refuses unsupported Program execution, including multi-stage
    Metal. A single-kernel Program retains its authored stage name without inventing
    a wrapper stage or treating the Program id as a device entry point.
    """
    document = authored_document(authored_bytes)
    if candidate.is_program:
        from open_cake_ir.evaluation.program import program_components
        manifest, children, _ = program_components(candidate)
        stages = [(stage['name'], stage['schedule'], children[stage['name']])
                  for stage in manifest.program.document['stages']]
    elif 'program_id' in document:
        program = Program.from_dict(document)
        if len(program.stages) != 1:
            raise ValueError('single-kernel source view requires one authored Program stage')
        stage = program.document['stages'][0]
        stages = [(stage['name'], stage['schedule'], candidate)]
    else:
        stages = [(None, document, candidate)]
    result = []
    for stage_id, schedule, child in stages:
        route = schedule['lowering']
        backend = BACKENDS[LoweringBackend(route['backend'])]
        if child.target != schedule['target']:
            raise ValueError('generated source stage target differs from its candidate')
        result.append(({'stage_id': stage_id, 'target': child.target,
                        'route': dict(route),
                        'language': backend.source_language,
                        'artifact_role': 'lowered_source'},
                       child.artifact_payloads.get('lowered_source')))
    return result


def source_views(sources, remaining):
    """Keep complete stage text in deterministic stage order; disclose omissions."""
    used = 0
    views = []
    omission_reason = 'stage_count_limit'
    for identity, payload in sources[:MAX_SOURCE_STAGES]:
        row = dict(identity)
        added_bytes = 0
        if payload is None:
            row.update(status='unavailable', reason='sealed_source_missing')
        else:
            text = payload.decode('utf-8')
            size = len(payload)
            row['byte_count'] = size
            if size > MAX_CANDIDATE_SOURCE_BYTES - used:
                row.update(status='omitted', reason='candidate_source_byte_limit')
            elif size > remaining - used:
                row.update(status='omitted', reason='turn_source_byte_limit')
            else:
                # Line numbers are intrinsic to this exact text. CAKE_OP markers
                # remain intact; no independently mutable source map is introduced.
                row.update(status='available', first_line=1, line_count=len(text.splitlines()),
                           source=text)
                added_bytes = size
        projected = {'stages': views + [row],
                     'omitted_stage_count': len(sources) - len(views) - 1,
                     'stage_omission_reason': 'view_byte_limit'}
        # Names and route metadata are author-controlled too. Preserve exact
        # identities or omit the remaining stages; never truncate an identifier.
        if len(json.dumps(projected, ensure_ascii=False).encode('utf-8')) > MAX_CANDIDATE_VIEW_BYTES:
            omission_reason = 'view_byte_limit'
            break
        views.append(row)
        used += added_bytes
    omitted = len(sources) - len(views)
    result = {'stages': views, 'omitted_stage_count': omitted}
    if omitted:
        result['stage_omission_reason'] = omission_reason
    return result, used


def candidate_source_feedback(*, candidates, launchables, authored, specification):
    """Project only this completed Turn's sealed search candidates in author order.

    Source text is capped per Turn and per candidate. Each candidate's serialized
    view also has a bound, including metadata; the total view is bounded by that
    limit times the Run's frozen maximum_candidates_per_turn.
    """
    if not generated_source_permission(specification.document.get('authoring', {})):
        return {}
    remaining = MAX_TURN_SOURCE_BYTES
    result = {}
    for identity in candidates:
        candidate = launchables.get(identity)
        if candidate is None:
            result[identity] = {'stages': [], 'omitted_stage_count': 0, 'status': 'unavailable',
                                'reason': 'no_sealed_search_candidate'}
            continue
        if candidate.candidate_sha256 != identity or identity not in authored:
            raise ValueError('generated source differs from the resolved candidate')
        sources = sealed_sources(candidate, authored[identity])
        authority = specification.document['authoring']
        target = specification.document['execution']['target']
        if (candidate.target != target or any(stage['target'] != target
            or stage['route']['backend'] != authority['lowering_route']['backend']
            for stage, _ in sources)):
            raise ValueError('generated source target or backend differs from the Run')
        view, used = source_views(sources, remaining)
        result[identity] = view
        remaining -= used
    return result
