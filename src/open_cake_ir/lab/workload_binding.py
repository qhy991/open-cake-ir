"""Bind authored Cake schedules to the frozen Workload at the Lab boundary."""

from collections.abc import Mapping

from open_cake_ir.compiler.ir import Program


def bind_schedule_workload(document, workload_sha256):
    metadata = document.get('metadata')
    if (not isinstance(metadata, Mapping) or 'workload_contract_sha256' in metadata
        and metadata['workload_contract_sha256'] != workload_sha256):
        raise ValueError('Schedule Workload binding differs from the frozen Workload')
    return {**document, 'metadata': {**metadata, 'workload_contract_sha256': workload_sha256}}


def bind_program_workload(program: Program, workload_sha256: str) -> Program:
    bound = program.document
    for stage in bound['stages']:
        metadata = stage['schedule']['metadata']
        if ('workload_contract_sha256' in metadata
                and metadata['workload_contract_sha256'] != workload_sha256):
            raise ValueError(f"Program stage {stage['name']!r} Workload binding differs from the frozen Workload")
        metadata['workload_contract_sha256'] = workload_sha256
    return Program.from_dict(bound)
