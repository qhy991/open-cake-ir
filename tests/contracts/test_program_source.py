"""Static Python Program composition reuses complete Cake Schedule stages."""
from pathlib import Path
import unittest

from open_cake_ir.compiler import Compiler
from open_cake_ir.compiler.program_frontend import parse_program

ROOT = Path(__file__).resolve().parents[2]


def source():
    stages = []
    for filename, name in (('epilogue_producer.py', 'producer'),
                           ('epilogue_consumer.py', 'consumer')):
        text = (ROOT/'examples/python'/filename).read_text()
        stages.append('@cake.schedule' + text.split('@cake.schedule', 1)[1].replace(
            'def candidate(', f'def {name}(', 1))
    return ('from open_cake_ir.compiler import frontend as cake\n\n'
            + '\n\n'.join(stages)
            + '\n\ncake.program(program_id="rounded-epilogue-python",\n'
              '    inputs=("a", "b", "bias"), outputs=("out",),\n'
              '    stages=(\n'
              '        cake.stage(name="producer", schedule=producer,\n'
              '                   bindings={"a": "a", "b": "b", "bias": "bias", "mid": "middle"}),\n'
              '        cake.stage(name="epilogue", schedule=consumer,\n'
              '                   bindings={"mid": "middle", "out": "out"}),\n'
              '    ))\n')


class PythonProgramSourceTests(unittest.TestCase):
    def test_program_declaration_is_one_bundle_candidate_not_two_stage_candidates(self):
        from open_cake_ir.lab.provider_documents import (
            PYTHON_CANDIDATE_BUNDLE_V1, _project_candidate_submission,
        )
        import json
        projected = _project_candidate_submission(source().encode(),
            submission_contract=PYTHON_CANDIDATE_BUNDLE_V1, arm='open_cake',
            environment_kind='open_cake', maximum_candidates_per_turn=2)
        self.assertEqual(len(projected), 1)
        member = json.loads(projected[0])
        self.assertEqual(set(member), {'python_program_source', 'program_id'})
        self.assertEqual(member['program_id'], 'rounded-epilogue-python')

    def test_two_python_stages_form_one_valid_program_and_lower(self):
        authored = parse_program(source(), filename='candidate-set.py',
                                 program_id='rounded-epilogue-python')
        program = authored.program
        self.assertEqual([stage.name for stage in program.stages], ['producer', 'epilogue'])
        self.assertEqual(program.inputs, ('a', 'b', 'bias'))
        self.assertEqual(program.outputs, ('out',))
        self.assertEqual(program.tensors['middle'].shape, (2, 8))
        lowered = Compiler.load(ROOT).lower_program(program)
        self.assertEqual(len(lowered.lowerings), 2)

    def test_lab_builds_and_replays_authored_python_program_bytes(self):
        from open_cake_ir.lab import CandidateSubmission, OpenCakeEnvironment, TritonToolchainBuilder
        from open_cake_ir.lab.provider_documents import PYTHON_CANDIDATE_BUNDLE_V1
        from open_cake_ir.serialization import canonical_json_bytes
        from tests.contracts.test_program_evaluation import workload_for, replay_program_candidate
        from tests.contracts.test_native_triton_pairing import CompilationFixture
        compiler = Compiler.load(ROOT)
        program = parse_program(source(), program_id='rounded-epilogue-python').program
        workload = workload_for(program)
        builder = TritonToolchainBuilder(workload=workload, case_id='primary',
                                        isolated_compiler=CompilationFixture())
        authority = {'input_format': 'python_source_v1',
                     'lowering_route': {'backend': 'triton', 'entry_point': 'epilogue_producer'},
                     'provider': {'submission_contract': PYTHON_CANDIDATE_BUNDLE_V1}}
        environment = OpenCakeEnvironment(compiler, builder, workload=workload, case_id='primary',
                                          authority_document=authority)
        payload = canonical_json_bytes({'python_program_source': source(),
                                        'program_id': 'rounded-epilogue-python'})
        submission = CandidateSubmission.seal(environment.media_type, payload)
        result = environment.build(submission)
        self.assertEqual(result.disposition, 'launchable', result.feedback)
        self.assertEqual(result.submission_sha256, submission.sha256)
        self.assertEqual(replay_program_candidate(compiler, program, result.launchable,
            result.launchable.artifact_payloads, authored_bytes=payload).canonical_sha256,
            result.launchable.canonical_sha256)

    def test_composition_refuses_read_before_producer_and_host_effects(self):
        for changed in (source().replace('"mid": "middle", "out": "out"',
                                         '"mid": "unproduced", "out": "out"'),
                        source() + '\nprint("host effect")\n'):
            with self.subTest(changed=changed[-50:]), self.assertRaises(ValueError):
                parse_program(changed, filename='candidate-set.py',
                              program_id='rounded-epilogue-python')
