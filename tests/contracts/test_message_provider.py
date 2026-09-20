"""Message confinement and native-usage replay; no real API calls or credentials."""
from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

from open_cake_ir.lab.message_provider import (
    CONTRACT, REVISION, MessageQualification, ResponsesHTTPTransport, ResponsesRunProvider,
    qualify, request_body, validate_exchange, response_submission,
)
from open_cake_ir.lab import RunSpecification
from open_cake_ir.lab.contracts import TurnRequest
from open_cake_ir.lab.faults import RunProtocolFault
from open_cake_ir.evidence import EvidenceStore
from open_cake_ir.serialization import canonical_json_bytes as encoded
from tests.contracts.test_lab import SemanticLabTestCase, FakeEnvironment, FakeEvaluator
from tests.contracts.test_run_specification import IndependentRunTests

ROOT = Path(__file__).resolve().parents[2]
CONFIG = {'harness':'responses','model':'cpu-message-model','reasoning_effort':'low',
          'max_output_tokens':128,'timeout_seconds':10,'sandbox':'messages_only','event_contract':CONTRACT}


class FakeTransport:
    def __init__(self, *, tool_turn=None, incomplete_turn=None):
        self.requests = []
        self.tool_turn = tool_turn
        self.incomplete_turn = incomplete_turn

    def __call__(self,payload):
        request = json.loads(payload)
        self.requests.append(request)
        message = json.loads(request['input'][-1]['content'])
        if 'expected' in message:
            envelope, turn = message['expected'], message['turn']
        else:
            turn = message['state_card']['iteration']
            envelope = {'schema_version':1,'arm':message['arm'],
                        'candidates':[{'run_id':message['run_id'],'turn':turn}]}
        output = [
            {'type':'reasoning','id':f'rs_{len(self.requests)}','summary':[],'encrypted_content':'opaque-CPU-fixture'},
            {'type':'message','id':f'msg_{len(self.requests)}','role':'assistant','phase':'final_answer',
             'content':[{'type':'output_text','text':json.dumps(envelope),'annotations':[]}]},
        ]
        if self.tool_turn == len(self.requests):
            output = [{'type':'function_call','name':'read_file','arguments':'{"path":"/withheld/pass.py"}'}]
        return encoded({'id':f'resp_{len(self.requests)}','model':CONFIG['model'],
            'status':'incomplete' if self.incomplete_turn == len(self.requests) else 'completed',
            'output':output,'usage':{'input_tokens':100,'output_tokens':20,'total_tokens':120,
                'input_tokens_details':{'cached_tokens':50},'output_tokens_details':{'reasoning_tokens':10}}})


class MessageProviderTests(SemanticLabTestCase):
    def fixture(self, transport=None):
        lab, original = IndependentRunTests.fixture(self)
        qualifier = FakeTransport()
        qualification = qualify(CONFIG,qualifier,fixture=True)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        receipt = Path(temporary.name).resolve()/'qualification.json'
        receipt.write_bytes(encoded(qualification.document))
        document = original.document
        scaffold = 'contracts/scaffolds/message-author/AGENTS.md'
        document['authoring']['scaffold'] = {'path':scaffold,'sha256':sha256((ROOT/scaffold).read_bytes()).hexdigest()}
        document['authoring']['provider'] = {**CONFIG,'revision':REVISION,
            'qualification':{'path':str(receipt),'canonical_sha256':qualification.canonical_sha256}}
        document['execution']['sandbox'] = 'messages_only'
        document['budget'].update(limit=240,checkpoints=[120,240],maximum_turns=2)
        specification = RunSpecification.from_dict(document)
        package = lab.task_package(specification,specification.run_id)
        provider = ResponsesRunProvider(qualification=qualification,task_packages={specification.run_id:package},
                                        transport=transport or FakeTransport())
        evaluator = FakeEvaluator(document['evaluation_protocol'],sha256(encoded(document['evaluation_protocol'])).hexdigest(),
                                  document['workload']['canonical_sha256'])
        return lab,specification,provider,evaluator,qualification

    def test_real_run_uses_no_tools_and_replays_exact_same_run_history(self):
        transport = FakeTransport()
        lab,spec,provider,evaluator,_ = self.fixture(transport)
        with tempfile.TemporaryDirectory() as directory:
            run = lab.execute_run(spec,Path(directory)/'evidence',provider=provider,
                environment=FakeEnvironment('open_cake',spec.document['authoring']),evaluator=evaluator)
            audit,replay = lab.audit_run(run)
            self.assertTrue(replay,replay.refusals)
            self.assertEqual(audit.endpoint_observation,'qualified')
            self.assertEqual(len(transport.requests),2)
            for request in transport.requests:
                self.assertEqual(request['tools'],[])
                self.assertEqual(request['tool_choice'],'none')
                self.assertIs(request['store'],False)
                self.assertNotIn('conversation',request)
                self.assertNotIn('previous_response_id',request)
            self.assertEqual(transport.requests[1]['input'][:len(transport.requests[0]['input'])],transport.requests[0]['input'])
            reasoning = [item for item in transport.requests[1]['input'] if item.get('type')=='reasoning']
            self.assertEqual(reasoning[0]['encrypted_content'],'opaque-CPU-fixture')
            events = EvidenceStore.open(run.evidence_root).replay_events(spec.run_id)
            turns = [event['payload'] for event in events if event['kind']=='provider_turn_completed']
            self.assertEqual([row['turn_provider_tokens'] for row in turns],[120,120])
            self.assertEqual([row['cumulative_provider_tokens'] for row in turns],[120,240])

    def test_returned_tool_call_is_never_executed_and_spent_tokens_remain_evidence(self):
        transport = FakeTransport(tool_turn=1)
        lab,spec,provider,evaluator,_ = self.fixture(transport)
        environment = FakeEnvironment('open_cake',spec.document['authoring'])
        with tempfile.TemporaryDirectory() as directory, patch.object(environment,'build',side_effect=AssertionError('tool output reached build')):
            run = lab.execute_run(spec,Path(directory)/'evidence',provider=provider,environment=environment,evaluator=evaluator)
            audit,replay = lab.audit_run(run)
            self.assertTrue(replay,replay.refusals)
            self.assertEqual(audit.protocol_adherence,'provider_fault')
            events = EvidenceStore.open(run.evidence_root).replay_events(spec.run_id)
            fault = next(event['payload'] for event in events if event['kind']=='run_fault')
            self.assertEqual(fault['provider_usage']['provider_tokens'],120)
            self.assertEqual(fault['terminal_provider_tokens'],120)
            self.assertEqual(evaluator.calls,0)

    def test_native_incomplete_response_does_not_lose_usage(self):
        transport = FakeTransport(incomplete_turn=1)
        lab,spec,provider,_,_ = self.fixture(transport)
        package = lab.task_package(spec,spec.run_id)
        request = TurnRequest(spec.run_id,spec.condition_id,1,0,None,{},1,
                              {'iteration':1,'previous_feedback':{}},'open_cake')
        with self.assertRaises(RunProtocolFault) as raised:
            provider.turn(request)
        self.assertEqual(raised.exception.reported_usage.provider_tokens,120)
        self.assertIn('provider_response',raised.exception.artifact_payloads)

    def test_two_runs_do_not_share_messages_or_response_items(self):
        _,spec,_,_,qualification = self.fixture()
        from open_cake_ir.lab.task_package import TaskPackage
        packages = {key:TaskPackage(key,key,f'SENTINEL_{key}','instructions') for key in ('first','second')}
        transport = FakeTransport()
        provider = ResponsesRunProvider(qualification=qualification,task_packages=packages,transport=transport)
        for key in packages:
            request = TurnRequest(key,key,1,0,None,{},1,{'iteration':1,'previous_feedback':{}},'open_cake')
            provider.turn(request)
        self.assertEqual(len(transport.requests[0]['input']),1)
        self.assertEqual(len(transport.requests[1]['input']),1)
        self.assertNotIn('SENTINEL_first',encoded(transport.requests[1]).decode())

    def test_replay_rejects_tools_and_foreign_conversation_state(self):
        fake = FakeTransport()
        qualification = qualify(CONFIG,fake,fixture=True)
        original = qualification.document['exchanges'][0]
        bundle = original['request']['input'][0]['content'].encode()
        for mutation in (
            lambda row:row['request'].update(tools=[{'type':'web_search'}]),
            lambda row:row['request'].update(conversation='foreign'),
            lambda row:row['request']['input'].insert(0,{'role':'user','content':'WITHHELD'}),
        ):
            row = deepcopy(original);mutation(row)
            with self.assertRaisesRegex(ValueError,'frozen context'):
                validate_exchange(encoded(row),config=CONFIG,history=[],bundle=bundle,
                                  run_id='provider-qualification',turn=1)

    def test_fixture_qualification_cannot_authorize_live_transport(self):
        qualification = qualify(CONFIG,FakeTransport(),fixture=True)
        with self.assertRaisesRegex(ValueError,'fixture qualification'):
            ResponsesRunProvider(qualification=qualification,task_packages={})
        with self.assertRaisesRegex(ValueError,'fixture qualification'):
            ResponsesRunProvider(qualification=qualification,task_packages={},transport=ResponsesHTTPTransport(10))

    def test_native_final_answer_phase_keeps_commentary_out_of_submission(self):
        class CommentaryTransport(FakeTransport):
            def __call__(self, payload):
                response = json.loads(super().__call__(payload))
                response['output'].insert(1, {'type':'message','id':'commentary-'+response['id'],
                    'role':'assistant','phase':'commentary','content':[{'type':'output_text',
                    'text':'Checking the supplied task.','annotations':[]}]})
                return encoded(response)
        transport = CommentaryTransport()
        qualification = qualify(CONFIG,transport,fixture=True)
        first = qualification.document['exchanges'][0]['response']
        self.assertEqual(json.loads(response_submission(first))['candidates'],[{'qualification_turn':1}])
        self.assertEqual([item['phase'] for item in transport.requests[1]['input']
            if item.get('role')=='assistant'],['commentary','final_answer'])
        first['output'] = [item for item in first['output'] if item.get('phase') != 'commentary']
        first['output'][-1].pop('phase')
        self.assertEqual(json.loads(response_submission(first))['candidates'],[{'qualification_turn':1}])
        first['output'][-1]['phase'] = 'final'
        with self.assertRaisesRegex(ValueError,'assistant phase'):
            response_submission(first)

    def test_live_qualification_refuses_injected_transport_and_provider_subclasses(self):
        # Synthetic live label is solely a constructor test: no live receipt is
        # persisted and no network call is made or qualified by this fixture.
        document = qualify(CONFIG,FakeTransport(),fixture=True).document
        document['scope'] = 'live_two_turn_message_provider'
        qualification = MessageQualification.from_dict(document)
        with self.assertRaisesRegex(ValueError,'owned HTTP transport'):
            ResponsesRunProvider(qualification=qualification,task_packages={},transport=FakeTransport())
        class ReplacementTransport(ResponsesHTTPTransport):
            def __call__(self, payload):
                raise AssertionError('must never run')
        with self.assertRaisesRegex(ValueError,'owned HTTP transport'):
            ResponsesRunProvider(qualification=qualification,task_packages={},transport=ReplacementTransport(10))
        class ReplacementProvider(ResponsesRunProvider):
            def turn(self, request):
                raise AssertionError('must never run')
        with self.assertRaisesRegex(ValueError,'owned Responses provider'):
            ReplacementProvider(qualification=qualification,task_packages={})
        provider = ResponsesRunProvider(qualification=qualification,task_packages={})
        provider._transport = FakeTransport()
        with self.assertRaisesRegex(ValueError,'owned HTTP transport'):
            provider.validate_transport(qualification)

    def test_admission_rechecks_transport_before_evidence_or_execution(self):
        lab,spec,provider,evaluator,_ = self.fixture()
        provider._transport = ResponsesHTTPTransport(10)
        environment = FakeEnvironment('open_cake',spec.document['authoring'])
        with tempfile.TemporaryDirectory() as directory, patch.object(ResponsesHTTPTransport,'__call__',
                side_effect=AssertionError('invalid transport reached network')):
            root = Path(directory)/'evidence'
            with self.assertRaisesRegex(ValueError,'fixture qualification'):
                lab.execute_run(spec,root,provider=provider,environment=environment,evaluator=evaluator)
            self.assertFalse(root.exists())

    def test_admission_refuses_withheld_material_before_transmission(self):
        transport = FakeTransport()
        lab,spec,provider,evaluator,_ = self.fixture(transport)
        provider._packages[spec.run_id] = replace(provider._packages[spec.run_id],
                                                 task_markdown='WITHHELD_E1_MATERIAL')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'evidence'
            with self.assertRaisesRegex(ValueError,'frozen Run materials'):
                lab.execute_run(spec,root,provider=provider,
                    environment=FakeEnvironment('open_cake',spec.document['authoring']),evaluator=evaluator)
            self.assertFalse(root.exists())
            self.assertEqual(transport.requests,[])

    def test_first_and_resumed_faults_replay_the_frozen_request_boundary(self):
        from open_cake_ir.lab.replay import replay_matched_run
        for fault_turn in (1,2):
            transport = FakeTransport(incomplete_turn=fault_turn)
            lab,spec,provider,evaluator,_ = self.fixture(transport)
            with self.subTest(fault_turn=fault_turn), tempfile.TemporaryDirectory() as directory:
                run = lab.execute_run(spec,Path(directory)/'evidence',provider=provider,
                    environment=FakeEnvironment('open_cake',spec.document['authoring']),evaluator=evaluator)
                audit,replay = lab.audit_run(run)
                self.assertTrue(replay,replay.refusals)
                evidence = EvidenceStore.open(run.evidence_root)
                fault = next(event['payload'] for event in evidence.replay_events(spec.run_id)
                             if event['kind']=='run_fault')
                reference = next(ref for ref in fault['objects'] if ref['role']=='provider_stdout')
                original = json.loads(evidence.read_object(reference))
                read_object = evidence.read_object
                for mutation in (
                    lambda row:row.update(run_id='foreign-run'),
                    lambda row:row.update(turn=99),
                    lambda row:row['request'].update(tools=[{'type':'web_search'}],tool_choice='auto'),
                    lambda row:row['request']['input'].insert(0,{'role':'user','content':'FOREIGN_HISTORY'}),
                    lambda row:row['request']['input'][-1].update(content=row['request']['input'][-1]['content'].replace(
                        '"task_markdown":','"unknown_material":"WITHHELD", "task_markdown":')),
                ):
                    changed = deepcopy(original); mutation(changed)
                    def read(ref):
                        return encoded(changed) if ref == reference else read_object(ref)
                    with patch.object(evidence,'read_object',side_effect=read):
                        replay = replay_matched_run(evidence,audit,spec,project_root=ROOT,
                            manifest_parser=lab._parse_manifest,task_package=lab.task_package)
                    self.assertFalse(replay)
                    self.assertIn('run_fault',str(replay.refusals[0]))

    def test_qualification_cli_retains_requests_without_real_network(self):
        from tools.qualify_message_provider import main
        transport = FakeTransport()
        with tempfile.TemporaryDirectory() as directory, patch.object(ResponsesHTTPTransport,'__call__',side_effect=transport):
            path = Path(directory)/'qualification'
            self.assertEqual(main(['--model',CONFIG['model'],'--reasoning-effort','low',
                '--evidence-root',str(path),'--fixture-only']),0)
            receipt = MessageQualification.load(path/'qualification.json')
            self.assertEqual(receipt.scope,'zero_gpu_contract_fixture_only')
            self.assertTrue((path/'turn-1.request.json').is_file())
            self.assertTrue((path/'turn-2.response.json').is_file())
            self.assertNotIn('Authorization', (path/'turn-1.request.json').read_text())


if __name__ == '__main__': unittest.main()
