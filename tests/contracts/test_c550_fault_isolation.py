"""Narrow candidate assertion classification and partial author-timeout evidence."""
import json,subprocess,unittest
from open_cake_ir.lab.triton_build import IsolatedTritonCompiler
from open_cake_ir.lab.claude import author_progress_before_timeout,CLAUDE_EXACT_FILE_EVENT_CONTRACT
from tools.launch_task_matrix import dispatch_must_stop

class C550FaultIsolation(unittest.TestCase):
    def test_known_layout_assertion_is_bounded_by_route_version_and_complete_signature(self):
        compiler=object.__new__(IsolatedTritonCompiler);compiler.triton_version='3.6.0'
        signature=['MACAMmaEncodingAttr::composeSharedLayoutForOperand',
                   'shape_judge && "tn and tk not meet conditon"',
                   'TritonGPUReduceDataDuplication','RuntimeError: PassManager::run failed']
        def outcome(text,code=1):return subprocess.CompletedProcess([],code,b'',text.encode())
        self.assertTrue(compiler._candidate_failure(outcome('\n'.join(signature)),{'code_object':'mcfatbin'}))
        for marker in signature:
            self.assertFalse(compiler._candidate_failure(outcome('\n'.join(x for x in signature if x!=marker)),{'code_object':'mcfatbin'}))
        for obj in ['cubin','hsaco',None]:
            self.assertFalse(compiler._candidate_failure(outcome('\n'.join(signature)),{'code_object':obj}))
        for extra in ['PermissionError: denied','ImportError: unavailable','bwrap: mount failed']:
            self.assertFalse(compiler._candidate_failure(outcome('\n'.join(signature)+'\n'+extra),{'code_object':'mcfatbin'}))
        compiler.triton_version='3.7.0'
        self.assertFalse(compiler._candidate_failure(outcome('\n'.join(signature)),{'code_object':'mcfatbin'}))
        self.assertTrue(compiler._candidate_failure(outcome('typed refusal',2),{}))

    def test_only_coherent_progress_timeout_is_local_and_never_success(self):
        session='01234567-89ab-cdef-0123-456789abcdef'
        events=[{'type':'system','subtype':'init','session_id':session,'model':'glm-5.3'},
                {'type':'system','subtype':'thinking_tokens','session_id':session,'estimated_tokens_delta':1}]
        def observed(xs):return author_progress_before_timeout(b'\n'.join(json.dumps(e).encode() for e in xs),model='glm-5.3',thread_id=session,event_contract=CLAUDE_EXACT_FILE_EVENT_CONTRACT)
        self.assertTrue(observed(events))
        self.assertFalse(observed(events[:1]))
        for extra in [{'type':'result','session_id':session},
                      {'type':'system','subtype':'api_retry','session_id':session},
                      {'type':'system','subtype':'thinking_tokens','session_id':'foreign','estimated_tokens_delta':1}]:
            self.assertFalse(observed(events+[extra]))
        report={'audit':{'protocol_adherence':'provider_fault','archive_integrity':True,'filesystem_custody_verified':True},'replay':{'refusals':[]}}
        fault={'exception_type':'ProviderDeliveryTimeout','stage':'provider','observed_quota':{'observed':'no_notice'}}
        self.assertFalse(dispatch_must_stop(report,0,fault))
        self.assertEqual(report['audit']['protocol_adherence'],'provider_fault')
        self.assertTrue(dispatch_must_stop(report,0,{**fault,'exception_type':'RunProtocolFault'}))
