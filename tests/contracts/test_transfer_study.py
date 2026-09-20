"""Four real Run paths with CPU-only providers/compilers; no transfer-gain claim."""
from copy import deepcopy
from hashlib import sha256
import json
import io
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from contextlib import redirect_stdout

from open_cake_ir.compiler import Compiler
from open_cake_ir.lab import StudyPlan, read_study
from open_cake_ir.lab.build import TritonToolchainBuilder
from open_cake_ir.lab.environments import OpenCakeEnvironment
from open_cake_ir.lab.endpoints import NORMAL_BUDGET_TERMINAL
from open_cake_ir.lab.message_provider import ResponsesRunProvider
from open_cake_ir.lab.study_analysis import factorial_effects, summarize_cells, task_bootstrap
from open_cake_ir.lab.study_execution import _case_key
from open_cake_ir.serialization import canonical_json_bytes as encoded
from open_cake_ir.tasks.workloads import create_task, load_workload
from tests.contracts import test_author_actions as action_fixture
from tests.contracts import test_message_provider as message_fixture
from tests.contracts.test_lab import SemanticLabTestCase, FakeEvaluator
from tests.contracts.test_native_triton_pairing import CompilationFixture

ROOT = Path(__file__).resolve().parents[2]
PASS = 'specialize_triton_warps'


class TransferStudyTests(SemanticLabTestCase):
    def fixture(self):
        lab,base,workload,program = action_fixture.AuthorActionTests.fixture(self,[])
        _,message,_,_,qualification = message_fixture.MessageProviderTests.fixture(self)
        document = base.document
        document['authoring']['provider'] = message.document['authoring']['provider']
        document['authoring']['scaffold'] = message.document['authoring']['scaffold']
        document['execution']['sandbox'] = 'messages_only'
        document['endpoint_policy'] = NORMAL_BUDGET_TERMINAL
        document['budget'].update(limit=240,checkpoints=[120,240])
        temporary = tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        environment = patch.dict(os.environ,{'OPEN_CAKE_CUSTODY_DIRECTORY':str(root/'registry')})
        environment.start();self.addCleanup(environment.stop)
        def upstream(backend,columns,name):
            value,_ = create_task('rmsnorm',backend=backend,rows=1,columns=columns)
            path = root/(name+'.json');path.write_bytes(encoded(value))
            reference = load_workload(path)
            return {'family':'normalization','case_id':'primary','evidence':['fixture://'+name],
                    'workload':{'workload_id':reference.workload_id,'path':str(path),
                                'canonical_sha256':reference.canonical_sha256}}
        plan = StudyPlan.from_dict({'schema_version':1,'study_id':'transfer-fixture','kind':'matched_search',
            'claim_scope':'system_qualification_only','source_target':'sm_103a',
            'conditions':{'control':{'experience':False,'passes':False},
                          'explanation':{'experience':True,'passes':False},
                          'rewrite':{'experience':False,'passes':True},
                          'both':{'experience':True,'passes':True}},
            'knowledge':{'materials':[{'schema_version':1,'knowledge_id':'width-fixture','version':'1',
                'mechanism':'SOURCE_MECHANISM_SENTINEL: CPU protocol fixture, no measured hardware gain.',
                'references':[{'role':'source','locator':'fixture://width-software-contract'}],
                'transformations':[PASS]}],'transformations':[PASS]},
            'discovery':[upstream('triton-b300',32,'discovery')],
            'adaptation':[upstream('triton-b200',64,'adaptation')],
            'tasks':[{'task_id':'silu-test','family':'activation','generalization':'unseen_family',
                      'run_template':{key:value for key,value in document.items()
                                      if key not in {'run_id','sequence','assignment','knowledge'}}}],
            'replicates':1,'allocation_seed':19,
            'analysis':{'endpoint':'confirmed_material_gain','weighting':'equal_tasks','uncertainty':None},
            'upstream_costs':{'discovery':[],'adaptation':[],'maintenance':[]}})
        return lab,plan,workload,program,qualification,root

    def factory(self,lab,workload,program,qualification,seen):
        parent = sha256(encoded(program)).hexdigest()
        class Transport(message_fixture.FakeTransport):
            def __call__(self,payload):
                response = json.loads(super().__call__(payload))
                bundle = json.loads(self.requests[-1]['input'][-1]['content'])
                turn = bundle['state_card']['iteration']
                action = {'action':'submit','candidate':program} if turn==1 else action_fixture.rewrite(parent)
                response['output'][-1]['content'][0]['text'] = json.dumps(
                    {'schema_version':1,'arm':bundle['arm'],'candidates':[action]})
                return encoded(response)
        def build(spec,directory):
            transport = Transport();seen[spec.run_id] = transport
            provider = ResponsesRunProvider(qualification=qualification,
                task_packages={spec.run_id:lab.task_package(spec,spec.run_id)},transport=transport)
            compiler = CompilationFixture()
            environment = OpenCakeEnvironment(Compiler.load(ROOT,ROOT/'compiler/revision.json'),
                TritonToolchainBuilder(workload=workload,case_id='primary',isolated_compiler=compiler),
                workload=workload,case_id='primary',authority_document=spec.document['authoring'])
            evaluator = action_fixture.ProgramEvaluator(spec.document['evaluation_protocol'],
                sha256(encoded(spec.document['evaluation_protocol'])).hexdigest(),workload.canonical_sha256)
            return {'provider':provider,'environment':environment,'evaluator':evaluator}
        return build

    def test_four_cells_execute_one_engine_with_actual_material_and_pass_isolation(self):
        lab,plan,workload,program,qualification,root = self.fixture()
        study = lab.prepare_study(plan,root/'study')
        before = lab.audit_study(study)
        self.assertFalse(before['complete']);self.assertEqual(before['allocated'],4)
        self.assertTrue(all(row['reason']=='not_started' for row in before['runs']))
        seen = {}
        from tools.transfer_study import main
        runtime = root/'runtime.json'
        runtime.write_text('{}')  # The injected CPU runtime replaces production adapters.
        with patch('tools.transfer_study.run_runtime_factory',return_value=self.factory(lab,workload,program,qualification,seen)) as factory, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(main(['execute','--study',str(study.root),'--runtime-config',str(runtime)]),0)
        factory.assert_called_once_with(ROOT,runtime)
        report = lab.audit_study(read_study(study.root))
        self.assertTrue(report['complete'])
        self.assertEqual(report['pipeline_verified'],4)
        self.assertEqual(report['evidence_scope'],'software_protocol_only')
        self.assertFalse(report['estimand_available']);self.assertIsNone(report['primary'])
        for row in report['runs']:
            condition = plan.document['conditions'][row['condition_id']]
            first = json.loads(seen[row['run_id']].requests[0]['input'][0]['content'])
            text = first['task_markdown']
            self.assertEqual('SOURCE_MECHANISM_SENTINEL' in text,condition['experience'])
            self.assertEqual('## Frozen reference: `transformation-api.json`' in text,condition['passes'])
            self.assertEqual(row['transforms_applied'],int(condition['passes']))
            self.assertEqual(row['transforms_refused'],int(not condition['passes']))
            self.assertIsNotNone(row['first_correct'])
            self.assertEqual(row['first_correct']['compilations'],1)
            self.assertEqual(len(seen[row['run_id']].requests[0]['input']),1)
        with self.assertRaisesRegex(ValueError,'already attempted'):
            lab.execute_study(study,runtime_factory=self.factory(lab,workload,program,qualification,{}))

    def test_allocations_and_treatments_freeze_before_run_and_reject_posthoc_edits(self):
        lab,plan,_,_,_,root = self.fixture()
        self.assertEqual(plan.allocations(),StudyPlan.from_dict(plan.document).allocations())
        runs = [plan.run_specification(row).document for row in plan.allocations()]
        common = [{key:value for key,value in row.items() if key not in {'run_id','sequence','assignment','knowledge'}} for row in runs]
        self.assertTrue(all(row==common[0] for row in common))
        study = lab.prepare_study(plan,root/'study')
        allocation = plan.allocations()[0]
        path = study.root/'runs'/allocation.run_id/'run.json'
        document = json.loads(path.read_bytes());document['assignment']['condition_id'] = 'posthoc'
        path.write_bytes(encoded(document))
        with self.assertRaisesRegex(ValueError,'predeclared'):
            lab.audit_study(study)

    def test_renamed_test_workload_copies_and_split_overlap_are_refused_before_output(self):
        lab,plan,_,_,_,root = self.fixture()
        document = plan.document
        duplicate = deepcopy(document['tasks'][0]);duplicate['task_id']='renamed-copy'
        document['tasks'].append(duplicate)
        with self.assertRaisesRegex(ValueError,'renamed duplicate'):
            lab.prepare_study(StudyPlan.from_dict(document),root/'duplicate')
        self.assertFalse((root/'duplicate').exists())
        document = plan.document
        document['tasks'][0]['family'] = 'normalization'
        with self.assertRaisesRegex(ValueError,'family membership'):
            StudyPlan.from_dict(document)
        changed = plan.document
        changed['conditions']['control']['passes'] = True
        with self.assertRaisesRegex(ValueError,'exactly once'):StudyPlan.from_dict(changed)
        changed = plan.document
        source,_ = create_task('silu',backend='triton-b300',rows=1,columns=32)
        path = root/'source-silu.json';path.write_bytes(encoded(source))
        source = load_workload(path)
        changed['discovery'][0].update(family='activation',workload={
            'workload_id':source.workload_id,'path':str(path),'canonical_sha256':source.canonical_sha256})
        changed['tasks'][0]['family'] = 'renamed-activation'
        with self.assertRaisesRegex(ValueError,'change family labels'):
            lab.prepare_study(StudyPlan.from_dict(changed),root/'renamed-family')

    def test_execution_failure_remains_allocated_and_does_not_replace_the_run(self):
        lab,plan,workload,program,qualification,root = self.fixture()
        study = lab.prepare_study(plan,root/'study')
        factory = self.factory(lab,workload,program,qualification,{})
        first = plan.allocations()[0].run_id
        def failed(spec,directory):
            if spec.run_id==first: raise OSError('CPU fixture unavailable runtime')
            return factory(spec,directory)
        lab.execute_study(study,runtime_factory=failed)
        report = lab.audit_study(study)
        self.assertTrue(report['complete']);self.assertEqual(report['allocated'],4)
        row = next(row for row in report['runs'] if row['run_id']==first)
        self.assertEqual(row['status'],'missing')
        self.assertEqual(row['reason'],'pre_execution_or_unsealed_failure')

    def test_unseen_shape_requires_seen_semantics_and_an_actual_new_shape(self):
        lab,plan,_,_,_,root = self.fixture()
        document = plan.document
        task = document['tasks'][0]
        task.update(family='normalization',generalization='unseen_shape')
        # New operator in a seen family is not evidence of shape generalization.
        with self.assertRaisesRegex(ValueError,'same non-shape contract'):
            lab.prepare_study(StudyPlan.from_dict(document),root/'new-operator')
        self.assertFalse((root/'new-operator').exists())
        task['family'] = 'activation'
        for split,backend,columns in [('discovery','triton-b300',32),('adaptation','triton-b200',64)]:
            value,_ = create_task('silu',backend=backend,rows=1,columns=columns)
            path = root/(split+'-silu.json');path.write_bytes(encoded(value))
            source = load_workload(path)
            document[split][0].update(family='activation',workload={
                'workload_id':source.workload_id,'path':str(path),'canonical_sha256':source.canonical_sha256})
        # Same SiLU semantics, source shapes 1x32/1x64, new test shape 2x8.
        self.assertEqual(len(lab.prepare_study(StudyPlan.from_dict(document),root/'new-shape').plan.allocations()),4)
        value,_ = create_task('silu',backend='triton-b300',rows=2,columns=8)
        path = root/'same-shape.json';path.write_bytes(encoded(value))
        source = load_workload(path)
        document['discovery'][0]['workload'] = {
            'workload_id':source.workload_id,'path':str(path),'canonical_sha256':source.canonical_sha256}
        with self.assertRaisesRegex(ValueError,'overlaps discovery/adaptation'):
            lab.prepare_study(StudyPlan.from_dict(document),root/'same-shape')

    def test_fixture_study_does_not_issue_task_population_intervals(self):
        _,plan,_,_,_,_ = self.fixture()
        document = plan.document
        document['analysis']['uncertainty'] = {'kind':'paired_task_percentile_bootstrap','draws':1000,
            'seed':1,'minimum_tasks':2,'pilot_evidence':'fixture://pilot'}
        with self.assertRaisesRegex(ValueError,'scientific'):StudyPlan.from_dict(document)
        with self.assertRaisesRegex(ValueError,'one-task'):task_bootstrap([[0.,1.,0.,1.]],draws=1000,seed=1)
        self.assertEqual(task_bootstrap([[0.,0.,0.,0.],[1.,1.,1.,1.]],draws=1000,seed=7),
                         {name:[0.,0.] for name in ('experience','passes','interaction')})

    def test_prepare_and_audit_cli_create_reviewable_inputs_without_running_providers(self):
        from tools.transfer_study import main
        _,plan,_,_,_,root = self.fixture()
        path = root/'plan.json';path.write_bytes(encoded(plan.document))
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(main(['prepare','--plan',str(path),'--output',str(root/'prepared')]),0)
        prepared = json.loads(output.getvalue())
        self.assertEqual(prepared['allocated_runs'],4)
        self.assertEqual(prepared['executed_runs'],0)
        output = io.StringIO()
        with redirect_stdout(output): self.assertEqual(main(['audit','--study',str(root/'prepared')]),0)
        report = json.loads(output.getvalue())
        self.assertFalse(report['complete'])
        self.assertTrue(all(row['reason']=='not_started' for row in report['runs']))

    def test_effects_use_task_weights_and_keep_missing_runs_in_denominators(self):
        _,plan,_,_,_,_ = self.fixture()
        document = plan.document
        second = deepcopy(document['tasks'][0]);second['task_id']='second-task'
        document['tasks'].append(second)
        # Arithmetic fixture only; preflight above independently rejects true
        # renamed workload copies. These rows are not hardware evidence.
        model = StudyPlan.from_dict(document)
        labels = ['control','explanation','rewrite','both']
        rows = []
        for task,rates,count in [('silu-test',[0,1,0,1],1),('second-task',[1,0,1,0],3)]:
            for label,success in zip(labels,rates):
                rows.extend({'task_id':task,'condition_id':label,'status':'success' if success else 'no_success',
                             'performance_eligible':False} for _ in range(count))
        summary = summarize_cells(model,rows)
        values = summary['strata']['unseen_family']
        self.assertEqual(set(values['cell_rates'].values()),{.5})
        self.assertEqual(values['effects'],{'experience':0.,'passes':0.,'interaction':0.})
        rows[0]['status']='missing'
        changed = summarize_cells(model,rows)['strata']['unseen_family']
        self.assertEqual(changed['missingness_rate_bounds']['control'],[.5,1.])
        self.assertEqual(changed['observed_subset_sensitivity']['tasks'],1)
        self.assertEqual(factorial_effects([0.,.2,.3,.9]),{'experience':.4,'passes':.5,'interaction':.4})
