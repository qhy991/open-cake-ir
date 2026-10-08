"""A task's archived failure does not become success or stop healthy siblings."""
import copy
import unittest
from tools.launch_task_matrix import dispatch_must_stop

class DispatchFaultScope(unittest.TestCase):
    def test_only_archived_known_author_faults_continue(self):
        report={'audit':{'archive_integrity':True,'filesystem_custody_verified':True,
                         'protocol_adherence':'provider_fault'},'replay':{'refusals':[]}}
        fault={'fault':'provider_fault','stage':'provider',
               'exception_message':'Claude write is outside the candidate envelope',
               'observed_quota':{'observed':'no_notice'}}
        self.assertFalse(dispatch_must_stop(report,0,fault))
        self.assertEqual(report['audit']['protocol_adherence'],'provider_fault')
        for key,value in [('exception_message','API connection unavailable'),
                          ('stage','evaluation'),('observed_quota',{'observed':'rejected'})]:
            wrong={**fault,key:value};self.assertTrue(dispatch_must_stop(report,0,wrong))
        for field in ['archive_integrity','filesystem_custody_verified']:
            bad=copy.deepcopy(report);bad['audit'][field]=False
            self.assertTrue(dispatch_must_stop(bad,0,fault))
        self.assertTrue(dispatch_must_stop(report,1,fault))
        self.assertTrue(dispatch_must_stop(report,0,None))
        bad=copy.deepcopy(report);bad['audit']['protocol_adherence']='broker_fault'
        self.assertTrue(dispatch_must_stop(bad,0,fault))
        bad=copy.deepcopy(report);bad['replay']['refusals']=['missing archive']
        self.assertTrue(dispatch_must_stop(bad,0,fault))
        report['audit']['protocol_adherence']='adhered'
        self.assertFalse(dispatch_must_stop(report,0))
