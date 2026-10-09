"""Large prompts cross the real supervisor and launcher shim without argv growth."""
import json,subprocess,sys,tempfile,unittest
from pathlib import Path
from open_cake_ir.lab.process import run_supervised,SupervisedProcessTimeout
from open_cake_ir.lab.claude_isolation import LAUNCHER

class ProviderStdinTransport(unittest.TestCase):
    def test_large_utf8_input_is_exact_and_empty_input_remains_eof(self):
        data=('中文\\n literal $() and apostrophe\n'*24000).encode()
        with tempfile.TemporaryDirectory() as directory:
            argv=[sys.executable,'-c','import sys;sys.stdout.buffer.write(sys.stdin.buffer.read())']
            result=run_supervised(argv,cwd=directory,timeout_seconds=10,input_bytes=data)
            self.assertEqual(result.returncode,0)
            self.assertEqual(result.stdout,data)
            self.assertEqual(run_supervised(argv,cwd=directory,timeout_seconds=10).stdout,b'')
            with self.assertRaises(SupervisedProcessTimeout):
                run_supervised([sys.executable,'-c','import time;time.sleep(5)'],
                               cwd=directory,timeout_seconds=1,input_bytes=data)

    def test_actual_launcher_shim_removes_only_credential_line(self):
        namespace={'CONFIG':{'bubblewrap':'/bwrap','home_root':'/tmp/test-home',
                            'native_executable':'/claude'},'__name__':'test'}
        exec(compile(LAUNCHER,'launcher','exec'),namespace)
        from unittest.mock import patch
        env={key:'fixture' for key in ['ANTHROPIC_BASE_URL','ANTHROPIC_MODEL',
             'ANTHROPIC_DEFAULT_OPUS_MODEL','ANTHROPIC_DEFAULT_SONNET_MODEL','ANTHROPIC_DEFAULT_HAIKU_MODEL']}
        data=('large-中文-prompt\n'*30000).encode()
        with patch.dict('os.environ',env):
            argv=namespace['command'](Path('/tmp/actor'),[sys.executable,'-c',
                'import sys,os;assert os.environ["ANTHROPIC_AUTH_TOKEN"]=="fixture-token";sys.stdout.buffer.write(sys.stdin.buffer.read())'])
        shim=argv[argv.index('-c')+1]
        # Execute the same trusted shim as bubblewrap, without needing Linux mounts.
        r=subprocess.run([sys.executable,'-c',shim,*argv[argv.index('-c')+2:]],
                         input=b'fixture-token\n'+data,capture_output=True)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(r.stdout,data)
