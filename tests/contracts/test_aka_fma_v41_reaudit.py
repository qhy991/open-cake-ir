from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

from tools.verify_aka_fma_v41_reaudit import verify

ROOT=Path(__file__).resolve().parents[2]

class AkaFmaV41ReauditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch=tempfile.TemporaryDirectory(prefix="aka-v41-source-")
        cls.compiler_root=Path(cls.scratch.name)/"source"
        subprocess.run(["git","clone","--shared","--no-checkout","--quiet",str(ROOT),str(cls.compiler_root)],
                       check=True,capture_output=True)
        subprocess.run(["git","-C",str(cls.compiler_root),"checkout","--quiet","--detach",
                        "d9d56e835cf96eecf70e0259b65bc1b20c4f6f0d"],check=True,capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.scratch.cleanup()

    def test_published_projection_passes_its_verifier(self):
        result=verify(ROOT/"docs/data/aka-fma-v41-reaudit-20260906",self.compiler_root)
        self.assertEqual(result["status"],"accepted")
        self.assertEqual(result["cases"],12)
        self.assertEqual(result["semantic_survivors"],4)
        self.assertEqual(result["runtime_scalar_blocked"],5)
        self.assertEqual(result["gpu_test"],"not_run")
        self.assertFalse(result["performance_measured"])
        self.assertEqual(result["verification_scope"],"projection_and_frozen_compiler_replay_only")

    def test_current_compiler_cannot_be_relabelled_as_historical_v41(self):
        with self.assertRaisesRegex(ValueError,"v41 replay source differs"):
            verify(ROOT/"docs/data/aka-fma-v41-reaudit-20260906",ROOT)

    def test_summary_claim_promotion_is_refused(self):
        for key,value in (("gpu_test","passed"),("performance_measured",True)):
            with self.subTest(key=key),tempfile.TemporaryDirectory() as directory:
                data=Path(directory)/"data"
                shutil.copytree(ROOT/"docs/data/aka-fma-v41-reaudit-20260906",data)
                summary=json.loads((data/"summary.json").read_text());summary[key]=value
                (data/"summary.json").write_text(json.dumps(summary))
                with self.assertRaisesRegex(ValueError,"summary claim boundary"):
                    verify(data,self.compiler_root)

if __name__=="__main__":unittest.main()
