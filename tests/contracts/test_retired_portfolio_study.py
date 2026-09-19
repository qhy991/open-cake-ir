"""Retired Study documents fail at their own boundary before live dependencies."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from open_cake_ir.lab.contracts import CampaignLock, StudyContract
from open_cake_ir.tasks.runtime import TaskLab

ROOT = Path(__file__).resolve().parents[2]


class RetiredPortfolioStudyTests(unittest.TestCase):
    def test_historical_template_is_refused_before_dependency_resolution(self):
        retained = subprocess.check_output(['git', 'show',
            'ba7c48b1:contracts/studies/flash-kmeans-r45-portfolio-reconstruction-template.json'], cwd=ROOT)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'study.json'
            path.write_bytes(retained)
            with patch('open_cake_ir.lab.preflight.resolve_execution_bindings') as resolve:
                with self.assertRaisesRegex(ValueError, "Study kind 'portfolio' is retired"):
                    TaskLab(ROOT).preflight(path)
                resolve.assert_not_called()

    def test_lock_names_retired_kind_before_loading_its_old_references(self):
        # The old document envelope remains recognizable even with no live dependency files.
        lock = dict.fromkeys(('study', 'workload', 'compiler_revision', 'resolved_inputs',
                              'run_order', 'evaluation_protocol', 'execution', 'analysis_plan',
                              'analysis_plan_sha256'))
        lock.update(schema_version=1, study={'kind': 'portfolio'})
        with self.assertRaisesRegex(ValueError, "Campaign Lock Study kind 'portfolio' is retired"):
            CampaignLock.from_dict(lock)

    def test_unknown_kind_is_not_mislabelled_as_the_retired_variant(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'study.json'
            path.write_text(json.dumps({'kind': 'unknown'}))
            with self.assertRaisesRegex(ValueError, "Study kind 'unknown' is unsupported"):
                StudyContract.load(path)
