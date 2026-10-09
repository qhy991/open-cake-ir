"""External evidence stays external; freezing never launches or revises a cohort."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("evolve_tool", ROOT / "tools/evolve.py")
evolve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evolve)


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


class DevelopmentReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.rows = []
        for task, host in (("gemm", "source"), ("norm", "destination")):
            row = dict(task=task, host=host, root=str(self.root / "runs" / task),
                       compiler="c" * 40, gateway="original", image="image", case_ids=["primary", "zeros"],
                       rows=16, columns=32, depth=None, model="model:high", reference_access="known_kernel",
                       wall_time_seconds=10800, search_seconds=9000, confirmation_seconds=1800, intaken=False)
            self.rows.append(row)
            campaign = Path(row["root"]) / "campaign"
            actual = dict(row, gateway="qualified-destination" if host == "destination" else "original")
            write(campaign / "development-binding.json", actual)
            best = dict(id="candidate", phase="search", compiler=row["compiler"], status="accepted",
                        checks=[dict(case_id=case, passed=True) for case in row["case_ids"]])
            confirmation = dict(best, id="final-confirmation", phase="confirmation")
            write(campaign / "ENDPOINT.json", {**{key: actual[key] for key in
                  ("task", "compiler", "gateway", "wall_time_seconds")}, "best": best,
                  "confirmation": confirmation, "error": None, "device_release_verified": True})
            write(campaign / "DONE.json", dict(terminal=True, status="completed", device_release_verified=True))
            write(campaign / "evaluations/candidate/outcome.json", best)
            write(campaign / "evaluations/final-confirmation/outcome.json", confirmation)
            write(campaign / "evaluations/rejected/outcome.json", dict(id="rejected", status="refused"))
        self.allocation = self.root / "allocation.json"
        write(self.allocation, dict(schema=evolve.ALLOCATION_SCHEMA, assignments=self.rows,
              source={"compiler": "c" * 40}, hosts={"source": {"role": "source", "gateway": "original"},
              "destination": {"role": "destination", "gateway": "qualified-destination"}}))

    def inspect(self, host="source"):
        return evolve.inspect_development(self.allocation, host=host)

    def change(self, relative, mutate, task="gemm"):
        path = self.root / "runs" / task / "campaign" / relative
        document = json.loads(path.read_text())
        mutate(document)
        write(path, document)

    def test_complete_review_preserves_accepted_rejected_and_gateway_owner_without_writes(self):
        before = {p: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns)
                  for p in self.root.rglob("*") if p.is_file()}
        result = self.inspect("destination")
        row = result["rows"][0]
        self.assertEqual(row["state"], "confirmed")
        self.assertEqual(row["gateway"], "qualified-destination")
        self.assertEqual(row["source_gateway"], "original")
        self.assertEqual({c["reported_status"] for c in row["candidates"]}, {"accepted", "refused"})
        self.assertFalse(result["summary"]["performance_claim"])
        self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mode, p.stat().st_mtime_ns)
                                 for p in self.root.rglob("*") if p.is_file()})

    def test_missing_terminal_never_means_safe_to_retry(self):
        (self.root / "runs/gemm/campaign/DONE.json").unlink()
        result = self.inspect()
        self.assertEqual(result["rows"][0]["state"], "incomplete")
        self.assertFalse(result["summary"]["ready_for_maintenance_review"])

    def test_completed_label_cannot_override_case_loss_or_outcome_conflict(self):
        self.change("ENDPOINT.json", lambda d: d["confirmation"]["checks"].pop())
        result = self.inspect()
        self.assertEqual(result["rows"][0]["state"], "invalid")
        self.assertIn("confirmation accepted without all assigned cases", result["rows"][0]["issues"])

    def test_compiler_conflict_and_release_failure_remain_visible(self):
        self.change("development-binding.json", lambda d: d.update(compiler="different"))
        self.assertEqual(self.inspect()["rows"][0]["state"], "invalid")
        self.change("development-binding.json", lambda d: d.update(compiler="c" * 40))
        self.change("DONE.json", lambda d: d.update(device_release_verified=False))
        self.assertEqual(self.inspect()["rows"][0]["state"], "release_pending")

    def test_released_failure_closes_discovery_but_is_not_confirmed(self):
        self.change("DONE.json", lambda d: d.update(status="failed"))
        self.change("ENDPOINT.json", lambda d: d.update(confirmation=None, error={"reason": "no candidate"}))
        result = self.inspect()
        self.assertEqual(result["rows"][0]["state"], "failed")
        self.assertTrue(result["summary"]["ready_for_maintenance_review"])

    def test_malformed_one_task_does_not_erase_other_assignments(self):
        self.change("ENDPOINT.json", lambda d: d.update(search_close=[]))
        # A truthy malformed record is unclassifiable, while empty means no closure.
        self.change("ENDPOINT.json", lambda d: d.update(search_close=["bad"]))
        self.assertEqual(self.inspect()["rows"][0]["state"], "invalid")

    def test_merge_keeps_missing_and_refuses_duplicate_or_changed_controls(self):
        report = self.root / "review.json"
        write(report, self.inspect())
        merged = evolve.merge_development(self.allocation, [report])
        self.assertEqual(merged["summary"]["assigned"], 2)
        self.assertEqual(merged["summary"]["states"], {"confirmed": 1, "unobserved": 1})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evolve.merge_development(self.allocation, [report, report])
        data = self.inspect()
        data["rows"][0]["compiler"] = "wrong"
        write(report, data)
        with self.assertRaisesRegex(ValueError, "controls"):
            evolve.merge_development(self.allocation, [report])

    def test_merge_checks_host_and_assignment_uniqueness(self):
        data = self.inspect()
        data["host"] = "destination"
        report = self.root / "review.json"
        write(report, data)
        with self.assertRaisesRegex(ValueError, "another host"):
            evolve.merge_development(self.allocation, [report])
        allocation = evolve.read_object(self.allocation)
        allocation["assignments"].append(allocation["assignments"][0])
        write(self.allocation, allocation)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.inspect()

    def test_destination_cannot_handoff_an_intaken_run(self):
        allocation = evolve.read_object(self.allocation)
        allocation["assignments"][1]["intaken"] = True
        write(self.allocation, allocation)
        with self.assertRaisesRegex(ValueError, "already intaken"):
            self.inspect("destination")


class BenchFreezeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkouts = {}
        for name in ("bench", "control", "successor"):
            repo = self.root / name
            repo.mkdir()
            self.checkouts[name] = repo
            write(repo / "compiler/revision.json", {"name": name})
            if name == "bench":
                write(repo / "suite.json", {"target_arch": "gfx938", "tasks": [{"id": "task"}, {"id": "task2"}]})
                (repo / "baseline.py").write_text("# fixture reference\n")
                (repo / "protocol.md").write_text("Fixture paired confirmation protocol.\n")
            self.git(repo, "init", "-q")
            self.git(repo, "add", "compiler/revision.json", *(["suite.json", "baseline.py", "protocol.md"] if name == "bench" else []))
            self.git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", name)
        self.draft = dict(hypothesis="The successor helps fresh search", task_ids=["task"], replicates=2,
                          allocation_seed=1, author=dict(model="model:high", scaffold="fixture@commit",
                          knowledge="none", reference_access="known_kernel", wall_time_seconds=10800,
                          confirmation_seconds=1800), measurement_protocol="protocol.md",
                          baseline_files={"task": "baseline.py"}, total_author_wall_seconds=43200)
        self.path = self.root / "draft.json"
        self.output = self.root / "experiment/plan.json"

    @staticmethod
    def git(root, *args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.PIPE, text=True)

    def freeze(self):
        write(self.path, self.draft)
        return evolve.freeze_bench(self.path, **self.checkouts, output=self.output)

    def test_freeze_binds_all_versions_and_allocations_without_creating_runs(self):
        result = self.freeze()
        self.assertEqual(len(result["allocations"]), 4)
        self.assertEqual({r["condition"] for r in result["allocations"]}, {"control", "successor"})
        self.assertFalse((self.output.parent / "runs").exists())
        self.assertEqual(result["bindings"]["bench"]["commit"], self.git(self.checkouts["bench"], "rev-parse", "HEAD").strip())
        before = self.output.read_bytes()
        self.assertEqual(self.freeze(), result)
        self.assertEqual(self.output.read_bytes(), before)
        self.draft["allocation_seed"] = 2
        with self.assertRaisesRegex(ValueError, "conflicting"):
            self.freeze()
        self.assertEqual(self.output.read_bytes(), before)

    def test_existing_workspaces_prevent_posthoc_freeze(self):
        (self.output.parent / "runs/t001-r1-control").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "precedes"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_two_plans_cannot_assign_the_same_cohort_workspaces(self):
        first = self.freeze()
        self.output = self.output.with_name("second-plan.json")
        self.draft["hypothesis"] = "A different comparison"
        with self.assertRaisesRegex(ValueError, "one plan.json"):
            self.freeze()
        self.output = self.root / "second-cohort/plan.json"
        second = self.freeze()
        self.assertFalse({r["workspace"] for r in first["allocations"]} &
                         {r["workspace"] for r in second["allocations"]})

    def test_dirty_sources_refuse_before_output(self):
        (self.checkouts["successor"] / "compiler/revision.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "clean"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_untracked_suite_cannot_supply_a_frozen_population(self):
        repo = self.checkouts["bench"]
        self.git(repo, "rm", "--cached", "suite.json")
        (repo / ".gitignore").write_text("suite.json\n")
        self.git(repo, "add", ".gitignore")
        self.git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Untrack suite")
        with self.assertRaises(subprocess.CalledProcessError):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_tracked_symlink_cannot_pin_an_external_baseline(self):
        repo = self.checkouts["bench"]
        external = self.root / "mutable-baseline.py"
        external.write_text("# mutable external reference\n")
        (repo / "baseline.py").unlink()
        (repo / "baseline.py").symlink_to(external)
        self.git(repo, "add", "baseline.py")
        self.git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Link baseline")
        with self.assertRaisesRegex(ValueError, "not a file"):
            self.freeze()
        self.assertFalse(self.output.exists())

    def test_no_source_change_does_not_manufacture_version_comparison(self):
        self.checkouts["successor"] = self.checkouts["control"]
        with self.assertRaisesRegex(ValueError, "No promotion"):
            self.freeze()

    def test_task_baseline_budget_and_token_controls_are_admitted_before_freeze(self):
        original = copy.deepcopy(self.draft)
        changes = [lambda d: d.update(task_ids=["unknown"]),
                   lambda d: d.update(task_ids=["task", "task"]),
                   lambda d: d.update(baseline_files={}),
                   lambda d: d.update(total_author_wall_seconds=1),
                   lambda d: d["author"].update(token_limit=100),
                   lambda d: d["author"].update(confirmation_seconds=10800),
                   lambda d: d.update(baseline_files={"task": "../outside.py"})]
        for change in changes:
            with self.subTest(change=change):
                self.draft = copy.deepcopy(original)
                change(self.draft)
                with self.assertRaises(ValueError):
                    self.freeze()
                self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
