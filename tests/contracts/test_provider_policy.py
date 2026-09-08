"""Treatment projection tests; no provider qualification or live invocation."""
import json
from pathlib import Path
import unittest

from open_cake_ir.lab.provider_policy import provider_configuration
from open_cake_ir.lab.claude import CLAUDE_EVENT_CONTRACT, CLAUDE_AUTHORING_TOOLS, terminal_schema

ROOT = Path(__file__).resolve().parents[2]


class ProviderPolicyTests(unittest.TestCase):
    def test_codex_model_and_effort_are_explicit_factors_not_hardcoded_sol(self):
        p = json.loads((ROOT / "contracts/studies/artifact-optimization-ralph-template.json").read_text())["arms"]["open_cake"]["provider"]
        p.update(model="gpt-6-astra", reasoning_effort="high")
        config = provider_configuration(p, "artifact_optimization_only", arms={"open_cake"})
        self.assertEqual(config["model"], "gpt-6-astra")
        self.assertEqual(config["reasoning_effort"], "high")
        for name in ("model", "reasoning_effort"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                provider_configuration({**p, name: ""}, "artifact_optimization_only", arms={"open_cake"})

    def claude(self):
        return {"revision": "fixture", "qualification": {}, "qualification_anchor": {}, "executable_sha256": "a" * 64,
            "harness": "claude-code", "model": "claude-fable-5", "reasoning_effort": "high",
            "sandbox": "none", "permission_mode": "acceptEdits", "safe_mode": True,
            "tools": list(CLAUDE_AUTHORING_TOOLS), "event_contract": CLAUDE_EVENT_CONTRACT,
            "terminal_schema": terminal_schema(),
            "cwd_policy": "independent_task_workspace", "reference_visibility": "workspace_task_files",
            "removed_environment": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]}

    def test_claude_is_only_admitted_as_declared_python_artifact_author(self):
        p = self.claude()
        config = provider_configuration(p, "artifact_optimization_only", arms={"open_cake"})
        self.assertEqual(config["sandbox"], "none")
        self.assertEqual(config["permission_mode"], "acceptEdits")
        self.assertEqual(config["terminal_schema"], terminal_schema())
        self.assertNotIn("code_mode_host", config)
        for scope, arms, change in (
            ("scientific_matched_search", {"open_cake", "direct_cuda"}, {}),
            ("artifact_optimization_only", {"open_cake", "native_triton"}, {}),
            ("artifact_optimization_only", {"open_cake"}, {"sandbox": "workspace-write"}),
            ("artifact_optimization_only", {"open_cake"}, {"tools": ["Bash"]}),
            ("artifact_optimization_only", {"open_cake"}, {"terminal_schema": {}}),
            ("artifact_optimization_only", {"open_cake"}, {"event_contract": "claude_stream_candidate_v1"}),
            ("artifact_optimization_only", {"open_cake"}, {"reasoning_effort": ""}),
        ):
            with self.subTest(scope=scope, arms=arms, change=change), self.assertRaises(ValueError):
                provider_configuration({**p, **change}, scope, arms=arms)


if __name__ == "__main__":
    unittest.main()
