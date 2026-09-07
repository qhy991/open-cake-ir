"""CPU probes for the selected Code Mode host and its frozen provider identity."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from open_cake_ir.lab.providers import (
    CODEX_DISABLED_FEATURES, CodexInvocationBuilder, CodexRunProvider,
    ProviderQualificationReceipt, resolve_codex_code_mode_host,
)
from open_cake_ir.lab.task_package import TaskPackage, materialize_task_package, render_task_request


class ProviderRuntimeContractTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.executable = self.write(self.root / "codex", b"CPU native CLI fixture")
        self.helper = self.write(self.root / "codex-code-mode-host", b"CPU host fixture")
        self.schema = self.root / "schema.json"
        self.schema.write_text("{}")

    def write(self, path, payload=b"CPU host fixture"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        path.chmod(0o700)
        return path

    def builder(self, executable=None, *, workspace=None, **kwargs):
        return CodexInvocationBuilder(
            executable=executable or self.executable,
            provider_revision="cpu-code-mode-fixture", model="gpt-5.6-sol",
            reasoning_effort="max", service_tier="default",
            workspace=workspace or self.root, output_schema=self.schema,
            removed_environment=("OPENAI_API_KEY", "ANTHROPIC_API_KEY"), **kwargs,
        )

    def test_standalone_sibling_is_selected_without_path_fallback(self):
        observed = resolve_codex_code_mode_host(self.executable)
        self.assertEqual(observed, {"path": str(self.helper), "sha256": sha256(self.helper.read_bytes()).hexdigest()})
        self.helper.unlink()
        elsewhere = self.write(self.root / "path-only/codex-code-mode-host")
        with patch.dict(os.environ, {"PATH": str(elsewhere.parent)}):
            with self.assertRaisesRegex(ValueError, "host is missing"):
                resolve_codex_code_mode_host(self.executable)

    def test_package_resources_precede_bin_and_require_package_metadata(self):
        executable = self.write(self.root / "package/bin/codex")
        sibling = self.write(executable.with_name("codex-code-mode-host"), b"sibling")
        resource = self.write(self.root / "package/codex-resources/codex-code-mode-host", b"preferred")
        self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(sibling))
        (self.root / "package/codex-package.json").write_text('{"version":"0.153.4"}')
        self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(resource))
        resource.unlink()
        self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(sibling))

    def test_package_cli_in_resources_and_macos_bundle_use_package_bin(self):
        package = self.root / "package"
        sibling = self.write(package / "bin/codex-code-mode-host")
        (package / "codex-package.json").write_text('{}')
        for relative in ("codex-resources/codex", "CodexCLI.app/Contents/MacOS/codex"):
            executable = self.write(package / relative)
            self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(sibling))

    def test_legacy_resources_follow_codex_home_and_install_method(self):
        codex_home = self.root / "home"
        executable = self.write(codex_home / "packages/standalone/releases/0.153.4/codex")
        sibling = self.write(executable.with_name("codex-code-mode-host"), b"sibling")
        resource = self.write(executable.parent / "codex-resources/codex-code-mode-host", b"resource")
        with patch.dict(os.environ, {"CODEX_HOME": str(codex_home)}, clear=True):
            self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(resource))
            with patch.dict(os.environ, {"CODEX_MANAGED_BY_NPM": "1"}):
                self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(sibling))
        with patch.dict(os.environ, {"CODEX_HOME": str(self.root)}, clear=True):
            self.assertEqual(resolve_codex_code_mode_host(executable)["path"], str(sibling))

    def test_explicit_invalid_codex_home_refuses_binding_and_both_turns(self):
        with patch("open_cake_ir.lab.providers.sanitized_environment", return_value={}):
            builder = self.builder()
        for value in ("home", ".", "", str(self.root / "missing"), str(self.executable), "/bad\x00home"):
            with self.subTest(home=value), patch(
                "open_cake_ir.lab.providers.sanitized_environment", return_value={"CODEX_HOME": value},
            ), patch("open_cake_ir.lab.providers.os.open") as open_helper:
                with self.assertRaisesRegex(ValueError, "CODEX_HOME.*absolute directory"):
                    self.builder()
                for thread in (None, "01234567-89ab-cdef-0123-456789abcdef"):
                    with self.assertRaisesRegex(ValueError, "CODEX_HOME.*absolute directory"):
                        builder.build("request", thread_id=thread)
                open_helper.assert_not_called()

    def test_absolute_codex_home_selects_same_helper_across_invocation_cwds(self):
        codex_home = self.root / "home"
        executable = self.write(codex_home / "packages/standalone/releases/0.153.4/codex")
        self.write(executable.with_name("codex-code-mode-host"), b"sibling")
        resource = self.write(executable.parent / "codex-resources/codex-code-mode-host", b"resource")
        workspace = self.root / "workspace"
        workspace.mkdir()
        original_cwd = Path.cwd()
        try:
            with patch("open_cake_ir.lab.providers.sanitized_environment",
                       return_value={"CODEX_HOME": str(codex_home)}):
                os.chdir(self.root)
                builder = self.builder(executable, workspace=workspace)
                self.assertEqual(builder.configuration["code_mode_host"]["path"], str(resource))
                for thread in (None, "01234567-89ab-cdef-0123-456789abcdef"):
                    invocation = builder.build("request", thread_id=thread)
                    os.chdir(invocation.cwd)
                    self.assertEqual(resolve_codex_code_mode_host(executable),
                                     builder.configuration["code_mode_host"])
                    os.chdir(self.root)
        finally:
            os.chdir(original_cwd)

    def test_unsafe_selected_helpers_fail_closed(self):
        for mode in (0o600, 0o722):
            with self.subTest(mode=mode):
                self.helper.chmod(mode)
                with self.assertRaisesRegex(ValueError, "custody or executable mode"):
                    resolve_codex_code_mode_host(self.executable)
        self.helper.chmod(0o700)
        alias = self.root / "alias"
        os.link(self.helper, alias)
        with self.assertRaisesRegex(ValueError, "custody"):
            resolve_codex_code_mode_host(self.executable)
        alias.unlink()
        self.helper.unlink()
        self.helper.symlink_to(self.executable)
        with self.assertRaisesRegex(ValueError, "symlink"):
            resolve_codex_code_mode_host(self.executable)
        self.helper.unlink()
        self.helper.symlink_to(self.root / "missing")
        with self.assertRaisesRegex(ValueError, "symlink"):
            resolve_codex_code_mode_host(self.executable)

    def test_preferred_symlink_directory_cannot_redirect_lookup(self):
        executable = self.write(self.root / "package/bin/codex")
        self.write(executable.with_name("codex-code-mode-host"))
        (self.root / "package/codex-package.json").write_text('{}')
        (self.root / "package/codex-resources").symlink_to(self.root, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "not canonical"):
            resolve_codex_code_mode_host(executable)

    def test_each_initial_and_resume_rejects_missing_or_mutated_bound_host(self):
        for thread in (None, "01234567-89ab-cdef-0123-456789abcdef"):
            for mutation in ("missing", "changed"):
                with self.subTest(thread=thread, mutation=mutation):
                    self.write(self.helper)
                    builder = self.builder()
                    if mutation == "missing":
                        self.helper.unlink()
                    else:
                        self.helper.write_bytes(b"different host bytes")
                    with self.assertRaisesRegex(ValueError, "host .*missing|bound runtime"):
                        builder.build("request", thread_id=thread)

    def test_new_preferred_helper_rejects_initial_and_resume_even_with_same_bytes(self):
        executable = self.write(self.root / "package/bin/codex")
        sibling = self.write(executable.with_name("codex-code-mode-host"))
        (self.root / "package/codex-package.json").write_text('{}')
        preferred = self.root / "package/codex-resources/codex-code-mode-host"
        for thread in (None, "01234567-89ab-cdef-0123-456789abcdef"):
            builder = self.builder(executable)
            self.write(preferred, sibling.read_bytes())
            with self.assertRaisesRegex(ValueError, "bound runtime"):
                builder.build("request", thread_id=thread)
            preferred.unlink()

    def test_expected_host_is_an_identity_check_and_never_an_override(self):
        elsewhere = self.write(self.root / "unrelated/codex-code-mode-host", self.helper.read_bytes())
        identity = resolve_codex_code_mode_host(self.executable)
        for mismatch in ({**identity, "path": str(elsewhere)}, {**identity, "sha256": "0" * 64}, {}):
            with self.subTest(mismatch=mismatch), self.assertRaisesRegex(ValueError, "bound runtime"):
                self.builder(code_mode_host=mismatch)

    def test_configuration_receipt_and_both_arms_bind_the_host(self):
        builders = {}
        packages = {}
        for arm in ("open_cake", "native_triton"):
            workspace = self.root / arm
            workspace.mkdir()
            package = TaskPackage(arm + "-1", arm, "# TASK.md\nFixed task.\n", "# AGENTS.md\nRules.\n")
            materialize_task_package(workspace, package)
            packages[package.run_id] = package
            builders[package.run_id] = self.builder(workspace=workspace)
        builder = next(iter(builders.values()))
        receipt = ProviderQualificationReceipt(
            provider_revision=builder.provider_revision,
            executable_sha256=sha256(self.executable.read_bytes()).hexdigest(),
            configuration_sha256=builder.configuration_sha256,
            initial_and_resume_equivalent=True, file_lifecycle_observed=True,
            usage_observed=True, qualified=True, scope="live_two_turn_current_provider",
        )
        provider = CodexRunProvider(qualification=receipt, builders=builders, task_packages=packages)
        self.assertEqual(provider.configuration["code_mode_host"], resolve_codex_code_mode_host(self.executable))
        old_configuration = dict(builder.configuration)
        old_configuration.pop("code_mode_host")
        old_configuration.pop("web_search")
        old_receipt = replace(receipt, configuration_sha256=sha256(json.dumps(old_configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        with self.assertRaisesRegex(ValueError, "configuration differs from provider qualification"):
            CodexRunProvider(qualification=old_receipt, builders=builders, task_packages=packages)
        self.helper.write_bytes(b"changed second arm")
        builders["native_triton-1"] = self.builder(workspace=self.root / "native_triton")
        with self.assertRaisesRegex(ValueError, "builder configurations differ"):
            CodexRunProvider(qualification=receipt, builders=builders, task_packages=packages)

    def test_closed_flags_and_ralph_projection_are_equal_on_initial_and_resume(self):
        builder = self.builder()
        package = TaskPackage("open_cake-1", "open_cake", "# TASK.md\n中文\n", "# AGENTS.md\nRules\n")
        common = []
        for turn, thread in ((1, None), (2, "01234567-89ab-cdef-0123-456789abcdef")):
            prompt, bundle = render_task_request(package, {"turn": turn})
            invocation = builder.build(prompt, thread_id=thread)
            self.assertEqual(invocation.argv[-1], prompt)
            delivered = json.loads(bundle)
            self.assertEqual(delivered["task_markdown"], package.task_markdown)
            self.assertEqual(delivered["agents_markdown"], package.agents_markdown)
            self.assertEqual(delivered["state_card"], {"turn": turn})
            common.append(invocation.argv[2:-1] if thread is None else invocation.argv[3:-2])
        self.assertEqual(*common)
        argv = common[0]
        self.assertIn("--ignore-user-config", argv)
        self.assertIn("--ignore-rules", argv)
        self.assertIn('web_search="disabled"', argv)
        self.assertIn('sandbox_mode="workspace-write"', argv)
        self.assertEqual(argv[argv.index("--enable") + 1], "code_mode_host")
        disabled = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--disable"]
        self.assertEqual(disabled, list(CODEX_DISABLED_FEATURES))
        for feature in ("shell_tool", "browser_use", "apps", "plugins", "hooks", "multi_agent"):
            self.assertIn(feature, disabled)
        for feature in ("code_mode", "code_mode_only", "code_mode_host"):
            self.assertNotIn(feature, disabled)
        self.assertEqual(builder.configuration["web_search"], "disabled")
        self.assertEqual(builder.configuration["model"], "gpt-5.6-sol")
        self.assertEqual(builder.configuration["reasoning_effort"], "max")
        self.assertEqual(builder.configuration["service_tier"], "default")


if __name__ == "__main__":
    unittest.main()
