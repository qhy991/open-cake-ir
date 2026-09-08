"""CPU-only runtime configuration admission at its parser and real entrypoints."""
from __future__ import annotations

import copy
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.lab.bindings import CAMPAIGN_BINDING, resolve_execution_bindings
from open_cake_ir.lab.runtime import (
    CommandBrokerSubmitter,
    broker_execution_sha256,
    load_runtime_config,
)


def runtime_document(toolchain_kind="triton"):
    toolchain = (
        {"python": "runtime/python", "bubblewrap": "bin/bwrap",
         "runtime_roots": ["runtime", "/lib64"], "triton_version": "fixture",
         "timeout_seconds": 30}
        if toolchain_kind == "triton"
        else {"nvcc": "cuda/bin/nvcc", "cuobjdump": "cuda/bin/cuobjdump"}
    )
    return {
        "schema_version": 1,
        "provider": {"executable": "bin/provider", "workspace_root": "new-workspaces"},
        "toolchain": toolchain,
        "broker": {"command": ["fixture-broker", "--policy", "project_root"],
                   "cwd": ".", "timeout_seconds": 60,
                   "service_user": "fixture", "service_group": "fixture-group"},
    }


def replace_field(document, location, value):
    changed = copy.deepcopy(document)
    parent = changed
    for name in location[:-1]:
        parent = parent[name]
    parent[location[-1]] = value
    return changed


class RuntimeConfigTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="cake-runtime-config-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "runtime.json"

    def write(self, value):
        self.path.write_text(json.dumps(value), encoding="utf-8")
        return self.path

    def parse(self, value, toolchain_kind="triton"):
        return load_runtime_config(self.write(value), toolchain_kind=toolchain_kind)

    def test_triton_paths_and_guest_aliases_stay_with_the_builder(self):
        host_runtime = self.root / "host-runtime"
        host_runtime.mkdir()
        (host_runtime / "python").write_bytes(b"CPU fixture")
        guest_runtime = self.root / "guest-runtime"
        guest_runtime.symlink_to(host_runtime, target_is_directory=True)
        document = runtime_document()
        document["toolchain"]["python"] = str(guest_runtime / "python")
        document["toolchain"]["runtime_roots"] = [str(guest_runtime), "/lib64"]
        parsed = self.parse(document)
        self.assertEqual(parsed["toolchain"], document["toolchain"])
        self.assertEqual(parsed["provider"], document["provider"])
        self.assertEqual(parsed["broker"]["cwd"], ".")
        self.assertEqual(parsed["broker"]["command"], tuple(document["broker"]["command"]))
        self.assertFalse((self.root / "new-workspaces").exists())

    def test_empty_runtime_roots_remains_the_builders_admission_decision(self):
        document = runtime_document()
        document["toolchain"]["runtime_roots"] = []
        self.assertEqual(self.parse(document)["toolchain"]["runtime_roots"], [])

    def test_nvcc_preserves_both_paths_without_injecting_builder_defaults(self):
        document = runtime_document("nvcc")
        parsed = self.parse(document, "nvcc")
        self.assertEqual(parsed["toolchain"], document["toolchain"])
        self.assertEqual(set(parsed["toolchain"]), {"nvcc", "cuobjdump"})
        for selected, value in (("triton", document), ("nvcc", runtime_document())):
            with self.subTest(selected=selected), self.assertRaisesRegex(ValueError, "toolchain"):
                self.parse(value, selected)

    def test_shell_and_list_commands_preserve_the_same_literal_argv(self):
        argv = ("fixture-broker", "--label", "two words", "$(touch marker)", "semi;colon")
        shell = "fixture-broker --label 'two words' '$(touch marker)' 'semi;colon'"
        for kind in ("triton", "nvcc"):
            results = []
            for command in (list(argv), shell):
                with self.subTest(toolchain=kind, command=command):
                    document = runtime_document(kind)
                    document["broker"]["command"] = command
                    results.append(self.parse(document, kind)["broker"]["command"])
            self.assertEqual(results, [argv, argv])

    def test_schema_and_timeouts_reject_bool_and_other_noninteger_values(self):
        for location in (("schema_version",), ("toolchain", "timeout_seconds"),
                         ("broker", "timeout_seconds")):
            for value in (True, False, "1", 1.0, 0, -1, None):
                with self.subTest(field=location, value=value), self.assertRaises(ValueError):
                    self.parse(replace_field(runtime_document(), location, value))
        with self.assertRaises(ValueError):
            self.parse(replace_field(runtime_document(), ("schema_version",), 2))

    def test_path_and_named_scalars_reject_nonstring_values_without_coercion(self):
        fields = [
            ("triton", ("provider", "executable")),
            ("triton", ("provider", "workspace_root")),
            ("triton", ("toolchain", "python")),
            ("triton", ("toolchain", "bubblewrap")),
            ("triton", ("toolchain", "triton_version")),
            ("triton", ("broker", "cwd")),
            ("triton", ("broker", "service_user")),
            ("triton", ("broker", "service_group")),
            ("nvcc", ("toolchain", "nvcc")),
            ("nvcc", ("toolchain", "cuobjdump")),
        ]
        for kind, location in fields:
            for value in (True, 1, None, ["path"], {}, ""):
                with self.subTest(field=location, value=value), self.assertRaisesRegex(ValueError, "string"):
                    self.parse(replace_field(runtime_document(kind), location, value), kind)

    def test_runtime_roots_requires_a_list_of_path_strings(self):
        for value in ("/runtime", None, {}, [1], [True], [None], [""]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "runtime_roots"):
                self.parse(replace_field(runtime_document(), ("toolchain", "runtime_roots"), value))

    def test_commands_refuse_empty_or_nonstring_argv_instead_of_stringifying(self):
        for value in ("", "  ", [], True, 1, None, {}, ["broker", 1],
                      ["broker", True], ["broker", None], ["broker", {}], ["broker", ""]):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "command"):
                self.parse(replace_field(runtime_document(), ("broker", "command"), value))
        with self.assertRaises(ValueError):
            self.parse(replace_field(runtime_document(), ("broker", "command"), "broker 'unterminated"))

    def test_sections_must_be_objects_with_exact_fields(self):
        for value in (None, [], "config", 1):
            with self.subTest(top_level=value), self.assertRaises(ValueError):
                self.parse(value)
        for section in ("provider", "toolchain", "broker"):
            for value in (None, [], "section", 1):
                with self.subTest(section=section, value=value), self.assertRaisesRegex(ValueError, section):
                    self.parse(replace_field(runtime_document(), (section,), value))
        for section, field in ((None, "broker"), ("provider", "workspace_root"),
                               ("toolchain", "runtime_roots"), ("broker", "command")):
            for operation in ("missing", "extra"):
                document = runtime_document()
                target = document if section is None else document[section]
                if operation == "missing":
                    del target[field]
                else:
                    target["unexpected"] = "value"
                with self.subTest(section=section, operation=operation), self.assertRaises(ValueError):
                    self.parse(document)
        document = runtime_document("nvcc")
        document["toolchain"]["timeout_seconds"] = 600
        with self.assertRaisesRegex(ValueError, "toolchain"):
            self.parse(document, "nvcc")

    def test_unknown_toolchain_kind_is_refused(self):
        with self.assertRaisesRegex(ValueError, "toolchain kind"):
            self.parse(runtime_document(), "cuda")


class RuntimeEntryPointTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="cake-runtime-entry-")
        self.addCleanup(directory.cleanup)
        self.external = Path(directory.name).resolve()
        self.project = self.external / "project"
        self.project.mkdir()
        self.runtime_path = self.external / "runtime.json"
        qualification = self.external / "qualification.json"
        qualification.write_text("{}", encoding="utf-8")
        anchor = self.external / "anchor.json"
        anchor.write_text("{}", encoding="utf-8")
        self.bindings_path = self.external / "bindings.json"
        self.bindings_path.write_text(json.dumps({
            "schema_version": 1, "qualification_path": str(qualification),
            "qualification_anchor_path": str(anchor), "runtime_config_path": str(self.runtime_path),
            "fixed_baseline_bundle_path": str(self.external / "unused-baseline.json"),
        }), encoding="utf-8")
        provider = {name: CAMPAIGN_BINDING for name in
                    ("revision", "executable_sha256", "qualification", "qualification_anchor", "code_mode_host")}
        self.study = SimpleNamespace(state="template", document={
            "arms": {name: {"provider": copy.deepcopy(provider), "toolchain_sha256": CAMPAIGN_BINDING}
                     for name in ("open_cake", "native_triton")},
            "execution": {"executor_revision": {"binding": "current_release"},
                          "broker_execution_sha256": CAMPAIGN_BINDING, "fixed_baseline": CAMPAIGN_BINDING},
        })

    def test_binding_and_execution_reject_the_same_malformed_documents_at_shared_parser(self):
        from open_cake_ir.tasks import compose

        malformed = [
            (("schema_version",), True),
            (("broker", "timeout_seconds"), True),
            (("toolchain", "timeout_seconds"), False),
            (("broker", "command"), ["broker", 1]),
            (("provider", "workspace_root"), None),
            (("toolchain",), []),
        ]
        lock = SimpleNamespace(study_kind="matched_search", document={
            "resolved_inputs": {"arm_environments": {"open_cake": {}, "native_triton": {}}},
        })
        for location, value in malformed:
            with self.subTest(field=location, value=value):
                self.runtime_path.write_text(json.dumps(replace_field(runtime_document(), location, value)), encoding="utf-8")
                with ExitStack() as stack:
                    parser = stack.enter_context(patch("open_cake_ir.lab.runtime.load_runtime_config", wraps=load_runtime_config))
                    stack.enter_context(patch("open_cake_ir.lab.providers.ProviderQualificationReceipt.load", return_value=object()))
                    blocked = [stack.enter_context(patch(name, side_effect=AssertionError("runtime admission must precede execution"))) for name in (
                        "open_cake_ir.lab.providers.resolve_codex_code_mode_host",
                        "open_cake_ir.lab.triton_build.IsolatedTritonCompiler",
                        "open_cake_ir.lab.bindings.resolve_executor",
                        "open_cake_ir.lab.runtime.broker_execution_sha256",
                        "open_cake_ir.lab.bindings.load_baseline_bundle",
                    )]
                    with self.assertRaises(ValueError) as binding_error:
                        resolve_execution_bindings(self.project, self.study, self.bindings_path)
                    parser.assert_called_once_with(self.runtime_path, toolchain_kind="triton")
                    for operation in blocked:
                        operation.assert_not_called()
                with ExitStack() as stack:
                    stack.enter_context(patch.object(compose, "_admit_executor", return_value=(object(), None)))
                    parser = stack.enter_context(patch.object(compose, "load_runtime_config", wraps=load_runtime_config))
                    blocked = [stack.enter_context(patch.object(compose, name, side_effect=AssertionError("runtime admission must precede execution"))) for name in (
                        "IsolatedTritonCompiler", "NvccToolchainBuilder", "CommandBrokerSubmitter", "TaskLab",
                    )]
                    stack.enter_context(patch.object(compose.ProviderQualificationReceipt, "load", side_effect=AssertionError("provider qualification must follow runtime parsing")))
                    with self.assertRaises(ValueError) as execution_error:
                        compose.execute_matched_from_config(self.project, lock, self.runtime_path, self.external / "unused-evidence")
                    parser.assert_called_once_with(self.runtime_path, toolchain_kind="triton")
                    self.assertEqual(str(execution_error.exception), str(binding_error.exception))
                    for operation in blocked:
                        operation.assert_not_called()

    def test_nvcc_execution_uses_the_nvcc_parser_before_any_runtime_construction(self):
        from open_cake_ir.tasks import compose

        lock = SimpleNamespace(study_kind="matched_search", document={
            "resolved_inputs": {"arm_environments": {"open_cake": {}, "direct_cuda": {}}},
        })
        for location, value in ((("broker", "timeout_seconds"), True),
                                (("broker", "command"), ["broker", None]),
                                (("toolchain", "nvcc"), False)):
            with self.subTest(field=location):
                self.runtime_path.write_text(json.dumps(replace_field(runtime_document("nvcc"), location, value)), encoding="utf-8")
                with patch.object(compose, "_admit_executor", return_value=(object(), None)), \
                     patch.object(compose, "load_runtime_config", wraps=load_runtime_config) as parser, \
                     patch.object(compose, "NvccToolchainBuilder", side_effect=AssertionError("unexpected build")) as builder:
                    with self.assertRaises(ValueError):
                        compose.execute_matched_from_config(self.project, lock, self.runtime_path, self.external / "unused-evidence")
                    parser.assert_called_once_with(self.runtime_path, toolchain_kind="nvcc")
                    builder.assert_not_called()

    def test_direct_broker_boundaries_refuse_invalid_scalars_before_files_or_accounts(self):
        common = {"command": ("fixture-broker",), "cwd": self.project, "timeout_seconds": 60,
                  "service_user": "fixture", "service_group": "fixture-group"}
        for field, value in (("timeout_seconds", True), ("timeout_seconds", "60"),
                             ("command", ("broker", 1)), ("command", ()),
                             ("service_user", True), ("service_group", None)):
            arguments = {**common, field: value}
            with self.subTest(field=field, value=value):
                with patch("open_cake_ir.lab.runtime_config.shutil.which", side_effect=AssertionError("unexpected executable lookup")) as which:
                    with self.assertRaises(ValueError):
                        broker_execution_sha256(**arguments, project_root=self.project)
                    which.assert_not_called()
                with patch("open_cake_ir.lab.runtime.pwd.getpwnam", side_effect=AssertionError("unexpected account lookup")) as account:
                    with self.assertRaises(ValueError):
                        CommandBrokerSubmitter(**arguments, workload_path=self.external / "missing-workload.json",
                            workload_sha256="unused", protocol_sha256="unused", executor=object())
                    account.assert_not_called()


if __name__ == "__main__":
    unittest.main()
