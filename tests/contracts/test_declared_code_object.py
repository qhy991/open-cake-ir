"""The compiled object is declared by the Target that produces it.

F-2026-09-15-004. The Evaluation layer held three frozensets of target ids -- METAL_TARGETS,
_CUBIN_TARGETS and AMDGCN_TARGETS -- partitioning all seven targets by code object, read
from six sites across evaluation and lab, none of which any Target document could state.
What these check is that the value now comes from the document, that an undeclared object
is refused rather than guessed, and that a target this checkout never declared still
reaches the Evaluation layer's own refusal.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from open_cake_ir.compiler.target import CodeObject, Target, TargetParseError
from open_cake_ir.evaluation.artifacts import (
    builds_metal_archive, executable_role, allowed_artifact_roles)

ROOT = Path(__file__).resolve().parents[2]
TARGETS = sorted((ROOT / "compiler/targets").glob("*.json"))
FIXTURE = ROOT / "tests/fixtures/targets/neutrality_third_vendor.json"


def document(name: str) -> dict:
    return json.loads((ROOT / "compiler/targets" / f"{name}.json").read_text(encoding="utf-8"))


class DeclaredCodeObject(unittest.TestCase):
    def test_every_declared_target_states_the_object_it_is_read_with(self) -> None:
        self.assertTrue(TARGETS)
        for path in TARGETS:
            with self.subTest(target=path.stem):
                value = json.loads(path.read_text(encoding="utf-8"))
                self.assertIn("code_object", value, "the Target must state its own object")
                self.assertEqual(Target.load(path).code_object.value, value["code_object"])
                self.assertEqual(executable_role(path.stem), value["code_object"])

    def test_an_undeclared_object_is_refused_and_never_guessed(self) -> None:
        for name in ("sm_100a", "apple_gpu_family7", "gfx938"):
            value = document(name)
            del value["code_object"]
            with self.subTest(target=name):
                with self.assertRaisesRegex(TargetParseError, "target.code_object"):
                    Target.from_dict(value)

    def test_an_object_outside_the_closed_set_is_refused(self) -> None:
        value = {**document("sm_100a"), "code_object": "elf"}
        with self.assertRaisesRegex(TargetParseError, "target.code_object"):
            Target.from_dict(value)

    def test_a_target_of_a_further_vendor_reaches_the_layer_through_the_field_alone(self) -> None:
        """The absence F-2026-09-15-004's change is for.

        The fixture is declared by no Revision and named in no set here, so reaching its
        executable role at all is the property: an eighth target is a document.
        """
        target = Target.from_dict(json.loads(FIXTURE.read_text(encoding="utf-8")))
        self.assertIs(target.code_object, CodeObject.HSACO)
        self.assertEqual(executable_role(target), "hsaco")
        self.assertFalse(builds_metal_archive(target))
        self.assertIn("hsaco", allowed_artifact_roles(target))

    def test_an_id_no_document_declares_refuses_in_this_layers_words(self) -> None:
        for target in ("synthetic_third_vendor", "sm_120a", "apple_gpu_family10"):
            with self.subTest(target=target), self.assertRaises(ValueError) as raised:
                executable_role(target)
            self.assertIn("no executable role in this Evaluation layer", str(raised.exception))
            self.assertNotIn("CUDA", str(raised.exception))

    def test_the_metal_paths_read_the_declaration_rather_than_a_target_list(self) -> None:
        declared = {path.stem: Target.load(path) for path in TARGETS}
        metal = {name for name, target in declared.items()
                 if target.code_object is CodeObject.METAL_BINARY_ARCHIVE}
        self.assertEqual(metal, {name for name in declared if builds_metal_archive(name)})
        self.assertFalse(builds_metal_archive("no_such_target"))


if __name__ == "__main__":
    unittest.main()
