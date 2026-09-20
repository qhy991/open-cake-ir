"""A code object's encoding is separate from its sealed launch authority."""

from dataclasses import replace
from hashlib import sha256
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from open_cake_ir.compiler import toolchain
from open_cake_ir.compiler.target import CodeObject
from open_cake_ir.evaluation.loaders import check_candidate_authority, check_launch_authority
from open_cake_ir.lab.build import _hidden_pointers


class PlatformHandoffBoundaries(unittest.TestCase):
    def test_an_integer_architecture_is_a_route_fact_not_a_cuda_test(self):
        route = replace(toolchain.route_for_code_object(CodeObject.CUBIN),
                        gpu_backend="another_integer_backend")
        with patch.object(toolchain, "_CODE_OBJECT_ROUTES", {CodeObject.CUBIN: route}):
            result = toolchain.triton_route({"target": "synthetic", "code_object": "cubin",
                                            "triton_arch": 80, "warp_size": 64})
            self.assertEqual(result.architecture, 80)
            for invalid in (True, "80", 0):
                with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, "contract differs"):
                    toolchain.triton_route({"target": "synthetic", "code_object": "cubin",
                                           "triton_arch": invalid, "warp_size": 64})

    def test_non_elf_sealed_bytes_have_authority_but_do_not_pass_an_elf_loader(self):
        payload = b"synthetic non-ELF executable format"
        manifest = SimpleNamespace(target="synthetic", kernel_name="kernel", canonical_sha256="a" * 64)
        candidate = SimpleNamespace(target="synthetic", entry_point="kernel",
            launch_spec_sha256=manifest.canonical_sha256,
            artifact_roles={"synthetic_binary": sha256(payload).hexdigest()})
        check_candidate_authority(candidate, payload, "synthetic_binary", manifest)
        with self.assertRaisesRegex(ValueError, "authority differs"):
            check_launch_authority(candidate, payload, "synthetic_binary", manifest)
        with self.assertRaisesRegex(ValueError, "authority differs"):
            check_candidate_authority(candidate, payload + b"changed", "synthetic_binary", manifest)
        with self.assertRaisesRegex(ValueError, "authority differs"):
            check_candidate_authority(candidate, payload, "another_role", manifest)

    def test_an_unimplemented_argument_inspector_cannot_inherit_cuda_scratch_pointers(self):
        with self.assertRaisesRegex(ValueError, "no kernel argument inspector"):
            _hidden_pointers(SimpleNamespace(gpu_backend="unknown"), {}, 3)


if __name__ == "__main__":
    unittest.main()
