"""Non-ASCII authority documents must cross runtime boundaries unchanged."""
from hashlib import sha256
import json
from types import SimpleNamespace
import unittest

from open_cake_ir.serialization import canonical_json_bytes
from open_cake_ir.lab.bindings import canonical
from open_cake_ir.lab.environments import OpenCakeEnvironment
from open_cake_ir.lab.provider_documents import ProviderQualificationReceipt
from open_cake_ir.tasks.flash_kmeans.environment import DirectCudaEnvironment


class RuntimeSerializationTests(unittest.TestCase):
    def test_non_ascii_authority_identity_matches_preflight_for_each_authoring_kind(self):
        authority = {"notes": "基线", "lowering_route": {"backend": "metal", "entry_point": "kernel"}}
        workload = SimpleNamespace(canonical_sha256="a" * 64, target="apple_gpu_family7",
                                   document={"semantics": {}}, tensor_abi=lambda _: [])
        cake = OpenCakeEnvironment(SimpleNamespace(state="released"), object(),
            authority_document=authority, workload=workload, case_id="fixture")
        direct = DirectCudaEnvironment(object(), toolchain_requirements={}, authority_document=authority)
        wire = json.dumps(authority, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
        expected = sha256(wire).hexdigest()
        self.assertEqual(cake.canonical_sha256, expected)
        self.assertEqual(direct.canonical_sha256, expected)
        self.assertEqual(canonical(authority), wire)

    def test_qualification_receipt_uses_the_same_utf8_wire_form(self):
        receipt = ProviderQualificationReceipt("中文提供器", "a" * 64, "b" * 64,
            True, True, True, True, "zero_gpu_contract_fixture_only")
        wire = json.dumps(receipt.document, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.assertEqual(receipt.canonical_sha256, sha256(wire).hexdigest())

    def test_ascii_encoding_stays_compatible_and_nonfinite_values_are_refused(self):
        value = {"notes": "baseline", "values": [1, False, None]}
        self.assertEqual(canonical_json_bytes(value),
                         json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
        for invalid in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                canonical_json_bytes({"value": invalid})
