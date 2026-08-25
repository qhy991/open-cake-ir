"""Successor Compiler revisions must preserve every v13 Corpus observation.

The digests below were produced from detached `origin/main` at 13b04cc with released
Compiler v13.  Revision identity is intentionally excluded; everything a Schedule author
or Lowering consumer observes is included, including ordered Findings, analysis,
generated source digest, source map and toolchain requirements.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import unittest
from pathlib import Path

from open_cake_ir.compiler import Compiler


ROOT = Path(__file__).resolve().parents[2]
REVISION_PATH = ROOT / "compiler/revision.lock.json"

V13_OBSERVATION_SHA256 = {
    "flash-kmeans-r16-accepted": "b75b328bf403d2b14d44c95ace563382be563aa9f3b0e6a6d0a2748bbbcf9e92",
    "flash-kmeans-r16-shape-drift": "30decf1278e72cdae209e89da59fb844966b750b797c0818a12e40fdaf0a9598",
    "flash-kmeans-r25-accepted": "c1d65b493facf252d2c3c6127922fdee7123a81071233b9651fe943185a7c4c5",
    "flash-kmeans-r25-shape-drift": "cbdbc0393e49b38aaffa2b60d7cd8e495562540e416e2bd0136a2a8efa3c4a12",
    "flash-kmeans-warp-specialized-argmin": "b6fdb05001ea2bf7cca204bf567a6b7c6ff979376ec0b08fd12f96d4398746dd",
    "gemm-bias-accepted": "03b9fc67d6746a7f17edd1d58f54f9443acb34a164b9ceb9472774dc821f7b91",
    "gemm-bias-shape-drift": "117704c297ce52bb02ffd824c753f870eab709afb92a8cb0b2916d6ed361eea7",
    "layernorm-accepted": "a6bf8ff8dc586c893a0ee0fc3823a92507d3ecc036c2e1748b892c02c029fb2a",
    "layernorm-shape-drift": "7c303cfbbc38605f02fe9922b690d224f126913d85c7ee41bd8d1b52f548979a",
    "rmsnorm-accepted": "acb4c54ec513972566a6b454f6c67e80660feb5c5d995fe7f2ae1ff224b7e32b",
    "rmsnorm-persistent": "ae920be09ec50a4c3b2a5a27090c8289e73bd878515ca842607a7fbae16a0540",
    "rmsnorm-residency-unmet": "e48b9234b9ed23a2d06b149c9eef3735bc33917ec4933c6e328c5a0ede0a5116",
    "rmsnorm-shape-drift": "becab9657853797960fe7c07ee4dd670475f191870f5769fd724730f5acb0a36",
    "softmax-accepted": "ba5875741e1dd0df4c186ff182fba12ad089d03a5e1aeaef21e5308693962ac7",
    "softmax-shape-drift": "2eee7d502ee9a1a331d4bfa6ad876dc55235f06eb779b6d01054aa0465390460",
    "swiglu-accepted": "819763e5c423bab2e148e7cce861b90526a796aec3785a6cde4db3b1fadc7ac2",
    "swiglu-shape-drift": "e3998d1bc3dc75d0134e85c82fe4f7c1f060467907dd2191a541470107df5e2e",
    "tinygemm2-r31-accepted": "4695f4d53028021c506b321118c1b45e1eaab8e2d6daa9d02e1999371cbc41cd",
    "tinygemm2-r31-reduction-drift": "ebe7a4b86b6c2c5d9eeaca650808c171174cfe10088eebe92dea05a5972015e4",
}


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _projection(compiler: Compiler, schedule_path: Path) -> dict[str, object]:
    assessment = compiler.assess_file(schedule_path)
    result: dict[str, object] = {
        "schedule_id": assessment.schedule_id,
        "schedule_sha256": assessment.schedule_sha256,
        "target": assessment.target,
        "profile": assessment.profile,
        "accepted": assessment.accepted,
        "lowering_eligible": assessment.lowering_eligible,
        "findings": [dataclasses.asdict(item) for item in assessment.findings],
        "analysis": dict(assessment.analysis),
        "lowering_parameters": dict(assessment.lowering_parameters),
        "calibration_available": assessment.calibration_available,
    }
    if assessment.lowering_eligible:
        lowering = compiler.lower(assessment)
        result["lowering"] = {
            "target": lowering.target,
            "profile": lowering.profile,
            "generated": lowering.generated,
            "entry_point": lowering.entry_point,
            "source_sha256": lowering.source_sha256,
            "source_map": dict(lowering.source_map),
            "toolchain_requirements": dict(lowering.toolchain_requirements),
        }
    return result


class V13CompatibilityTest(unittest.TestCase):
    def test_successor_preserves_all_nineteen_v13_observations(self) -> None:
        compiler = Compiler.load(ROOT, REVISION_PATH)
        manifest = json.loads((ROOT / "corpus/manifest.json").read_text(encoding="utf-8"))
        cases = {case["case_id"]: case for case in manifest["cases"]}

        self.assertTrue(V13_OBSERVATION_SHA256.keys() <= cases.keys())
        for case_id, expected in V13_OBSERVATION_SHA256.items():
            with self.subTest(case_id=case_id):
                observed = _projection(compiler, ROOT / cases[case_id]["schedule"])
                self.assertEqual(_digest(observed), expected)


if __name__ == "__main__":
    unittest.main()
