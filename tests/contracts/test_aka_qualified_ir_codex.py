from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.run_aka_qualified_ir_codex import parent_completion_schema  # noqa: E402


class AkaQualifiedIrCodexTests(unittest.TestCase):
    def test_parent_bridge_schema_is_strict_and_binds_source_coordinate(self) -> None:
        entry = {
            "parent_coordinate": "a" * 40 + ":path/to/data.jsonl:7",
            "record_field": "input",
        }
        schema = parent_completion_schema(entry)

        def assert_strict(node: object) -> None:
            if not isinstance(node, dict):
                return
            node_type = node.get("type")
            if node_type == "object":
                self.assertFalse(node.get("additionalProperties", True))
                properties = node.get("properties")
                self.assertIsInstance(properties, dict)
                self.assertEqual(set(node.get("required", [])), set(properties))
            if "const" in node:
                self.assertIn("type", node)
            for value in node.values():
                if isinstance(value, dict):
                    assert_strict(value)
                elif isinstance(value, list):
                    for item in value:
                        assert_strict(item)

        assert_strict(schema)
        original = schema["properties"]["original_parent"]["properties"]
        self.assertEqual(original["case_path"]["const"], entry["parent_coordinate"])
        self.assertEqual(original["record_field"]["const"], "input")


if __name__ == "__main__":
    unittest.main()
