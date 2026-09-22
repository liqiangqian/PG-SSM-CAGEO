import tempfile
import unittest
from pathlib import Path

from scripts.scan_sensitive_artifacts import scan_repository


ROOT = Path(__file__).resolve().parents[1]


class SensitiveArtifactTests(unittest.TestCase):
    def test_scanner_rejects_generic_secret_coordinate_and_binary_patterns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "leak.txt").write_text("pass" + "word=not-an-example-value\n", encoding="utf-8")
            (root / "points.csv").write_text("latitude,longitude,value\n1,2,3\n", encoding="utf-8")
            (root / "weights.pkl").write_bytes(b"test")
            categories = {finding.category for finding in scan_repository(root)}
            self.assertEqual(categories, {"credential pattern", "coordinate-like columns", "sensitive extension"})

    def test_scanner_accepts_documented_example_placeholders(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text("pass" + "word=example-value\nAPI_KEY=dummy\n", encoding="utf-8")
            self.assertEqual(scan_repository(root), [])

    def test_scanner_rejects_dotenv_and_windows_absolute_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("PUBLIC_SETTING=1\n", encoding="utf-8")
            (root / "config.md").write_text("input=D:" + "\\restricted\\records.csv\n", encoding="utf-8")
            findings = scan_repository(root)
            self.assertTrue(any(item.path == ".env" and item.category == "sensitive extension" for item in findings))
            self.assertTrue(any(item.path == "config.md" and item.category == "absolute local path" for item in findings))

    def test_active_public_tree_has_no_sensitive_or_legacy_findings(self):
        self.assertEqual(scan_repository(ROOT), [])
        self.assertFalse((ROOT / "field_analysis").exists())
        self.assertFalse((ROOT / "archive").exists())


if __name__ == "__main__":
    unittest.main()
