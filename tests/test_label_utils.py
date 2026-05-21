import json
import unittest
from pathlib import Path

from src.label_utils import load_label_zh_map, translate_sign


class LabelUtilsTest(unittest.TestCase):
    def test_translate_sign_uses_zh_map_and_falls_back_to_english(self) -> None:
        zh_map = {"wait": "等待"}

        self.assertEqual(translate_sign("wait", zh_map), "等待")
        self.assertEqual(translate_sign("unknown_sign", zh_map), "unknown_sign")

    def test_load_label_zh_map_reads_json_as_string_map(self) -> None:
        path = Path("outputs") / "test_label_utils_labels.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"wait": "等待"}, ensure_ascii=False), encoding="utf-8")

        self.assertEqual(load_label_zh_map(path), {"wait": "等待"})


if __name__ == "__main__":
    unittest.main()
