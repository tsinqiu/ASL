from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


def load_labels(path: str | Path) -> dict[int, str]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return {int(index): str(sign) for index, sign in raw.items()}


def load_zh_map(path: str | Path) -> dict[str, str]:
    map_path = Path(path)
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)
    return {str(sign): str(meaning) for sign, meaning in raw.items()}


def format_prediction(sign: str, confidence: float, zh_map: Mapping[str, str]) -> str:
    zh = zh_map.get(sign, sign)
    label = sign if zh == sign else f"{sign} / {zh}"
    return f"{label} confidence={confidence:.4f}"
