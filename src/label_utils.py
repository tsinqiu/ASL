from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping


DEFAULT_ZH_MAP_PATH = Path("data") / "asl_label_zh_map.json"


def load_label_zh_map(path: str | Path = DEFAULT_ZH_MAP_PATH) -> dict[str, str]:
    map_path = Path(path)
    if not map_path.exists():
        return {}
    with map_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected JSON object in {map_path}, got {type(raw).__name__}")
    return {str(key): str(value) for key, value in raw.items()}


def translate_sign(sign: str, zh_map: Mapping[str, str]) -> str:
    return str(zh_map.get(sign, sign))


def format_sign_translation(sign: str, zh_map: Mapping[str, str]) -> str:
    zh = translate_sign(sign, zh_map)
    return sign if zh == sign else f"{sign} / {zh}"
