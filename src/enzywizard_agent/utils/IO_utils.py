from __future__ import annotations

from pathlib import Path


def normalize_path(value: str | Path | None) -> str | None:
    if value is None:
        return None
    value_text = str(value).strip()
    if not value_text:
        return None
    return str(Path(value_text))
