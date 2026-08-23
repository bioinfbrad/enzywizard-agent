from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    model: str
    workdir: Path
    max_turns: int = 3
