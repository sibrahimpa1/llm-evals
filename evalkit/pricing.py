"""Token cost per model. Prices live in data/pricing.json so they can be updated without a code change."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from .cases import DATA_DIR


def load_prices(path: Path = DATA_DIR / "pricing.json") -> Dict[str, Dict[str, float]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return raw["models"]


def cost_usd(model: str, input_tokens: int, output_tokens: int, prices: Dict[str, Dict[str, float]]) -> float:
    """Cost of one call. Prices are USD per million tokens. An unknown model costs 0 and is flagged by the caller."""
    p = prices.get(model)
    if p is None:
        return 0.0
    return round((input_tokens * p["input_per_mtok"] + output_tokens * p["output_per_mtok"]) / 1_000_000, 6)
