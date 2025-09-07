from dataclasses import dataclass
from typing import Optional
import yaml

@dataclass
class SamplerCfg:
    every_nth: int = 5
    max_frames: int = 500

@dataclass
class ScoringCfg:
    w_edge: float = 0.4
    w_motion: float = 0.3
    w_blur: float = 0.3

@dataclass
class OutputCfg:
    save_heatmaps: bool = True
    save_every_n: int = 5

@dataclass
class AppCfg:
    sampler: SamplerCfg = SamplerCfg()
    scoring: ScoringCfg = ScoringCfg()
    output: OutputCfg = OutputCfg()

def load_config(path: Optional[str]) -> AppCfg:
    if path is None:
        return AppCfg()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    s = data.get("sampler", {})
    sc = data.get("scoring", {})
    o = data.get("output", {})
    return AppCfg(
        sampler=SamplerCfg(**s),
        scoring=ScoringCfg(**sc),
        output=OutputCfg(**o),
    )
