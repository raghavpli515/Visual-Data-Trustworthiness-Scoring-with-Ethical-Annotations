import os, json
from typing import Dict, Any, List, Tuple, Optional

import cv2
import numpy as np

from ..config import AppCfg
from ..utils.video_io import read_frames
from ..scoring.heuristics import edge_energy, blur_score, motion_inconsistency
from ..scoring.calibration import calibrate
from ..explain.visual import save_edge_heatmap
from ..explain.text import textual_reasons

def infer_video(path: str, out_dir: str, cfg: AppCfg) -> Dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    frames: List[Dict[str, Any]] = []
    prev_frame: Optional[np.ndarray] = None

    for idx, frame in read_frames(path, cfg.sampler.every_nth, cfg.sampler.max_frames):
        e = edge_energy(frame)
        b = blur_score(frame)
        m = motion_inconsistency(prev_frame, frame)
        raw = cfg.scoring.w_edge * (1.0 - e) + cfg.scoring.w_motion * m + cfg.scoring.w_blur * b
        score = calibrate(raw)

        heat_path = None
        if cfg.output.save_heatmaps and (idx // cfg.sampler.every_nth) % cfg.output.save_every_n == 0:
            heat_path = save_edge_heatmap(frame, os.path.join(out_dir, "heatmaps"), idx)

        frames.append({
            "index": idx,
            "score": score,
            "explanations": textual_reasons(e, m, b),
            "heatmap_path": heat_path,
        })
        prev_frame = frame

    # Aggregate: take top-k suspicious frames and form simple segments (placeholder)
    scores = [f["score"] for f in frames]
    global_score = float(np.clip(float(np.mean(scores)), 0, 1)) if scores else 0.0

    results = {
        "video_path": path,
        "summary": {
            "global_score": global_score,
            "frames_evaluated": len(frames),
            "flagged_segments": [],
        },
        "frames": frames,
    }

    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results
