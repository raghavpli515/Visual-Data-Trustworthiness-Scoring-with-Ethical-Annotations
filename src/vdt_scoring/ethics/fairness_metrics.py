import numpy as np
from typing import Dict, List

def subgroup_false_positive_rates(y_true: List[int], y_pred: List[int], groups: List[str]) -> Dict[str, float]:
    fprs = {}
    for g in set(groups):
        idxs = [i for i, gg in enumerate(groups) if gg == g]
        if not idxs:
            continue
        y_t = [y_true[i] for i in idxs]
        y_p = [y_pred[i] for i in idxs]
        # FPR = FP / (FP + TN)
        fp = sum(1 for t, p in zip(y_t, y_p) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_t, y_p) if t == 0 and p == 0)
        fprs[g] = fp / max(1, (fp + tn))
    return fprs
