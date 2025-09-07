from typing import List

def textual_reasons(edge: float, motion: float, blur: float) -> List[str]:
    reasons = []
    if blur > 0.6:
        reasons.append("High blur/compression indicators")
    if motion > 0.5:
        reasons.append("Temporal inconsistency peaks across frames")
    if edge < 0.1:
        reasons.append("Very low edge detail; possible heavy compression or defocus")
    if not reasons:
        reasons.append("No strong manipulation indicators; low-risk frame")
    return reasons
