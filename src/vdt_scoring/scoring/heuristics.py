import numpy as np
import cv2

def edge_energy(frame: np.ndarray) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    # Normalize by image size to [0,1]
    return float(np.clip(np.mean(mag) / 255.0, 0, 1))

def blur_score(frame: np.ndarray) -> float:
    # Lower variance of Laplacian indicates blur. Convert to a "risk" score.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    var_lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Heuristic mapping: small variance -> higher risk (more blur/compression)
    return float(np.clip(1.0 - (var_lap / 1000.0), 0, 1))

def motion_inconsistency(prev_frame: np.ndarray, frame: np.ndarray) -> float:
    # Simple temporal difference as inconsistency proxy
    if prev_frame is None:
        return 0.0
    prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    cur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev, cur)
    score = float(np.clip(np.mean(diff) / 255.0, 0, 1))
    return score
