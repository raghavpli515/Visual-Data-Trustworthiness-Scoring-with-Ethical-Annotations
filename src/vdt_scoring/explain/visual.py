import numpy as np
import cv2
import os

def save_edge_heatmap(frame: np.ndarray, out_dir: str, idx: int) -> str:
    os.makedirs(out_dir, exist_ok=True)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(mag_norm, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
    path = os.path.join(out_dir, f"frame_{idx:06d}.png")
    cv2.imwrite(path, overlay)
    return path
