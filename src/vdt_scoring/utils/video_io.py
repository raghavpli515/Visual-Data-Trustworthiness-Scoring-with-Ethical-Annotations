import cv2
import numpy as np
from typing import Iterator, Tuple

def read_frames(path: str, every_nth: int = 5, max_frames: int = 500) -> Iterator[Tuple[int, np.ndarray]]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    idx = 0
    grabbed = True
    yielded = 0
    while grabbed and yielded < max_frames:
        grabbed, frame = cap.read()
        if not grabbed:
            break
        if idx % every_nth == 0:
            yield idx, frame
            yielded += 1
        idx += 1
    cap.release()
