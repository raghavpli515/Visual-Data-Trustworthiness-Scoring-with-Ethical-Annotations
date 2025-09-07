import cv2
import numpy as np
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=120)
    args = ap.parse_args()

    w, h = 320, 240
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.out, fourcc, 24.0, (w, h))

    pos = 10
    for i in range(args.frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # moving square
        cv2.rectangle(frame, (pos, 100), (pos+40, 140), (0, 255, 0), -1)
        pos = (pos + 3) % (w - 40)
        # inject a blur/compression-like region occasionally
        if 40 < i < 60:
            frame = cv2.GaussianBlur(frame, (15, 15), 10)
        # inject a sudden jump (temporal inconsistency)
        if i == 80:
            pos = 200
        out.write(frame)
    out.release()
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
