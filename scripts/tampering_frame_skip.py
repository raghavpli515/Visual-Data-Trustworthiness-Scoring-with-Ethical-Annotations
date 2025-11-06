import os
import random
import cv2
from tqdm import tqdm

def simulate_frame_skipping(input_folder, output_folder, skip_ratio=0.4):
    """
    Simulates temporal tampering by skipping frames randomly.
    :param input_folder: Folder with extracted frames.
    :param output_folder: Folder to save tampered frames.
    :param skip_ratio: Fraction of frames to skip (default 20%).
    """
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])
    num_to_skip = int(len(frames) * skip_ratio)
    skip_indices = set(random.sample(range(len(frames)), num_to_skip))

    for idx, frame_name in enumerate(tqdm(frames, desc=f"Frame skipping in {os.path.basename(input_folder)}")):
        if idx in skip_indices:
            continue
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is not None:
            cv2.imwrite(os.path.join(output_folder, frame_name), frame)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate frame skipping tampering.")
    parser.add_argument("--input_root", required=True, help="Root folder with original frame folders.")
    parser.add_argument("--output_root", required=True, help="Destination folder for tampered versions.")
    parser.add_argument("--skip_ratio", type=float, default=0.2, help="Fraction of frames to skip.")
    args = parser.parse_args()

    for folder in os.listdir(args.input_root):
        src = os.path.join(args.input_root, folder)
        dst = os.path.join(args.output_root, folder + "_tampered_skip")
        if os.path.isdir(src):
            simulate_frame_skipping(src, dst, args.skip_ratio)
    print("\n✅ Frame skipping tampering simulation completed.")


