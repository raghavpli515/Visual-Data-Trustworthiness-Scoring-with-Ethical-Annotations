import os
import cv2
import random
from tqdm import tqdm

def simulate_cropping(input_folder, output_folder, crop_ratio=0.3):
    """
    Simulates spatial tampering by randomly cropping frames.
    :param input_folder: Folder with extracted frames.
    :param output_folder: Folder to save cropped frames.
    :param crop_ratio: How much area to crop from the frame (0.0–0.5 recommended).
    """
    os.makedirs(output_folder, exist_ok=True)
    frames = sorted([f for f in os.listdir(input_folder) if f.endswith('.jpg')])

    for frame_name in tqdm(frames, desc=f"Cropping in {os.path.basename(input_folder)}"):
        frame_path = os.path.join(input_folder, frame_name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        h, w = frame.shape[:2]
        ch, cw = int(h * (1 - crop_ratio)), int(w * (1 - crop_ratio))

        x_start = random.randint(0, w - cw)
        y_start = random.randint(0, h - ch)
        cropped = frame[y_start:y_start + ch, x_start:x_start + cw]
        cropped = cv2.resize(cropped, (w, h))  # Resize back to original dimensions

        cv2.imwrite(os.path.join(output_folder, frame_name), cropped)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate cropping tampering.")
    parser.add_argument("--input_root", required=True, help="Root folder with original frame folders.")
    parser.add_argument("--output_root", required=True, help="Destination folder for cropped versions.")
    parser.add_argument("--crop_ratio", type=float, default=0.3, help="Portion of frame to crop.")
    args = parser.parse_args()

    for folder in os.listdir(args.input_root):
        src = os.path.join(args.input_root, folder)
        dst = os.path.join(args.output_root, folder + "_tampered_crop")
        if os.path.isdir(src):
            simulate_cropping(src, dst, args.crop_ratio)
    print("\n✅ Cropping tampering simulation completed.")
