import cv2
import os
import argparse
import csv
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=2):
    """
    Extract frames from a given video at specified FPS.
    Saves frames as JPG in the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    step = max(1, int(orig_fps / fps))

    count, saved = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % step == 0:
            frame_name = f"frame_{saved:05d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), frame)
            saved += 1
        count += 1
    cap.release()
    return saved

def process_dataset(input_root, output_root, fps=2, csv_path="outputs/logs/frame_index.csv"):
    """
    Iterate over all video files in dataset and extract frames.
    """
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["video_name", "frame_folder", "num_frames"])

        for root, _, files in os.walk(input_root):
            for file in tqdm(files, desc=f"Processing {os.path.basename(input_root)}"):
                if file.lower().endswith((".mp4", ".avi", ".mov")):
                    video_path = os.path.join(root, file)
                    video_name = os.path.splitext(file)[0]
                    output_dir = os.path.join(output_root, video_name)
                    frames_saved = extract_frames(video_path, output_dir, fps)
                    writer.writerow([video_name, output_dir, frames_saved])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from videos for trustworthiness dataset.")
    parser.add_argument("--input_root", type=str, required=True, help="Path to the input dataset folder (with videos).")
    parser.add_argument("--output_root", type=str, required=True, help="Path to save extracted frames.")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract (default: 2).")
    args = parser.parse_args()

    process_dataset(args.input_root, args.output_root, args.fps)
    print("\n✅ Frame extraction completed successfully.")
