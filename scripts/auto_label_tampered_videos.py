from ultralytics import YOLO
import cv2
import os
from tqdm import tqdm

# === CONFIGURATION ===
video_folder = "tampered_videos"          # Folder with your tampered videos
output_folder = "auto_labels"             # Output directory for frames + YOLO labels
model_path = "yolov8x.pt"                 # Pretrained model (you can switch to 'yolov8m.pt' for speed)

# Create output directories
os.makedirs(output_folder, exist_ok=True)

# Load the YOLO model
model = YOLO(model_path)

# Process each video
for video_name in os.listdir(video_folder):
    if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(video_folder, video_name)
    video_stem = os.path.splitext(video_name)[0]
    save_dir = os.path.join(output_folder, video_stem)
    img_dir = os.path.join(save_dir, "images")
    label_dir = os.path.join(save_dir, "labels")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nProcessing {video_name} ({total_frames} frames)...")

    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame_name = f"frame_{frame_idx:05d}.jpg"
        frame_path = os.path.join(img_dir, frame_name)
        cv2.imwrite(frame_path, frame)

        # Run detection
        results = model.predict(frame, verbose=False)

        # Save results in YOLO format
        label_path = os.path.join(label_dir, frame_name.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            for box in results[0].boxes:
                cls = int(box.cls[0])
                x_center, y_center, w, h = box.xywhn[0]
                f.write(f"{cls} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        frame_idx += 1

    cap.release()

print("\n✅ Auto-labeling complete! YOLOv8-format images and labels saved in:", output_folder)
