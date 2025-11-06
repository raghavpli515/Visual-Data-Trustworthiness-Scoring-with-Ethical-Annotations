import os
import json
import cv2
import matplotlib.pyplot as plt

def annotate_video(video_folder, output_json):
    annotations = []
    frames = sorted([f for f in os.listdir(video_folder) if f.endswith(".jpg")])

    print(f"\nAnnotating video: {video_folder} ({len(frames)} frames)")
    print("Press 'a' for authentic, 't' for tampered, 'q' to quit (then close image window each time).")

    for frame_name in frames:
        frame_path = os.path.join(video_folder, frame_name)
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        plt.imshow(frame)
        plt.title(f"Frame: {frame_name}")
        plt.axis('off')
        plt.show(block=True)

        user_input = input("Label this frame → (a=authentic, t=tampered, q=quit): ").strip().lower()

        if user_input == 'q':
            break
        elif user_input == 'a':
            trust_score = 0.9
            flags = {"manipulated": False, "cropped": False, "context_loss": False, "bias_detected": False}
        elif user_input == 't':
            trust_score = 0.3
            flags = {"manipulated": True, "cropped": True, "context_loss": True, "bias_detected": False}
        else:
            print("Invalid input, skipping frame...")
            continue

        annotations.append({
            "frame_id": frame_name,
            "trustworthiness_score": trust_score,
            "ethical_flags": flags,
            "notes": "Manually annotated"
        })

    plt.close('all')

    result = {
        "video_id": os.path.basename(video_folder),
        "source": "manual_annotation",
        "total_frames": len(annotations),
        "annotations": annotations
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Saved annotations → {output_json}")
