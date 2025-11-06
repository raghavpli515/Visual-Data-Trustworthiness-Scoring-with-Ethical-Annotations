import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os, json
from tqdm import tqdm

# Load pretrained model
def load_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Identity()  # use embeddings only
    model.eval()
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def compute_score(embedding):
    """
    Converts an embedding into a trustworthiness score.
    (In real pipeline, this would be learned; here we use embedding smoothness.)
    """
    variance = torch.var(embedding)
    return max(0.0, min(1.0, 1.0 - variance.item() * 50))  # heuristic scaling

def auto_annotate(video_folder, output_json):
    model = load_model()
    annotations = []

    frames = sorted([f for f in os.listdir(video_folder) if f.endswith('.jpg')])
    print(f"\nAuto-annotating {len(frames)} frames in {os.path.basename(video_folder)}...")

    for frame_name in tqdm(frames):
        frame_path = os.path.join(video_folder, frame_name)
        image = Image.open(frame_path).convert("RGB")
        img_t = transform(image).unsqueeze(0)

        with torch.no_grad():
            embedding = model(img_t)
            score = compute_score(embedding)

        # Decide ethical flags from trustworthiness score
        flags = {
            "manipulated": score < 0.5,
            "cropped": False,
            "context_loss": score < 0.6,
            "bias_detected": False
        }

        annotations.append({
            "frame_id": frame_name,
            "trustworthiness_score": round(score, 3),
            "ethical_flags": flags,
            "notes": "Auto-generated using ResNet-18 heuristic"
        })

    result = {
        "video_id": os.path.basename(video_folder),
        "source": "auto_annotation_resnet18",
        "total_frames": len(annotations),
        "annotations": annotations
    }

    with open(output_json, "w") as f:
        json.dump(result, f, indent=4)

    print(f"\n✅ Auto-annotations saved → {output_json}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Automatic ethical annotation using pretrained CNN.")
    parser.add_argument("--video_folder", required=True, help="Folder path containing frames.")
    parser.add_argument("--output_json", required=True, help="Output JSON file path.")
    args = parser.parse_args()

    auto_annotate(args.video_folder, args.output_json)
