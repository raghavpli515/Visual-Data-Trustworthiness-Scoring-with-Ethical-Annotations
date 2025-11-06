

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
from tqdm import tqdm
import numpy as np
import clip  # pip install git+https://github.com/openai/CLIP.git

# --- CONFIG ---
FRAMES_DIR = r"D:/Computer Vision/vdt-ethical/dataset/faceforensics_tampered_skip/01__hugging_happy_tampered_skip"
OUTPUT_JSON = r"D:/Computer Vision/vdt-ethical/dataset/faceforensics_tampered_skip/01__hugging_happy_tampered_skip/trustworthiness_metadata_v2.json"

# --- MODEL SETUP ---

# 1. ResNet for trustworthiness
resnet = models.resnet18(pretrained=True)
resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False

classifier = nn.Sequential(
    nn.Linear(1000, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)
classifier.eval()

# 2. CLIP for semantic-naturalness scoring
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess_clip = clip.load("ViT-B/32", device=device)
clip_model.eval()

# --- TRANSFORMS ---
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- FFT ANALYSIS FUNCTION ---
def compute_frequency_score(image: Image.Image):
    """Estimate how natural the frequency distribution is."""
    gray = np.array(image.convert("L"))
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    high_freq_ratio = np.mean(magnitude_spectrum > np.median(magnitude_spectrum))
    return 1 - high_freq_ratio  # lower high-freq = more natural

# --- INFERENCE LOOP ---
metadata = []

with torch.no_grad():
    for frame_file in tqdm(sorted(os.listdir(FRAMES_DIR))):
        if not frame_file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        frame_path = os.path.join(FRAMES_DIR, frame_file)
        image = Image.open(frame_path).convert('RGB')

        # ResNet-based trustworthiness
        resnet_tensor = resnet_transform(image).unsqueeze(0)
        resnet_features = resnet(resnet_tensor)
        trust_score = classifier(resnet_features).item()

        # CLIP-based semantic realism
        clip_image = preprocess_clip(image).unsqueeze(0).to(device)
        text_tokens = clip.tokenize(["a real human face", "a synthetic or fake face"]).to(device)
        logits_per_image, _ = clip_model(clip_image, text_tokens)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        semantic_score = float(probs[0])  # probability of “real human face”

        # Frequency-based artifact score
        freq_score = compute_frequency_score(image)

        # Combined score
        combined_score = round(float((trust_score + semantic_score + freq_score) / 3), 3)

        # Ethical flags
        ethical_flags = {
            "manipulated": bool(combined_score < 0.5),
            "deepfake_artifact": bool(freq_score < 0.4),
            "semantic_inconsistency": bool(semantic_score < 0.5),
            "context_loss": bool(trust_score < 0.3),
        }

        metadata.append({
            "frame_id": frame_file,
            "trustworthiness_score": round(trust_score, 3),
            "semantic_score": round(semantic_score, 3),
            "frequency_score": round(freq_score, 3),
            "combined_score": combined_score,
            "ethical_flags": {k: str(v) for k, v in ethical_flags.items()},
            "notes": "ResNet + CLIP + FFT multi-modal trustworthiness assessment"
        })

# --- SAVE JSON ---
with open(OUTPUT_JSON, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"✅ Enhanced trustworthiness metadata saved to {OUTPUT_JSON}")
