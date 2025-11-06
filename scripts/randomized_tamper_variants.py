#!/usr/bin/env python3
"""
Randomized tampering script:
- Inputs: frame folders under FRAMES_DIR (one folder per video)
- Outputs: tampered frame folders per variant + metadata JSON per variant
Features:
- Randomized crop ratio (0.1 - 0.65)
- Randomized skip interval (2 - 15)
- Compression level (CRF-like 20 - 40 mapped to JPEG quality)
- Combos: crop+compress, skip+compress
- Realism vs variety: 70% subtle variants, 30% severe variants
"""

import os
import cv2
import json
import random
from tqdm import tqdm
from datetime import datetime

# ---------------- CONFIG ----------------
FRAMES_DIR = r"D:/Computer Vision/vdt-ethical/dataset/faceforensics_frames"   # input root (contains per-video folders)
OUTPUT_ROOT = r"D:/Computer Vision/vdt-ethical/dataset/faceforensics_tampered_variants"  # output root
N_VARIANTS_PER_VIDEO = 4   # 3-6 recommended
SUBTLE_PROB = 0.7          # 70% subtle, 30% severe
RANDOM_SEED = 42
# ----------------------------------------

random.seed(RANDOM_SEED)

# --- helpers for parameter sampling ---
def sample_crop_ratio(severity):
    if severity == "subtle":
        return round(random.uniform(0.10, 0.25), 3)
    else:  # severe
        return round(random.uniform(0.30, 0.65), 3)

def sample_skip_interval(severity):
    if severity == "subtle":
        # subtle skipping -> larger interval = less frequent skips
        return random.randint(8, 15)
    else:
        return random.randint(2, 7)

def sample_crf():
    return random.randint(20, 40)

def crf_to_jpeg_quality(crf):
    """
    Map CRF (20-40) to JPEG quality (95 - 30).
    Lower CRF => higher quality; higher CRF => lower quality.
    """
    # linear mapping: CRF 20 -> 95, CRF 40 -> 30
    q = int(round(95 - ( (crf - 20) / 20 ) * (95 - 30) ))
    q = max(10, min(95, q))
    return q

# --- image operations ---
def apply_crop_and_resize(img, crop_ratio):
    h, w = img.shape[:2]
    # crop area size = (1 - crop_ratio) of original
    ch = int(h * (1 - crop_ratio))
    cw = int(w * (1 - crop_ratio))
    if ch <= 0 or cw <= 0:
        return img.copy()
    # random top-left within valid range
    x0 = random.randint(0, w - cw)
    y0 = random.randint(0, h - ch)
    cropped = img[y0:y0 + ch, x0:x0 + cw]
    # resize back to original to keep consistent shape
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return resized, [x0, y0, cw, ch]

def compress_and_save_as_jpeg(img, out_path, jpeg_quality):
    # imencode then write to disk to emulate compression artifact
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    success, encimg = cv2.imencode('.jpg', img, encode_param)
    if not success:
        raise IOError("JPEG encoding failed for " + out_path)
    with open(out_path, 'wb') as f:
        f.write(encimg.tobytes())

# --- main tampering per-folder ---
def create_variant_from_frame_folder(src_folder, dst_variant_folder, tamper_spec):
    """
    src_folder: path to folder containing frames (images)
    dst_variant_folder: path to output folder for this variant (will contain 'frames/' and metadata)
    tamper_spec: dict specifying operations and parameters
    """
    os.makedirs(dst_variant_folder, exist_ok=True)
    frames_out_dir = os.path.join(dst_variant_folder, "frames")
    os.makedirs(frames_out_dir, exist_ok=True)

    frames = sorted([f for f in os.listdir(src_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))])
    if len(frames) == 0:
        return {"status":"no_frames"}

    skipped_indices = set()
    crop_box = None
    jpeg_quality = None

    # If skip is requested, compute indices to skip (every nth)
    if tamper_spec.get("skip_interval"):
        si = tamper_spec["skip_interval"]
        # skip rule: skip frames where index % si == 0 (keeps pattern reproducible)
        # but to add some randomness, randomly remove up to 20% additional frames for severe
        for i in range(len(frames)):
            if (i % si) == 0:
                skipped_indices.add(i)

    # If compression requested, compute jpeg quality
    if tamper_spec.get("crf") is not None:
        jpeg_quality = crf_to_jpeg_quality(tamper_spec["crf"])

    # Process each frame
    saved = 0
    failed = []
    for i, frame_name in enumerate(tqdm(frames, desc=f"Tampering {os.path.basename(src_folder)} -> {os.path.basename(dst_variant_folder)}")):
        if i in skipped_indices:
            continue

        src_path = os.path.join(src_folder, frame_name)
        img = cv2.imread(src_path)
        if img is None:
            failed.append(frame_name)
            continue

        # start with original
        out_img = img.copy()
        applied_crop = False

        # apply crop if requested
        if tamper_spec.get("crop_ratio") is not None:
            out_img, crop_box = apply_crop_and_resize(out_img, tamper_spec["crop_ratio"])
            applied_crop = True

        # save with compression if requested
        out_path = os.path.join(frames_out_dir, frame_name)
        if jpeg_quality is not None:
            try:
                compress_and_save_as_jpeg(out_img, out_path, jpeg_quality)
            except Exception as e:
                failed.append(frame_name)
                continue
        else:
            # save lossless-ish PNG to preserve quality
            cv2.imwrite(out_path, out_img)

        saved += 1

    # write metadata.json for this variant
    metadata = {
        "variant_of": os.path.basename(src_folder),
        "variant_id": os.path.basename(dst_variant_folder),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tamper_spec": tamper_spec,
        "num_input_frames": len(frames),
        "num_output_frames": saved,
        "skipped_frames_count": len(skipped_indices),
        "crop_box": crop_box,
        "jpeg_quality": jpeg_quality,
        "failed_frames": failed
    }

    with open(os.path.join(dst_variant_folder, "metadata.json"), 'w') as jf:
        json.dump(metadata, jf, indent=4)

    return metadata

# --- orchestration ---
def generate_variants_for_all_videos(frames_root, out_root, n_variants=4):
    os.makedirs(out_root, exist_ok=True)
    video_folders = sorted([d for d in os.listdir(frames_root) if os.path.isdir(os.path.join(frames_root, d))])

    summary = []
    for video in video_folders:
        src_folder = os.path.join(frames_root, video)
        # sample number of variants (or fixed n)
        for vidx in range(1, n_variants + 1):
            # decide severity
            severity = "subtle" if random.random() < SUBTLE_PROB else "severe"

            # choose tamper mode: single or combo
            # options: "crop", "skip", "compress", "crop+compress", "skip+compress"
            modes = ["crop", "skip", "compress", "crop+compress", "skip+compress"]
            # bias selection slightly toward combos and crop/skip
            mode = random.choices(modes, weights=[2,2,1,2,2], k=1)[0]

            tamper_spec = {"mode": mode, "severity": severity}
            if "crop" in mode:
                tamper_spec["crop_ratio"] = sample_crop_ratio(severity)
            if "skip" in mode:
                tamper_spec["skip_interval"] = sample_skip_interval(severity)
            if "compress" in mode:
                crf = sample_crf()
                tamper_spec["crf"] = crf

            variant_name = f"{video}__variant_{vidx}_{mode}_s-{severity}"
            dst_variant_folder = os.path.join(out_root, variant_name)

            meta = create_variant_from_frame_folder(src_folder, dst_variant_folder, tamper_spec)
            summary.append({
                "video": video,
                "variant": variant_name,
                "metadata": meta
            })

    # write global summary
    with open(os.path.join(out_root, "variants_summary.json"), 'w') as sf:
        json.dump(summary, sf, indent=4)

    print("\n✅ All variants generated. Summary saved to", os.path.join(out_root, "variants_summary.json"))
    return summary

# ----------------- CLI entrypoint -----------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate randomized tampered frame-folder variants.")
    parser.add_argument("--frames_root", default=FRAMES_DIR, help="Root folder containing per-video frame folders")
    parser.add_argument("--out_root", default=OUTPUT_ROOT, help="Root folder to save tampered variants")
    parser.add_argument("--n_variants", type=int, default=N_VARIANTS_PER_VIDEO, help="Variants per original video")
    args = parser.parse_args()

    print("Frames root:", args.frames_root)
    print("Output root:", args.out_root)
    print("Variants per video:", args.n_variants)
    generate_variants_for_all_videos(args.frames_root, args.out_root, args.n_variants)
