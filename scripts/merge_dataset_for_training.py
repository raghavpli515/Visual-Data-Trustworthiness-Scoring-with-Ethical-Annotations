import os, shutil, random
from tqdm import tqdm
from pathlib import Path

AUTH_ROOT   = Path(r"D:/Computer Vision/vdt-ethical/dataset/faceforensics_frames")
TAMPER_ROOT = Path(r"D:/Computer Vision/vdt-ethical/dataset/faceforensics_tampered_variants")
OUT_ROOT    = Path(r"D:/Computer Vision/vdt-ethical/dataset/unified")
SPLIT       = (0.7, 0.2, 0.1)   # train/val/test
RANDOM_SEED = 42
CLEAN_UNIFIED = True            # set True to wipe unified/ before writing

random.seed(RANDOM_SEED)

def find_images_recursively(root: Path):
    """Return list[Path] of all .jpg/.jpeg/.png under root (any depth)."""
    exts = {".jpg", ".jpeg", ".png"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts]

def collect_frames(root: Path, label: int):
    """
    Collect (image_path, label, rel_key) tuples.
    rel_key is a unique, stable identifier derived from relative path to avoid name collisions.
    """
    items = []
    images = find_images_recursively(root)
    for img in images:
        # Build a unique key from the path *relative* to the class root
        rel = img.relative_to(root)  # e.g., videoA/frame_00001.jpg or variantX/frames/frame_00001.jpg
        # Create a filename-safe slug: replace separators with double underscores
        rel_slug = "__".join(rel.parts)
        items.append((img, label, rel_slug))
    return items

def ensure_clean_dirs(base: Path):
    for split_name in ["train", "val", "test"]:
        img_dir = base / split_name / "images"
        lbl_dir = base / split_name / "labels"
        if CLEAN_UNIFIED and (base / split_name).exists():
            shutil.rmtree(base / split_name, ignore_errors=True)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

def balanced_split(auth_items, tamp_items):
    """Balance authentic and tampered by downsampling the larger set; then split each class and merge."""
    n = min(len(auth_items), len(tamp_items))
    if n == 0:
        raise RuntimeError("No balanced data available: one of the classes is empty. "
                           "Check TAMPER_ROOT and that it contains images (likely under .../variant/frames/).")
    random.shuffle(auth_items)
    random.shuffle(tamp_items)
    auth_items = auth_items[:n]
    tamp_items = tamp_items[:n]

    combined = auth_items + tamp_items
    random.shuffle(combined)
    total = len(combined)
    n_train = int(SPLIT[0] * total)
    n_val   = int(SPLIT[1] * total)
    train = combined[:n_train]
    val   = combined[n_train:n_train+n_val]
    test  = combined[n_train+n_val:]
    return train, val, test

def copy_and_write(items, split_name, out_root: Path):
    img_dir = out_root / split_name / "images"
    lbl_dir = out_root / split_name / "labels"
    print(f"Copying {split_name} set ({len(items)} images)...")
    for src, label, rel_slug in tqdm(items):
        # Make unique image name using rel_slug
        dst_img = img_dir / f"{rel_slug}"
        # Ensure .jpg extension for consistency
        if dst_img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            dst_img = dst_img.with_suffix(".jpg")
        # Corresponding label file
        dst_lbl = lbl_dir / (dst_img.stem + ".txt")

        shutil.copy2(src, dst_img)
        with open(dst_lbl, "w") as f:
            f.write(str(label))

if __name__ == "__main__":
    print("Collecting authentic frames...")
    authentic = collect_frames(AUTH_ROOT, 0)
    print(f"  Found authentic: {len(authentic)}")

    print("Collecting tampered frames (recursively, includes .../variant/frames/)...")
    tampered = collect_frames(TAMPER_ROOT, 1)
    print(f"  Found tampered:  {len(tampered)}")

    ensure_clean_dirs(OUT_ROOT)
    train, val, test = balanced_split(authentic, tampered)

    copy_and_write(train, "train", OUT_ROOT)
    copy_and_write(val,   "val",   OUT_ROOT)
    copy_and_write(test,  "test",  OUT_ROOT)

    print("\n✅ Unified dataset ready at:", OUT_ROOT)
    print(f"Counts → train:{len(train)}  val:{len(val)}  test:{len(test)}")
