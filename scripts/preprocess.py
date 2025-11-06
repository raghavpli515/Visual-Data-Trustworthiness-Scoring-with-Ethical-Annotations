import os, glob

# 1️⃣  Check absolute folder existence
path = r"D:\Computer Vision\vdt-ethical\dataset\faceforensics_frames"
print("Exists:", os.path.exists(path))

# 2️⃣  Check subdirectories and images
print("Subdirs:", os.listdir(path))
print("JPG count:", len(glob.glob(os.path.join(path, "*.jpg"))))

# 3️⃣  Dive one level deeper if you see folders printed above
print(glob.glob(os.path.join(path, "*", "*.jpg"))[:5])
