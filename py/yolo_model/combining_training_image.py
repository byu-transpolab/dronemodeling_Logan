import os
import shutil
import random

# ------------------------------
# CONFIG

# Takes a folder with all the folders of annotated images and labels and combines them into one folder with 80% in a training folder, and 20% in a validation folder. 

# ------------------------------

# Path to the "train" folder where building_seed folders live
train_root = "/Users/willicon/Desktop/Annotated Photos Provo Orem Area"

# Output location on Desktop
base_output = "/Users/willicon/Desktop"


# Parent folder for output
training_data_root = os.path.join(base_output, "training_data")
os.makedirs(training_data_root, exist_ok=True)

# Training + validation folders
training_root = os.path.join(training_data_root, "training")
validation_root = os.path.join(training_data_root, "validation")

# Create images/labels inside each
for root in [training_root, validation_root]:
    os.makedirs(os.path.join(root, "images"), exist_ok=True)
    os.makedirs(os.path.join(root, "labels"), exist_ok=True)

# ------------------------------
# FIND SEED FOLDERS
# ------------------------------

seed_folders = sorted([
    f for f in os.listdir(train_root)
    if os.path.isdir(os.path.join(train_root, f)) and not f.startswith('.')
])

if not seed_folders:
    raise RuntimeError("No building_seed folders found inside 'train'.")

print("Found seed folders:", seed_folders)

# ------------------------------
# COPY GLOBAL FILES (classes.txt, notes.json)
# ------------------------------

first_seed = seed_folders[0]
first_seed_path = os.path.join(train_root, first_seed)

for fname in ["classes.txt", "notes.json"]:
    src = os.path.join(first_seed_path, fname)
    if os.path.exists(src):
        shutil.copy(src, training_root)
        shutil.copy(src, validation_root)
        print(f"Copied {fname} → training/ and validation/")
    else:
        print(f"Warning: {fname} not found in {first_seed}")

# ------------------------------
# FILE MOVEMENT FUNCTION
# ------------------------------

def move_file_set(names, src_img_dir, src_lbl_dir, dest_root):
    """Copy matching image/label pairs into training OR validation."""
    for n in names:
        # Try common possible image extensions
        possible_exts = [".jpg", ".png", ".jpeg"]
        img_src = None
        for ext in possible_exts:
            candidate = os.path.join(src_img_dir, n + ext)
            if os.path.exists(candidate):
                img_src = candidate
                break

        if img_src is None:
            print(f"Missing image for {n}, skipping")
            continue

        lbl_src = os.path.join(src_lbl_dir, n + ".txt")
        if not os.path.exists(lbl_src):
            print(f"Missing label for {n}, skipping")
            continue

        # Dest paths
        img_dest = os.path.join(dest_root, "images", os.path.basename(img_src))
        lbl_dest = os.path.join(dest_root, "labels", os.path.basename(lbl_src))

        shutil.copy(img_src, img_dest)
        shutil.copy(lbl_src, lbl_dest)

# ------------------------------
# PROCESS EACH SEED FOLDER
# ------------------------------

for seed in seed_folders:
    print(f"\nProcessing {seed}...")

    seed_path = os.path.join(train_root, seed)
    img_dir = os.path.join(seed_path, "images")
    lbl_dir = os.path.join(seed_path, "labels")

    img_files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    lbl_files = sorted([f for f in os.listdir(lbl_dir)
                        if f.lower().endswith(".txt")])

    # Pair by basename
    base_img = {os.path.splitext(f)[0] for f in img_files}
    base_lbl = {os.path.splitext(f)[0] for f in lbl_files}
    base_names = sorted(list(base_img & base_lbl))

    random.shuffle(base_names)

    # 20/80 split
    split_idx = int(len(base_names) * 0.20)
    val_names = base_names[:split_idx]
    train_names = base_names[split_idx:]

    # Move files
    move_file_set(val_names, img_dir, lbl_dir, validation_root)
    move_file_set(train_names, img_dir, lbl_dir, training_root)

    print(f"→ {len(val_names)} to validation, {len(train_names)} to training")

# ------------------------------
# DONE
# ------------------------------

print("\nAll done! Check your Desktop → training_data/")


# Process each building_seed folder
for seed in seed_folders:
    print(f"\nProcessing {seed}...")
    seed_path = os.path.join(train_root, seed)

    img_dir = os.path.join(seed_path, "images")
    lbl_dir = os.path.join(seed_path, "labels")

    # Get image and label filenames
    img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    lbl_files = sorted([f for f in os.listdir(lbl_dir) if f.lower().endswith(".txt")])

    # Convert to matching pairs by basename
    base_names = set(os.path.splitext(f)[0] for f in img_files) & \
                 set(os.path.splitext(f)[0] for f in lbl_files)

    base_names = list(base_names)

    random.shuffle(base_names)

    # Split 20 / 80
    split_idx = int(len(base_names) * 0.20)
    val_names = base_names[:split_idx]
    train_names = base_names[split_idx:]

    def move_file_set(names, dest_root):
        for n in names:
            img_src = os.path.join(img_dir, n + ".jpg")
            if not os.path.exists(img_src):
                img_src = os.path.join(img_dir, n + ".png")
            if not os.path.exists(img_src):
                img_src = os.path.join(img_dir, n + ".jpeg")

            lbl_src = os.path.join(lbl_dir, n + ".txt")

            img_dest = os.path.join(dest_root, "images", os.path.basename(img_src))
            lbl_dest = os.path.join(dest_root, "labels", os.path.basename(lbl_src))

            shutil.copy(img_src, img_dest)
            shutil.copy(lbl_src, lbl_dest)

    # Move 20% → validation
    move_file_set(val_names, validation_root)

    # Move 80% → training
    move_file_set(train_names, training_root)

    print(f"{len(val_names)} → validation, {len(train_names)} → training")

print("\nDone! Your training/validation sets are ready.")