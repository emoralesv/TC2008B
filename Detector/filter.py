import os
import shutil
from pathlib import Path


DATASET_DIR = Path("Rugoso_Test-1").resolve()
print("Full dataset path:", DATASET_DIR)

# Output root
OUTPUT_DIR = DATASET_DIR.parent / "Rugoso_filtered"
OUTPUT_DIR.mkdir(exist_ok=True)

# Splits to process
SPLITS = ["train", "valid"]

# YOLO class names (index = class id)
names = [
    'Plantas-de-jitomate',  # 0
    'T_F_H',                # 1
    'T_F_R',                # 2  <-- positive
    'T_Fl_H',               # 3
    'T_Fl_R',               # 4  <-- positive
    'T_L_H',                # 5
    'T_L_R',                # 6  <-- positive
    'T_S_H',                # 7
    'T_S_R'                 # 8  <-- positive
]


positive_ids = {2, 4, 6, 8}


reindex = {2: 0, 4: 1, 6: 2, 8: 3}

print("Keeping class IDs:", positive_ids)
print("Keeping class names:", [names[i] for i in positive_ids])
print("Reindex mapping:", reindex)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")



def filter_yolo_labels(input_txt: Path, output_txt: Path) -> bool:
    """
    Read a YOLO label file, keep only positive classes, reindex their IDs,
    and write filtered labels to output_txt.

    Returns
    -------
    had_positive : bool
        True if at least one positive object was kept and written.
    """
    if not input_txt.exists():
        return False

    filtered_lines = []

    with open(input_txt, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            # First value is the class id
            try:
                class_id = int(parts[0])
            except ValueError:
                # Skip malformed lines
                continue

            # Keep only positive classes
            if class_id in positive_ids:
                # Reindex class id
                new_class_id = reindex[class_id]
                parts[0] = str(new_class_id)
                new_line = " ".join(parts) + "\n"
                filtered_lines.append(new_line)

    # Ensure directory exists
    output_txt.parent.mkdir(parents=True, exist_ok=True)

    with open(output_txt, "w") as f:
        f.writelines(filtered_lines)

    return len(filtered_lines) > 0


def find_image_for_label(txt_path: Path, labels_dir: Path, images_dir: Path) -> Path | None:
    """
    Given a label file path inside labels_dir, try to find an image with the same
    stem and one of the allowed IMAGE_EXTENSIONS inside images_dir, preserving
    the same relative path.
    """
    relative = txt_path.relative_to(labels_dir)  # e.g. img_001.txt or sub/dir/img_001.txt

    for ext in IMAGE_EXTENSIONS:
        candidate = images_dir / relative.with_suffix(ext)
        if candidate.exists():
            return candidate

    return None



total_txt = 0
total_with_positive = 0
total_images_copied = 0

for split in SPLITS:
    print(f"\n=== Processing split: {split} ===")

    labels_dir = DATASET_DIR / split / "labels"
    images_dir = DATASET_DIR / split / "images"

    out_labels_dir = OUTPUT_DIR / split / "labels"
    out_images_dir = OUTPUT_DIR / split / "images"

    if not labels_dir.exists():
        print(f"[WARNING] Labels dir does not exist: {labels_dir}, skipping split.")
        continue

    num_txt = 0
    num_with_positive = 0
    num_images_copied = 0

    for txt_path in labels_dir.rglob("*.txt"):
        num_txt += 1
        total_txt += 1

        # Parallel output path for labels
        relative = txt_path.relative_to(labels_dir)
        output_txt = out_labels_dir / relative

        print("Processing label:", txt_path)

        # Filter + reindex
        had_positive = filter_yolo_labels(txt_path, output_txt)

        if not had_positive:
            print("  -> No positive labels found. Skipping image copy.")
            # Optionally remove empty label file:
            # output_txt.unlink(missing_ok=True)
            continue

        num_with_positive += 1
        total_with_positive += 1

        # Find corresponding image for this label
        img_path = find_image_for_label(txt_path, labels_dir, images_dir)
        if img_path is None:
            print("  [WARNING] No corresponding image found for:", txt_path)
            continue

        img_relative = img_path.relative_to(images_dir)
        output_img = out_images_dir / img_relative
        output_img.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(img_path, output_img)
        num_images_copied += 1
        total_images_copied += 1
        print("  -> Copied image:", img_path, "->", output_img)

    print(f"\nSplit '{split}' summary:")
    print("  Label files processed:", num_txt)
    print("  Label files with positives:", num_with_positive)
    print("  Images copied:", num_images_copied)

print("\n=== GLOBAL SUMMARY ===")
print("Total label files processed:", total_txt)
print("Total label files with positives:", total_with_positive)
print("Total images copied:", total_images_copied)
print("Filtered dataset is in:", OUTPUT_DIR)
