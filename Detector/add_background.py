import random
import requests
from pathlib import Path


YOLO_DATASET_ROOT = Path("/home/eduardo/rugose/Rugoso_Test-1/")


N_TRAIN_BG = 30
N_VAL_BG = 6


TRAIN_BG_PREFIX = "coco_bg_tr_"
VAL_BG_PREFIX = "coco_bg_val_"


MAX_COCO_ID = 581918  


IMAGES_TRAIN_DIR = YOLO_DATASET_ROOT / "images" / "train"
IMAGES_VAL_DIR = YOLO_DATASET_ROOT / "images" / "val"
LABELS_TRAIN_DIR = YOLO_DATASET_ROOT / "labels" / "train"
LABELS_VAL_DIR = YOLO_DATASET_ROOT / "labels" / "val"



def make_unique_name(target_dir: Path, filename: str) -> str:
    """
    If filename already exists in target_dir, append _1, _2, ... before extension.
    """
    base = Path(filename).stem
    ext = Path(filename).suffix
    candidate = filename
    k = 1
    while (target_dir / candidate).exists():
        candidate = f"{base}_{k}{ext}"
        k += 1
    return candidate


def download_coco_image(split: str, img_id: int, save_path: Path) -> bool:
    """
    Download one COCO image by global ID for a given split.
    split: 'train2017' or 'val2017'
    """
    url = f"http://images.cocodataset.org/{split}/{img_id:012d}.jpg"
    try:
        r = requests.get(url, stream=True, timeout=15)
    except Exception as e:
        print(f"  ✖ Error requesting {url}: {e}")
        return False

    if r.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(r.content)
        return True
    else:
        # print(f"  ✖ {url} -> HTTP {r.status_code}")
        return False


def download_backgrounds_for_split(
    split_name: str,
    n_images: int,
    img_dir: Path,
    lbl_dir: Path,
    prefix: str,
):
    """
    Download n_images random COCO images for a given split and
    register them as pure background in a YOLO dataset.
    """
    print(f"\n==============================")
    print(f"Adding COCO backgrounds for {split_name}")
    print(f"Target images dir: {img_dir}")
    print(f"Target labels dir: {lbl_dir}")
    print(f"Images to download: {n_images}")
    print("==============================\n")

    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    used_ids = set()
    downloaded = 0

    coco_split_folder = "train2017" if split_name == "train" else "val2017"

    while downloaded < n_images:
        img_id = random.randint(0, MAX_COCO_ID)
        if img_id in used_ids:
            continue
        used_ids.add(img_id)

        base_name = f"{img_id:012d}.jpg"
        filename = prefix + base_name
        filename = make_unique_name(img_dir, filename)

        save_path_img = img_dir / filename
        save_path_lbl = lbl_dir / (Path(filename).stem + ".txt")

        print(f"Downloading {coco_split_folder}/{img_id:012d}.jpg -> {filename} ...")

        if download_coco_image(coco_split_folder, img_id, save_path_img):
            # Create empty label (background)
            if not save_path_lbl.exists():
                save_path_lbl.touch()

            downloaded += 1
            print(f"  ✔ Saved ({downloaded}/{n_images})")
        else:
            print("  ✖ Download failed or image not found, retrying with another ID...")

    print(f"\n✅ DONE for {split_name}: {downloaded} background images added.")
    print(f"Images in: {img_dir}")
    print(f"Labels in: {lbl_dir}")



def main():
    if not YOLO_DATASET_ROOT.exists():
        raise FileNotFoundError(f"YOLO_DATASET_ROOT not found: {YOLO_DATASET_ROOT}")

    # Train backgrounds from COCO train2017
    if N_TRAIN_BG > 0:
        download_backgrounds_for_split(
            split_name="train",
            n_images=N_TRAIN_BG,
            img_dir=IMAGES_TRAIN_DIR,
            lbl_dir=LABELS_TRAIN_DIR,
            prefix=TRAIN_BG_PREFIX,
        )

    # Val backgrounds from COCO val2017
    if N_VAL_BG > 0:
        download_backgrounds_for_split(
            split_name="val",
            n_images=N_VAL_BG,
            img_dir=IMAGES_VAL_DIR,
            lbl_dir=LABELS_VAL_DIR,
            prefix=VAL_BG_PREFIX,
        )

    print("\n🎉 All requested COCO background images have been added.")


if __name__ == "__main__":
    main()
