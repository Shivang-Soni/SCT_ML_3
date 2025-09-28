import os
import cv2
import shutil

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
TARGET_SIZE = (64, 64)

os.makedirs(PROCESSED_DIR, exist_ok=True)

for cls in ["Cats", "Dogs"]:
    src_folder = os.path.join(RAW_DIR, cls)
    dst_folder = os.path.join(PROCESSED_DIR, cls)
    os.makedirs(dst_folder, exist_ok=True)

    for filename in os.listdir(src_folder):
        src_path = os.path.join(src_folder, filename)
        dst_path = os.path.join(dst_folder, filename)
        try:
            img = cv2.imread(src_path)
            if img is None:
                continue
            img = cv2.resize(img, TARGET_SIZE)
            cv2.imwrite(dst_path, img)
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
