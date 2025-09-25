import os
import numpy as np
from PIL import Image

def load_images_from_folder(folder_path, target_size=(64, 64)):
    """
    Loads images from folder, resizes, and flattens them.
    Cats labeled 0, Dogs labeled 1.
    """
    X = []
    Y = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        label = 0 if "cat" in filename.lower() else 1
        img_path = os.path.join(folder_path, filename)

        try: 
            img = Image.open(img_path).convert("RGB")
            img = img.resize(target_size)
            img_array = np.array(img).flatten()

            X.append(img_array)
            Y.append(label)  

        except Exception as e:
            print(f"Error loading {img_path}: {e}") 

    return np.array(X), np.array(Y)
