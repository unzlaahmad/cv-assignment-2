import os
import shutil
import random

DATASET_PATH = "data/raw"
OUTPUT_PATH = "data"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Create folders
for split in ["train", "val", "test"]:
    for category in ["fire", "smoke"]:
        os.makedirs(os.path.join(OUTPUT_PATH, split, category), exist_ok=True)

# Process data
for category in ["fire", "smoke"]:
    category_path = os.path.join(DATASET_PATH, category)

    if not os.path.exists(category_path):
        print(f"❌ Missing folder: {category_path}")
        continue

    images = os.listdir(category_path)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = int(total * (TRAIN_RATIO + VAL_RATIO))

    train_imgs = images[:train_end]
    val_imgs = images[train_end:val_end]
    test_imgs = images[val_end:]

    for img in train_imgs:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(OUTPUT_PATH, "train", category, img))

    for img in val_imgs:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(OUTPUT_PATH, "val", category, img))

    for img in test_imgs:
        shutil.copy(os.path.join(category_path, img),
                    os.path.join(OUTPUT_PATH, "test", category, img))

print("✅ Dataset split completed!")