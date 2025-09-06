import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Paths
dataset_dir = "images for phrases"
output_dir = "datasets/ISL_Phrases"

# Train/val/test split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# Create output folders
for split in ["train", "val", "test"]:
    for cls in os.listdir(dataset_dir):
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

# Split data
for cls in os.listdir(dataset_dir):
    cls_path = os.path.join(dataset_dir, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    train_files, test_files = train_test_split(images, test_size=(1-train_ratio), random_state=42)
    val_files, test_files = train_test_split(test_files, test_size=(test_ratio/(val_ratio+test_ratio)), random_state=42)

    for f in train_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "train", cls, f))
    for f in val_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "val", cls, f))
    for f in test_files:
        shutil.copy(os.path.join(cls_path, f), os.path.join(output_dir, "test", cls, f))

print("✅ Dataset split into train/val/test folders at:", output_dir)
