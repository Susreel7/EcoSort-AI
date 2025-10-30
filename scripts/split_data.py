import os
import shutil
import random

DATASET_DIR = "../dataset"          # contains 'cardboard', 'e-waste', etc.
OUTPUT_DIR = "../waste_data_split" # output folder
TRAIN_RATIO = 0.8                # 80% training, 20% validation

# Create train/val directories
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Loop through each waste category
for category in os.listdir(DATASET_DIR):
    category_path = os.path.join(DATASET_DIR, category)
    if not os.path.isdir(category_path):
        continue

    # Create subfolders for each category in train and val
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)

    # Get all image files in the category
    images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    random.shuffle(images)

    # Split dataset
    split_index = int(len(images) * TRAIN_RATIO)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Copy images to their respective folders
    for img in train_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(train_dir, category, img))
    for img in val_images:
        shutil.copy(os.path.join(category_path, img), os.path.join(val_dir, category, img))

    print(f"✅ {category}: {len(train_images)} train, {len(val_images)} val")

print("\n🎉 Dataset successfully split into training and validation sets!")
