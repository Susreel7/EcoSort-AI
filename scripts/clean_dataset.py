import os
from PIL import Image, UnidentifiedImageError
import shutil

TRAIN_DIR = "../waste_data_split/train"
VAL_DIR = "../waste_data_split/val"

# Minimum size threshold (You can reduce if needed)
MIN_SIZE = 100

def clean_directory(directory):
    print(f"\n🧹 Cleaning folder: {directory}")
    deleted_count = 0
    fixed_count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Skip non-image files
            if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')):
                os.remove(file_path)
                deleted_count += 1
                print(f"❌ Removed non-image file: {file_path}")
                continue

            try:
                with Image.open(file_path) as img:
                    # Check corrupted
                    img.verify()

                # Reopen for actual processing
                with Image.open(file_path) as img:
                    # Convert to RGB
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                        img.save(file_path)
                        fixed_count += 1

                    # Check size
                    if img.width < MIN_SIZE or img.height < MIN_SIZE:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"🗑️ Removed small image: {file_path}")
                        continue

            except UnidentifiedImageError:
                os.remove(file_path)
                deleted_count += 1
                print(f"⚠️ Removed corrupted image: {file_path}")
            except Exception as e:
                print(f"❗ Error processing {file_path}: {e}")

    print(f"\n✅ Done cleaning: {directory}")
    print(f"   ➤ Fixed color format: {fixed_count}")
    print(f"   ➤ Removed corrupted/small/non-image: {deleted_count}\n")


# Run cleaning for training and validation sets
clean_directory(TRAIN_DIR)
clean_directory(VAL_DIR)

print("\n🎉 Dataset cleaning completed! You can now safely run training.\n")
