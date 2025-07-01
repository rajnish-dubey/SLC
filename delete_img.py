import os
import random

# Path to your dataset directory
dataset_path = 'resources/'

# Percentage of images to delete
DELETE_PERCENT = 0.7

# Set seed for reproducibility (optional)
random.seed(42)

# Loop through all folders inside resources
for subfolder in sorted(os.listdir(dataset_path)):
    subfolder_path = os.path.join(dataset_path, subfolder)

    # Check if it's a folder and named with a letter or digit
    if os.path.isdir(subfolder_path) and subfolder.lower().isalnum():
        images = [f for f in os.listdir(subfolder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images = len(images)

        if total_images == 0:
            print(f"[SKIPPED] No images found in '{subfolder}'")
            continue

        # Calculate number to delete
        delete_count = int(DELETE_PERCENT * total_images)
        images_to_delete = random.sample(images, delete_count)

        # Delete images
        for img in images_to_delete:
            os.remove(os.path.join(subfolder_path, img))

        print(f"[DONE] Deleted {delete_count} out of {total_images} images in '{subfolder}'")

print("\n✅ All folders processed. 70% of images deleted from each a-z / 0-9 folder.")
