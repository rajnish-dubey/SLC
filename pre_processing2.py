import os
import numpy as np
import cv2
from pathlib import Path

# Optimized grayscale + blur + adaptive threshold
# Otsu removed for speed, cv2.IMREAD_GRAYSCALE used for efficiency
def process_image(path: Path, min_value: int = 70) -> 'np.ndarray':
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    res = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    return res

# Define input and output directories
#data for training change it to your dataset
input_path = Path("resources") 
output_path = Path("output2") #processed daasetdirectory
train_path = output_path / "train"
test_path = output_path / "test"

# Create output directories if they don't exist
train_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

# Track total images processed
total, train_count, test_count = 0, 0, 0

# Process each gesture folder
for gesture_dir in input_path.iterdir():
    if gesture_dir.is_dir():
        print(f"Processing: {gesture_dir.name}")

        # Process only lowercase .jpg files
        files = sorted([f for f in gesture_dir.iterdir() if f.suffix.lower() == ".jpg"])
        split_idx = int(0.8 * len(files))

        # Create gesture subfolders in train and test directories
        gesture_train_path = train_path / gesture_dir.name
        gesture_test_path = test_path / gesture_dir.name
        gesture_train_path.mkdir(parents=True, exist_ok=True)
        gesture_test_path.mkdir(parents=True, exist_ok=True)

        # Preprocess and save to appropriate split
        for i, file_path in enumerate(files):
            processed_img = process_image(file_path)
            save_path = gesture_train_path / file_path.name if i < split_idx else gesture_test_path / file_path.name
            cv2.imwrite(str(save_path), processed_img)

            total += 1
            if i < split_idx:
                train_count += 1
            else:
                test_count += 1

print(f"\n✅ Total images processed: {total}")
print(f"🟢 Training images: {train_count}")
print(f"🔵 Testing images: {test_count}")