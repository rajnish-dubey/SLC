import os
import numpy as np
import cv2
from pathlib import Path

# Apply grayscale + blur + adaptive + otsu thresholding
def process_image(path: str, min_value: int = 70) -> 'np.ndarray':
    frame = cv2.imread(path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    _, res = cv2.threshold(
        th3, min_value, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return res

# Input: colored images downloaded from Kaggle
#data for training change it to your dataset
input_path = Path("resources")  
output_path = Path("output") #data for model
train_path = output_path / "train"
test_path = output_path / "test"

# Create required output directories
train_path.mkdir(parents=True, exist_ok=True)
test_path.mkdir(parents=True, exist_ok=True)

# Track stats
total, train_count, test_count = 0, 0, 0

# Process each folder (A-Z, 0-9)
for gesture_dir in input_path.iterdir():
    if gesture_dir.is_dir():
        print(f"Processing: {gesture_dir.name}")

        # Gather .jpg/.png/.jpeg images
        files = list(gesture_dir.glob("*.jpg")) + \
                list(gesture_dir.glob("*.png")) + \
                list(gesture_dir.glob("*.jpeg"))
        split_idx = int(0.8 * len(files))  # 80-20 split

        # Create gesture-specific train/test folders
        gesture_train_path = train_path / gesture_dir.name
        gesture_test_path = test_path / gesture_dir.name
        gesture_train_path.mkdir(parents=True, exist_ok=True)
        gesture_test_path.mkdir(parents=True, exist_ok=True)

        for i, file_path in enumerate(files):
            total += 1
            processed_img = process_image(str(file_path))

            if i < split_idx:
                save_path = gesture_train_path / file_path.name
                train_count += 1
            else:
                save_path = gesture_test_path / file_path.name
                test_count += 1

            cv2.imwrite(str(save_path), processed_img)

print(f"✅ Total images processed: {total}")
print(f"🟢 Training images: {train_count}")
print(f"🔵 Testing images: {test_count}")
