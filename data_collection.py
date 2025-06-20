import os
import cv2
import string
import numpy as np

# Create dataset directory for capturing raw gesture data (no train/test split)
os.makedirs("data", exist_ok=True)

# Create subfolders for digits (0-9) and uppercase letters (A-Z)
for label in [str(i) for i in range(10)] + list(string.ascii_uppercase):
    os.makedirs(f"data/{label}", exist_ok=True)

# Configuration
minValue = 70
save_binary = False #Toggle to save processed binary images or original ROI
frame_delay = 10

# Initialize camera
cap = cv2.VideoCapture(0)
last_key = None

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Flip for mirror view

    # Count number of images already captured per label
    count = {}
    for label in [str(i) for i in range(10)] + list(string.ascii_uppercase):
        count[label] = len(os.listdir(f"data/{label}"))

    # Display count of captured images
    y_offset = 70
    for label in [str(i) for i in range(10)] + list(string.ascii_lowercase):
        upper_label = label.upper() if label.isalpha() else label
        text = f"{upper_label} : {count[upper_label]}"
        cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
        y_offset += 10

    # Define region of interest (ROI)
    x1, y1, x2, y2 = 220, 10, 620, 410
    cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
    roi = frame[y1:y2, x1:x2]

    cv2.imshow("Frame", frame)
    
    # Preprocess ROI
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 1)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2 )
    _, test_image = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV +  cv2.THRESH_OTSU) #inversion
    test_image = cv2.resize(test_image, (300, 300))

    cv2.imshow("test", test_image)

    interrupt = cv2.waitKey(frame_delay)
    key_pressed = chr(interrupt & 0xFF).lower() if interrupt != -1 else None

    if interrupt & 0xFF == 27:  # ESC to exit
        break

    # Save only once per key press
    if key_pressed and key_pressed != last_key:
        if key_pressed in [str(i) for i in range(10)]:
            label = key_pressed
        elif key_pressed in string.ascii_lowercase:
            label = key_pressed.upper()
        else:
            label = None

        if label and label in count:
            filename = f"data/{label}/{count[label]}.jpg"
            image_to_save = test_image if save_binary else roi
            cv2.imwrite(filename, image_to_save)

    last_key = key_pressed

# Cleanup
cap.release()
cv2.destroyAllWindows()
