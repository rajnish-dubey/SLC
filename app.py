from flask import Flask, render_template, request, jsonify, Response
import numpy as np
import cv2
import base64
import io
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import operator
from string import ascii_uppercase
import threading
import queue
import time

app = Flask(__name__)

# Global variables for word formation
current_word = ""
last_prediction = ""
prediction_count = 0
PREDICTION_THRESHOLD = 10
space_gesture_counter = 0
SPACE_GESTURE_THRESHOLD = 20

# Character tracking
char_counts = {char: 0 for char in ascii_uppercase}
char_counts['blank'] = 0
blank_flag = 0

# Load the model
print('Loading model...')
model = load_model('models/model-bw-best.keras')
model.summary()
print('Model loaded successfully!')

# Add a global variable for last confidence
last_confidence = 0.0

def preprocess_image(image, target_size=(128, 128)):
    """Preprocess the image for model prediction"""
    # Convert to grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize image
    image = cv2.resize(image, target_size)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    image = clahe.apply(image)
    
    # Apply Gaussian blur
    image = cv2.GaussianBlur(image, (3,3), 0)
    
    # Apply adaptive thresholding
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Normalize pixel values
    image = image / 255.0
    
    # Add channel dimension
    image = np.expand_dims(image, axis=-1)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

def get_last_confidence():
    """Get the last prediction confidence."""
    global last_confidence
    return round(last_confidence, 2)

def process_prediction(prediction, confidence):
    """Process the prediction using character counting and blank detection"""
    global current_word, char_counts, blank_flag, space_gesture_counter, last_prediction, last_confidence
    
    # Update last prediction and confidence
    last_prediction = prediction
    last_confidence = confidence
    
    # Update character count
    if confidence > 85:
        space_gesture_counter = 0
        char_counts[prediction] += 1
        
        # Check if the character has been consistently predicted
        if char_counts[prediction] > 30:  # Reduced from 60 to make it more responsive
            # Check if other characters are being confused
            for char in ascii_uppercase:
                if char == prediction:
                    continue
                diff = char_counts[prediction] - char_counts[char]
                if abs(diff) <= 10:  # Reduced from 20 to make it more responsive
                    # Reset counts if there's confusion
                    char_counts = {char: 0 for char in ascii_uppercase}
                    char_counts['blank'] = 0
                    return current_word, prediction, False
            
            # Reset counts
            char_counts = {char: 0 for char in ascii_uppercase}
            char_counts['blank'] = 0
            
            # Handle blank prediction
            if prediction == 'blank':
                if blank_flag == 0:
                    blank_flag = 1
                    return current_word + " ", 'space', True
            else:
                blank_flag = 0
                return current_word + prediction, prediction, True
    else:
        space_gesture_counter += 1
        if space_gesture_counter >= SPACE_GESTURE_THRESHOLD:
            space_gesture_counter = 0
            return current_word + " ", 'space', True
            
    return current_word, prediction, False

def generate_frames():
    """Generate frames from webcam with predictions."""
    global current_word
    
    camera = cv2.VideoCapture(0)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Process frame
        processed_frame = preprocess_image(frame)
        
        # Make prediction
        prediction = model.predict(processed_frame, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction)) * 100
        
        # Get the label for the predicted class
        current_prediction = CLASS_LABELS.get(predicted_class + 1, "Unknown")
        
        # Process prediction and update word
        current_word, gesture, word_updated = process_prediction(current_prediction, confidence)
        
        # Draw prediction and word on frame
        cv2.putText(frame, f"Prediction: {current_prediction}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {confidence:.2f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Word: {current_word}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Convert frame to jpg
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    camera.release()

# Comprehensive Indian Sign Language Labels
CLASS_LABELS = {
    # Numbers (1-9)
    1: "1", 2: "2", 3: "3", 4: "4", 5: "5",
    6: "6", 7: "7", 8: "8", 9: "9",
    
    # Alphabet (A-Z)
    10: "a", 11: "b", 12: "c", 13: "d", 14: "e",
    15: "f", 16: "g", 17: "h", 18: "i", 19: "j",
    20: "k", 21: "l", 22: "m", 23: "n", 24: "o",
    25: "p", 26: "q", 27: "r", 28: "s", 29: "t",
    30: "u", 31: "v", 32: "w", 33: "x", 34: "y",
    35: "z"
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_word')
def get_word():
    """Return the current word and prediction details."""
    global current_word, last_prediction
        return jsonify({
        'word': current_word,
        'prediction': last_prediction,
        'confidence': get_last_confidence()
    })

@app.route('/clear_word')
def clear_word():
    """Clear the current word and reset all counters."""
    global current_word, char_counts, blank_flag, space_gesture_counter, last_prediction, last_confidence
    current_word = ""
    char_counts = {char: 0 for char in ascii_uppercase}
    char_counts['blank'] = 0
    blank_flag = 0
    space_gesture_counter = 0
    last_prediction = "Waiting for gesture..."
    last_confidence = 0.0
        return jsonify({
        'success': True,
        'word': current_word,
        'prediction': last_prediction,
        'confidence': last_confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)