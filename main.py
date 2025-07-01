import tkinter as tk
import tensorflow as tf
from PIL import Image, ImageTk
import cv2
import numpy as np
import time

class SignLanguageApp:
    def __init__(self):
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Sign Language Detector")
        self.root.geometry("1200x800")

        # Load the trained model
        print("Loading model...")
        self.model = None
        try:
            self.model = tf.keras.models.load_model('models/model-bw-best.keras')
            print("Model loaded successfully!")
            print("Model input shape:", self.model.input_shape)
            print("Model output shape:", self.model.output_shape)
            
            # Get the input size from model
            self.input_size = self.model.input_shape[1]  # Should be 64 or 128
            print(f"Using input size: {self.input_size}x{self.input_size}")
            
        except Exception as e:
            print("Error loading model:", str(e))
            self.root.destroy()
            raise Exception("Failed to load model. Please check if the model file exists and is valid.")

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            self.root.destroy()
            raise Exception("Failed to open camera")

        # Initialize variables
        self.current_word = ""
        self.current_symbol = ""
        self.last_symbol = ""
        self.symbol_count = 0
        self.blank_count = 0
        self.THRESHOLD = 20  # Frames to wait before adding a letter

        # Create UI elements
        self.create_ui()

        # Start video loop
        self.video_loop()

    def create_ui(self):
        # Create frames
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(pady=10)

        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(pady=10)

        # Create labels
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(side=tk.LEFT, padx=10)

        self.processed_label = tk.Label(self.video_frame)
        self.processed_label.pack(side=tk.LEFT, padx=10)

        # Create text displays
        self.debug_label = tk.Label(self.control_frame, text="Debug Info:", font=("Arial", 12))
        self.debug_label.pack()

        self.symbol_label = tk.Label(self.control_frame, text="Current Symbol:", font=("Arial", 20))
        self.symbol_label.pack()

        self.current_symbol_label = tk.Label(self.control_frame, text="", font=("Arial", 40))
        self.current_symbol_label.pack()

        self.word_label = tk.Label(self.control_frame, text="Current Word:", font=("Arial", 20))
        self.word_label.pack()

        self.current_word_label = tk.Label(self.control_frame, text="", font=("Arial", 40))
        self.current_word_label.pack()

        # Create buttons
        self.clear_button = tk.Button(self.control_frame, text="Clear Word", command=self.clear_word, font=("Arial", 16))
        self.clear_button.pack(pady=10)

        self.quit_button = tk.Button(self.control_frame, text="Quit", command=self.quit_app, font=("Arial", 16))
        self.quit_button.pack(pady=10)

    def preprocess_frame(self, frame):
        try:
            # Get ROI
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            roi = frame[y1:y1+roi_size, x1:x1+roi_size]
            
            # Convert to grayscale
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Resize to match model input size
            resized = cv2.resize(gray, (self.input_size, self.input_size))
            
            # Normalize to [0,1] - matching the training preprocessing
            normalized = resized.astype('float32') / 255.0
            
            # Add dimensions to match model input shape
            processed = np.expand_dims(normalized, axis=-1)
            processed = np.expand_dims(processed, axis=0)
            
            debug_info = f"Shape after processing: {processed.shape}, Range: [{processed.min():.2f}, {processed.max():.2f}]"
            self.debug_label.config(text=debug_info)
            
            return processed, resized  # Return resized instead of binary for display
        except Exception as e:
            print("Error in preprocessing:", str(e))
            return None, None

    def get_prediction(self, frame):
        # Preprocess the frame
        processed, enhanced = self.preprocess_frame(frame)
        if processed is None or not hasattr(self, 'model'):
            return "?", 0, enhanced
        
        try:
            # Get prediction
            prediction = self.model.predict(processed, verbose=0)
            predicted_class = np.argmax(prediction)
            confidence = float(np.max(prediction)) * 100

            # Print raw prediction values for debugging
            print(f"Raw prediction values: {prediction}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")

            # Map class to character
            char_map = {
                0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
                8: "I", 9: "J", 10: "K", 11: "L", 12: "M", 13: "N", 14: "O", 15: "P",
                16: "Q", 17: "R", 18: "S", 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
                24: "Y", 25: "Z", 26: "1", 27: "2", 28: "3", 29: "4", 30: "5", 31: "6",
                32: "7", 33: "8", 34: "9"
            }

            predicted_char = char_map.get(int(predicted_class), "?")
            return predicted_char, confidence, enhanced
        except Exception as e:
            print("Error in prediction:", str(e))
            return "?", 0, enhanced

    def update_word(self, symbol, confidence):
        if confidence < 85:  # If confidence is low, increment blank count
            self.blank_count += 1
            if self.blank_count > 30:  # Add space after 30 frames of low confidence
                if self.current_word and self.current_word[-1] != " ":
                    self.current_word += " "
                self.blank_count = 0
                self.symbol_count = 0
            return

        self.blank_count = 0
        
        if symbol == self.last_symbol:
            self.symbol_count += 1
            if self.symbol_count >= self.THRESHOLD:
                if not self.current_word or self.current_word[-1] != symbol:
                    self.current_word += symbol
                self.symbol_count = 0
        else:
            self.last_symbol = symbol
            self.symbol_count = 0

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw ROI on frame
            h, w = frame.shape[:2]
            roi_size = min(h, w) // 2
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            cv2.rectangle(frame, (x1, y1), (x1 + roi_size, y1 + roi_size), (0, 255, 0), 2)
            
            # Get prediction
            predicted_symbol, confidence, processed_frame = self.get_prediction(frame)
            
            # Update word
            self.update_word(predicted_symbol, confidence)
            
            # Convert frames to PhotoImage
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)
            
            if processed_frame is not None:
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)
                processed_frame = Image.fromarray(processed_frame)
                processed_frame = ImageTk.PhotoImage(processed_frame)
                
                # Update labels
                self.video_label.configure(image=frame)
                setattr(self.video_label, 'image', frame)
                self.processed_label.configure(image=processed_frame)
                setattr(self.processed_label, 'image', processed_frame)
            
            # Update text displays
            self.current_symbol_label.configure(text=f"{predicted_symbol} ({confidence:.1f}%)")
            self.current_word_label.configure(text=self.current_word)
        
        # Schedule next update
        self.root.after(10, self.video_loop)

    def clear_word(self):
        self.current_word = ""
        self.current_symbol = ""
        self.last_symbol = ""
        self.symbol_count = 0
        self.blank_count = 0

    def quit_app(self):
        print("Closing application...")
        self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.root.mainloop() 