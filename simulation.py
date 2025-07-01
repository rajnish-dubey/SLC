import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import time
from collections import deque, Counter
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class ISLConverter:
    def __init__(self, model_path='models/classify_model.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = [
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
        # Video capture settings
        self.cap = None
        self.is_recording = False
        self.frame_queue = queue.Queue(maxsize=2)
        
        # Prediction settings
        self.prediction_history = deque(maxlen=10)  # Store last 10 predictions
        self.confidence_threshold = 0.7
        self.stability_threshold = 5  # Need 5 consistent predictions
        self.last_prediction_time = 0
        self.prediction_interval = 1.0  # Predict every 1 second
        
        # Word building
        self.current_word = ""
        self.word_history = []
        self.last_added_letter = ""
        self.last_letter_time = 0
        self.letter_hold_time = 2.0  # Hold letter for 2 seconds before adding
        
        # GUI components
        self.root = None
        self.video_label = None
        self.prediction_label = None
        self.word_label = None
        self.confidence_bar = None
        self.history_text = None
        
    def load_model(self):
        """Load the trained model"""
        try:
            print("Loading model...")
            self.model = load_model(self.model_path)
            print("✓ Model loaded successfully!")
            print(f"Model input shape: {self.model.input_shape}")
            print(f"Model output shape: {self.model.output_shape}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for prediction"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Resize to 64x64 to match model input
            resized = cv2.resize(rgb_frame, (64, 64))
            
            # Normalize pixel values to [0, 1]
            normalized = resized.astype('float32') / 255.0
            
            # Add batch dimension
            batch_frame = np.expand_dims(normalized, axis=0)
            
            return batch_frame
            
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None
    
    def predict_frame(self, frame):
        """Predict ISL sign from frame"""
        try:
            processed_frame = self.preprocess_frame(frame)
            if processed_frame is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(processed_frame, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'predictions': predictions[0],
                'timestamp': time.time()
            }
            
        except Exception as e:
            print(f"Error predicting frame: {e}")
            return None
    
    def get_stable_prediction(self):
        """Get stable prediction based on history"""
        if len(self.prediction_history) < self.stability_threshold:
            return None
        
        # Get recent predictions with high confidence
        recent_predictions = [
            pred['predicted_class'] for pred in self.prediction_history
            if pred['confidence'] >= self.confidence_threshold
        ]
        
        if len(recent_predictions) < self.stability_threshold:
            return None
        
        # Check if we have consistent predictions
        prediction_counts = Counter(recent_predictions[-self.stability_threshold:])
        most_common = prediction_counts.most_common(1)[0]
        
        if most_common[1] >= self.stability_threshold - 1:  # Allow 1 inconsistency
            return {
                'predicted_class': most_common[0],
                'confidence': np.mean([
                    pred['confidence'] for pred in self.prediction_history[-self.stability_threshold:]
                    if pred['predicted_class'] == most_common[0]
                ]),
                'stability': most_common[1] / self.stability_threshold
            }
        
        return None
    
    def add_letter_to_word(self, letter):
        """Add letter to current word with smart spacing"""
        current_time = time.time()
        
        # Check if this is a new letter or same letter held
        if letter != self.last_added_letter:
            self.current_word += letter
            self.last_added_letter = letter
            self.last_letter_time = current_time
            print(f"Added letter: {letter} | Current word: {self.current_word}")
        elif current_time - self.last_letter_time > self.letter_hold_time:
            # Same letter held for long time, might be intentional repetition
            self.current_word += letter
            self.last_letter_time = current_time
            print(f"Repeated letter: {letter} | Current word: {self.current_word}")
    
    def add_space(self):
        """Add space to current word"""
        if self.current_word and not self.current_word.endswith(' '):
            self.current_word += ' '
            self.last_added_letter = ' '
            print(f"Added space | Current word: '{self.current_word}'")
    
    def finish_word(self):
        """Finish current word and add to history"""
        if self.current_word.strip():
            self.word_history.append(self.current_word.strip())
            print(f"Word completed: '{self.current_word.strip()}'")
            self.current_word = ""
            self.last_added_letter = ""
    
    def clear_current_word(self):
        """Clear current word"""
        self.current_word = ""
        self.last_added_letter = ""
        print("Current word cleared")
    
    def backspace(self):
        """Remove last character from current word"""
        if self.current_word:
            self.current_word = self.current_word[:-1]
            print(f"Backspace | Current word: '{self.current_word}'")
    
    def capture_frames(self):
        """Capture frames in separate thread"""
        while self.is_recording:
            ret, frame = self.cap.read()
            if ret:
                # Put frame in queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except queue.Empty:
                        pass
            time.sleep(0.03)  # ~30 FPS
    
    def create_gui(self):
        """Create the GUI interface"""
        self.root = tk.Tk()
        self.root.title("ISL Sign Language Converter")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)
        
        # Left panel for video and controls
        left_panel = ttk.LabelFrame(main_frame, text="Video Feed & Controls", padding="10")
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        # Video display
        self.video_label = ttk.Label(left_panel, text="Video feed will appear here")
        self.video_label.pack(pady=10)
        
        # Control buttons
        button_frame = ttk.Frame(left_panel)
        button_frame.pack(pady=10)
        
        self.start_button = ttk.Button(button_frame, text="Start Camera", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Camera", command=self.stop_camera, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Word building controls
        word_control_frame = ttk.Frame(left_panel)
        word_control_frame.pack(pady=10)
        
        ttk.Button(word_control_frame, text="Add Space", command=self.add_space).pack(side=tk.LEFT, padx=2)
        ttk.Button(word_control_frame, text="Finish Word", command=self.finish_word).pack(side=tk.LEFT, padx=2)
        ttk.Button(word_control_frame, text="Backspace", command=self.backspace).pack(side=tk.LEFT, padx=2)
        ttk.Button(word_control_frame, text="Clear Word", command=self.clear_current_word).pack(side=tk.LEFT, padx=2)
        
        # Right panel for predictions and text
        right_panel = ttk.LabelFrame(main_frame, text="Predictions & Text", padding="10")
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(5, 0))
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(3, weight=1)
        
        # Current prediction
        ttk.Label(right_panel, text="Current Prediction:", font=('Arial', 12, 'bold')).grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        self.prediction_label = ttk.Label(right_panel, text="None", font=('Arial', 24, 'bold'), foreground='blue')
        self.prediction_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 10))
        
        # Confidence bar
        ttk.Label(right_panel, text="Confidence:", font=('Arial', 10)).grid(row=2, column=0, sticky=tk.W)
        self.confidence_bar = ttk.Progressbar(right_panel, length=300, mode='determinate')
        self.confidence_bar.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(5, 10))
        
        # Current word
        ttk.Label(right_panel, text="Current Word:", font=('Arial', 12, 'bold')).grid(row=4, column=0, sticky=tk.W, pady=(10, 5))
        self.word_label = ttk.Label(right_panel, text="", font=('Arial', 18), foreground='green', background='white', relief='sunken', padding="10")
        self.word_label.grid(row=5, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Word history
        ttk.Label(right_panel, text="Word History:", font=('Arial', 12, 'bold')).grid(row=6, column=0, sticky=tk.W, pady=(10, 5))
        
        # Scrollable text for history
        history_frame = ttk.Frame(right_panel)
        history_frame.grid(row=7, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        history_frame.columnconfigure(0, weight=1)
        history_frame.rowconfigure(0, weight=1)
        
        self.history_text = tk.Text(history_frame, height=10, font=('Arial', 11), wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(history_frame, orient=tk.VERTICAL, command=self.history_text.yview)
        self.history_text.configure(yscrollcommand=scrollbar.set)
        
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Settings frame
        settings_frame = ttk.LabelFrame(right_panel, text="Settings", padding="5")
        settings_frame.grid(row=8, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").grid(row=0, column=0, sticky=tk.W)
        self.confidence_scale = ttk.Scale(settings_frame, from_=0.5, to=0.95, 
                                        orient=tk.HORIZONTAL, length=200,
                                        command=self.update_confidence_threshold)
        self.confidence_scale.set(self.confidence_threshold)
        self.confidence_scale.grid(row=0, column=1, padx=(10, 0))
        
        self.confidence_value_label = ttk.Label(settings_frame, text=f"{self.confidence_threshold:.2f}")
        self.confidence_value_label.grid(row=0, column=2, padx=(5, 0))
    
    def update_confidence_threshold(self, value):
        """Update confidence threshold"""
        self.confidence_threshold = float(value)
        self.confidence_value_label.config(text=f"{self.confidence_threshold:.2f}")
    
    def start_camera(self):
        """Start camera and prediction"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_recording = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # Start frame capture thread
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.capture_thread.start()
            
            # Start GUI update loop
            self.update_gui()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {e}")
    
    def stop_camera(self):
        """Stop camera and prediction"""
        self.is_recording = False
        if self.cap:
            self.cap.release()
        
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.video_label.config(image='', text="Camera stopped")
    
    def update_gui(self):
        """Update GUI with latest frame and predictions"""
        if not self.is_recording:
            return
        
        try:
            # Get latest frame
            frame = None
            while not self.frame_queue.empty():
                frame = self.frame_queue.get_nowait()
            
            if frame is not None:
                # Display frame
                display_frame = cv2.resize(frame, (480, 360))
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                
                # Add ROI rectangle for hand detection area
                cv2.rectangle(display_frame, (120, 90), (360, 270), (0, 255, 0), 2)
                cv2.putText(display_frame, "Place hand here", (130, 85), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert to PhotoImage
                image = Image.fromarray(display_frame)
                photo = ImageTk.PhotoImage(image)
                self.video_label.config(image=photo, text="")
                self.video_label.image = photo
                
                # Make prediction if enough time has passed
                current_time = time.time()
                if current_time - self.last_prediction_time >= self.prediction_interval:
                    # Extract ROI for prediction
                    roi = frame[90:270, 120:360]  # Corresponding to rectangle
                    
                    prediction = self.predict_frame(roi)
                    if prediction:
                        self.prediction_history.append(prediction)
                        self.last_prediction_time = current_time
                        
                        # Update prediction display
                        self.prediction_label.config(text=prediction['predicted_class'])
                        self.confidence_bar['value'] = prediction['confidence'] * 100
                        
                        # Check for stable prediction
                        stable_pred = self.get_stable_prediction()
                        if stable_pred and stable_pred['confidence'] >= self.confidence_threshold:
                            self.add_letter_to_word(stable_pred['predicted_class'])
                
                # Update word display
                self.word_label.config(text=self.current_word)
                
                # Update history
                if self.word_history:
                    history_text = "\n".join([f"{i+1}. {word}" for i, word in enumerate(self.word_history)])
                    self.history_text.delete(1.0, tk.END)
                    self.history_text.insert(1.0, history_text)
            
        except Exception as e:
            print(f"GUI update error: {e}")
        
        # Schedule next update
        if self.is_recording:
            self.root.after(50, self.update_gui)  # 20 FPS GUI update
    
    def save_session(self):
        """Save current session to file"""
        if not self.word_history and not self.current_word:
            messagebox.showwarning("Warning", "No text to save!")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save ISL Session"
        )
        
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("ISL Sign Language Converter Session\n")
                    f.write("=" * 40 + "\n\n")
                    
                    if self.word_history:
                        f.write("Completed Words:\n")
                        for i, word in enumerate(self.word_history, 1):
                            f.write(f"{i}. {word}\n")
                        f.write("\n")
                    
                    if self.current_word:
                        f.write(f"Current Word: {self.current_word}\n")
                    
                    f.write(f"\nSession saved at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                
                messagebox.showinfo("Success", f"Session saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save session: {e}")
    
    def run(self):
        """Run the application"""
        if not self.load_model():
            messagebox.showerror("Error", "Failed to load model. Please check the model path.")
            return
        
        self.create_gui()
        
        # Add menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Session", command=self.save_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        print("ISL Converter GUI started. Load model and start camera to begin.")
        self.root.mainloop()
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_recording:
            self.stop_camera()
        self.root.destroy()

def main():
    """Main function"""
    print("ISL Sign Language Converter")
    print("=" * 40)
    
    # Check if model file exists
    model_path = 'models/classify_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure your trained model is saved at the correct path.")
        return
    
    # Create and run converter
    converter = ISLConverter(model_path)
    converter.run()

if __name__ == "__main__":
    main()



