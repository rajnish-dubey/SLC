import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

class ISLPredictor:
    def __init__(self, model_path='models/sign_language_model.h5'):
        self.model_path = model_path
        self.model = None
        self.class_names = [
             '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
            'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]
        
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
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not read image from {image_path}")
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # CRITICAL FIX: Resize to 64x64 to match model input
            image = cv2.resize(image, (64, 64))
            
            # Normalize pixel values to [0, 1]
            image = image.astype('float32') / 255.0
            
            # Add batch dimension
            image = np.expand_dims(image, axis=0)
            
            print(f"Preprocessed image shape: {image.shape}")
            return image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_image(self, image_path):
        """Predict ISL sign from image"""
        try:
            print(f"Testing with image: {image_path}")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            if processed_image is None:
                return None
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get predicted class
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            predicted_class = self.class_names[predicted_class_idx]
            
            return {
                'predicted_class': predicted_class,
                'confidence': float(confidence),
                'predictions': predictions[0]
            }
            
        except Exception as e:
            print(f"Error processing image: {e}")
            return None
    
    def predict_webcam(self):
        """Real-time prediction from webcam"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print("Press 'q' to quit, 'space' to capture and predict")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display frame
            cv2.imshow('ISL Recognition - Press Space to Predict', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space key
                # Save temporary image
                temp_path = 'temp_capture.jpg'
                cv2.imwrite(temp_path, frame)
                
                # Predict
                result = self.predict_image(temp_path)
                if result:
                    print(f"\nPrediction: {result['predicted_class']}")
                    print(f"Confidence: {result['confidence']:.2%}")
                
                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        cap.release()
        cv2.destroyAllWindows()

def main():
    print("Simple ISL Model Tester")
    print("1. Webcam test")
    print("2. Image file test")
    
    choice = input("Choose option (1 or 2): ").strip()
    
    # Initialize predictor
    predictor = ISLPredictor()
    
    # Load model
    if not predictor.load_model():
        return
    
    if choice == '1':
        predictor.predict_webcam()
    elif choice == '2':
        image_path = input("Enter image path: ").strip()
        
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return
        
        result = predictor.predict_image(image_path)
        if result:
            print(f"\n{'='*50}")
            print(f"Predicted Sign: {result['predicted_class']}")
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"{'='*50}")
            
            # Show top 3 predictions
            top_indices = np.argsort(result['predictions'])[::-1][:3]
            print("\nTop 3 predictions:")
            for i, idx in enumerate(top_indices, 1):
                class_name = predictor.class_names[idx]
                confidence = result['predictions'][idx]
                print(f"{i}. {class_name}: {confidence:.2%}")
    else:
        print("Invalid choice. Please select 1 or 2.")

if __name__ == "__main__":
    main()