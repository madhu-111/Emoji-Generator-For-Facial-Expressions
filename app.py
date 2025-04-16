import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

class EmotionEmojiGenerator:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emoji_mapping = {
            'angry': '😠',
            'disgust': '🤢',
            'fear': '😨',
            'happy': '😊',
            'sad': '😢',
            'surprise': '😲',
            'neutral': '😐'
        }
        
        # Initialize model
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
    def build_model(self):
        """Build a scikit-learn pipeline for emotion detection"""
        # Create pipeline with feature scaling and SVM classifier
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True))
        ])
        return self.model
    
    def extract_features_from_image(self, image_path):
        """Extract HOG features from a single image file"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        
        # Resize to 48x48
        image = cv2.resize(image, (48, 48))
        
        # Extract HOG features
        features = self.extract_hog_features(image)
        return features
    
    def extract_hog_features(self, image):
        """Extract HOG features from an image array"""
        if image is None:
            return None
            
        # Ensure image is 48x48
        if image.shape != (48, 48):
            image = cv2.resize(image, (48, 48))
            
        # HOG parameters
        win_size = (48, 48)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        nbins = 9
        
        # Initialize HOG descriptor
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
        
        # Compute HOG features
        features = hog.compute(image)
        return features.flatten()
    
    def load_dataset(self, dataset_dir):
        """Load dataset from directory structure"""
        X = []  # Features
        y = []  # Labels
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(dataset_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
                
            print(f"Loading {emotion} images...")
            image_files = [f for f in os.listdir(emotion_dir) 
                          if f.endswith('.jpg') or f.endswith('.png')]
            
            for img_file in image_files[:500]:  # Limit to 500 per class to avoid memory issues
                img_path = os.path.join(emotion_dir, img_file)
                features = self.extract_features_from_image(img_path)
                if features is not None:
                    X.append(features)
                    y.append(emotion_idx)
        
        return np.array(X), np.array(y)
    
    def train_model(self, dataset_dir, test_size=0.2):
        """Train the model using the provided dataset"""
        print("Loading dataset...")
        X, y = self.load_dataset(dataset_dir)
        
        if len(X) == 0:
            print("No valid images found in the dataset!")
            return None
            
        print(f"Dataset loaded: {X.shape[0]} samples")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42)
        
        print("Training model...")
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def save_model(self, model_path):
        """Save the trained model"""
        if self.model is not None:
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
        else:
            print("No model to save!")
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file not found: {model_path}")
            return False
    
    def preprocess_image(self, image):
        """Preprocess an image for prediction"""
        # Convert to grayscale if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            # Process the largest face
            face_idx = np.argmax([w*h for (x, y, w, h) in faces])
            (x, y, w, h) = faces[face_idx]
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Resize to 48x48
            face_roi = cv2.resize(face_roi, (48, 48))
            
            # Extract HOG features
            features = self.extract_hog_features(face_roi)
            
            return features, (x, y, w, h)
        else:
            return None, None
    
    def predict_emotion(self, image):
        """Predict emotion from an image"""
        if self.model is None:
            print("No model loaded!")
            return None, None, None, None
            
        features, face_coords = self.preprocess_image(image)
        
        if features is not None:
            # Reshape features for prediction if needed
            features = features.reshape(1, -1)
            
            # Get prediction
            prediction = self.model.predict_proba(features)[0]
            emotion_idx = np.argmax(prediction)
            emotion = self.emotions[emotion_idx]
            confidence = prediction[emotion_idx]
            emoji = self.emoji_mapping[emotion]
            
            return emotion, confidence, emoji, face_coords
        else:
            return None, None, None, None


class EmotionEmojiUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Emotion Emoji Generator")
        self.root.geometry("800x600")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize the generator
        self.generator = EmotionEmojiGenerator()
        self.generator.build_model()
        
        # Try to load a pre-trained model if available
        model_path = "emotion_model.joblib"
        if not self.generator.load_model(model_path):
            messagebox.showinfo("Model Not Found", 
                              "No pre-trained model found. Please train a model using the 'Train Model' button.")
        
        # Initialize video capture
        self.cap = None
        self.is_webcam_on = False
        
        # Create UI elements
        self.create_widgets()
    
    def create_widgets(self):
        # Header frame
        header_frame = tk.Frame(self.root, bg="#4a6fa5")
        header_frame.pack(fill=tk.X, pady=0)
        
        title_label = tk.Label(header_frame, text="Emotion Emoji Generator", 
                              font=("Helvetica", 24, "bold"), bg="#4a6fa5", fg="white")
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Left panel for video/image
        self.left_panel = tk.Frame(main_container, bg="#ffffff", width=400)
        self.left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Video canvas
        self.canvas = tk.Canvas(self.left_panel, bg="black", width=400, height=300)
        self.canvas.pack(pady=10)
        
        # Controls frame
        controls_frame = tk.Frame(self.left_panel, bg="#ffffff")
        controls_frame.pack(fill=tk.X, pady=10)
        
        self.webcam_btn = tk.Button(controls_frame, text="Start Webcam", 
                                   command=self.toggle_webcam, bg="#4a6fa5", fg="white",
                                   font=("Helvetica", 12), width=15)
        self.webcam_btn.pack(side=tk.LEFT, padx=10)
        
        self.upload_btn = tk.Button(controls_frame, text="Upload Image", 
                                  command=self.upload_image, bg="#4a6fa5", fg="white",
                                  font=("Helvetica", 12), width=15)
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.snapshot_btn = tk.Button(controls_frame, text="Take Snapshot", 
                                    command=self.take_snapshot, bg="#4a6fa5", fg="white",
                                    font=("Helvetica", 12), width=15)
        self.snapshot_btn.pack(side=tk.LEFT, padx=10)
        
        # Right panel for results
        self.right_panel = tk.Frame(main_container, bg="#ffffff", width=300)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        result_label = tk.Label(self.right_panel, text="Emotion Results", 
                              font=("Helvetica", 16, "bold"), bg="#ffffff")
        result_label.pack(pady=10)
        
        # Emoji display
        self.emoji_label = tk.Label(self.right_panel, text="😐", 
                                  font=("Helvetica", 72), bg="#ffffff")
        self.emoji_label.pack(pady=20)
        
        # Emotion text
        self.emotion_label = tk.Label(self.right_panel, text="No emotion detected", 
                                    font=("Helvetica", 14), bg="#ffffff")
        self.emotion_label.pack(pady=5)
        
        # Confidence bar
        self.confidence_frame = tk.Frame(self.right_panel, bg="#ffffff")
        self.confidence_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(self.confidence_frame, text="Confidence:", 
               font=("Helvetica", 12), bg="#ffffff").pack(side=tk.LEFT)
        
        self.confidence_bar = tk.Canvas(self.confidence_frame, height=20, width=200,
                                      bg="#e0e0e0", highlightthickness=1)
        self.confidence_bar.pack(side=tk.LEFT, padx=10)
        
        # Training section
        training_frame = tk.Frame(self.right_panel, bg="#ffffff", padx=10, pady=10)
        training_frame.pack(fill=tk.X, pady=20)
        
        tk.Label(training_frame, text="Model Training", 
               font=("Helvetica", 14, "bold"), bg="#ffffff").pack(anchor="w")
        
        tk.Button(training_frame, text="Train Model", command=self.train_model,
                bg="#4a6fa5", fg="white", font=("Helvetica", 12)).pack(pady=10)
        
        self.status_label = tk.Label(training_frame, text="Model status: Ready", 
                                   font=("Helvetica", 10), bg="#ffffff")
        self.status_label.pack(pady=5)
    
    def toggle_webcam(self):
        if self.is_webcam_on:
            # Turn off webcam
            self.is_webcam_on = False
            self.webcam_btn.config(text="Start Webcam")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.canvas.delete("all")
        else:
            # Turn on webcam
            self.cap = cv2.VideoCapture(0)
            if self.cap.isOpened():
                self.is_webcam_on = True
                self.webcam_btn.config(text="Stop Webcam")
                self.update_frame()
            else:
                self.status_label.config(text="Error: Could not open webcam")
    
    def update_frame(self):
        if self.is_webcam_on and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # Make a copy of the frame for processing
                process_frame = frame.copy()
                
                # Process frame for emotion detection
                emotion, confidence, emoji, face_coords = self.generator.predict_emotion(process_frame)
                
                # Draw rectangle around face if detected
                if face_coords is not None:
                    x, y, w, h = face_coords
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Update UI with results
                if emotion:
                    self.update_results(emotion, confidence, emoji)
                
                # Convert to RGB for tkinter
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)
                
                # Display in canvas
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
            self.root.after(10, self.update_frame)
    
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )
        
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                # Process image for emotion detection
                emotion, confidence, emoji, face_coords = self.generator.predict_emotion(image)
                
                # Draw rectangle around face if detected
                if face_coords is not None:
                    x, y, w, h = face_coords
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Update UI with results
                if emotion:
                    self.update_results(emotion, confidence, emoji)
                else:
                    self.update_results(None, 0, "😐")
                
                # Display image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_image)
                img = img.resize((400, 300), Image.Resampling.LANCZOS)
                self.photo = ImageTk.PhotoImage(image=img)
                
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
    
    def take_snapshot(self):
        if self.cap is not None and self.is_webcam_on:
            ret, frame = self.cap.read()
            if ret:
                # Save snapshot
                file_path = filedialog.asksaveasfilename(
                    defaultextension=".jpg",
                    filetypes=[("JPEG files", "*.jpg")]
                )
                
                if file_path:
                    cv2.imwrite(file_path, frame)
                    self.status_label.config(text=f"Snapshot saved to {file_path}")
    
    def update_results(self, emotion, confidence, emoji):
        if emotion:
            self.emoji_label.config(text=emoji)
            self.emotion_label.config(text=f"Detected emotion: {emotion.capitalize()}")
            
            # Update confidence bar
            self.confidence_bar.delete("all")
            bar_width = int(200 * confidence)
            self.confidence_bar.create_rectangle(0, 0, bar_width, 20, fill="#4a6fa5", outline="")
            self.confidence_bar.create_text(100, 10, text=f"{confidence:.2%}", fill="black")
        else:
            self.emoji_label.config(text="😐")
            self.emotion_label.config(text="No face detected")
            self.confidence_bar.delete("all")
    
    def train_model(self):
        # Open a dialog to select training data directory
        dataset_dir = filedialog.askdirectory(title="Select Dataset Directory")
        if dataset_dir:
            self.status_label.config(text="Training model... This may take a while.")
            self.root.update()
            
            # Train the model (this will run in main thread - could be improved with threading)
            try:
                accuracy = self.generator.train_model(dataset_dir)
                if accuracy is not None:
                    # Save the model
                    self.generator.save_model("emotion_model.joblib")
                    self.status_label.config(text=f"Model trained with {accuracy:.2%} accuracy and saved!")
                else:
                    self.status_label.config(text="Training failed. Check dataset format.")
            except Exception as e:
                messagebox.showerror("Training Error", f"An error occurred during training: {str(e)}")
                self.status_label.config(text="Training failed. See error message.")


if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionEmojiUI(root)
    root.mainloop()