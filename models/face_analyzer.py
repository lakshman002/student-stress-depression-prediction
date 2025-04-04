# face_analyzer.py
# Module to analyze facial emotions using FER (Facial Expression Recognition)

import os
import cv2
import numpy as np
from fer import FER

# Disable GPU usage for TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

class FaceAnalyzer:
    def __init__(self):
        self.emotion_detector = None

    def load_model(self):
        if self.emotion_detector is None:
            self.emotion_detector = FER(mtcnn=True)

    def predict(self, image_path):
        try:
            self.load_model()  # Load model only when needed
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not load image at {image_path}. Check file path or format.")
                return 0.5, "Unknown"

            max_size = 1000
            height, width = image.shape[:2]
            if max(height, width) > max_size:
                scale = max_size / max(height, width)
                image = cv2.resize(image, (int(width * scale), int(height * scale)))

            # Detect emotions using FER
            emotions = self.emotion_detector.detect_emotions(image)
            if not emotions:
                print(f"No faces detected in {image_path}. Using default stress score.")
                return 0.6, "Unknown"

            dominant_emotion = emotions[0]['emotions']
            print(f"Raw emotion scores: {dominant_emotion}")
            emotion_name = max(dominant_emotion, key=dominant_emotion.get)
            stress_score = (
                dominant_emotion.get('angry', 0) * 0.9 +
                dominant_emotion.get('fear', 0) * 0.8 +
                dominant_emotion.get('sad', 0) * 0.7 +
                dominant_emotion.get('neutral', 0) * 0.5 +
                dominant_emotion.get('surprise', 0) * 0.4 +
                dominant_emotion.get('happy', 0) * 0.2
            )
            return min(max(stress_score, 0.0), 1.0), emotion_name
        except Exception as e:
            print(f"Error in face analysis: {str(e)}. Using default stress score.")
            return 0.6, "Unknown"