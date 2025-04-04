# app.py
# Flask application to predict student stress and depression levels
# Uses text analysis, face emotion detection, and behavior analysis
# Author: [Lakshman]
# Date: April 2025

from flask import Flask, request, jsonify, render_template
import os
import json
from models.behavior_analyzer import BehaviorAnalyzer
from models.text_analyzer import TextAnalyzer
from models.face_analyzer import FaceAnalyzer
from models.ensemble_predictor import EnsemblePredictor
from werkzeug.utils import secure_filename

# Initialize Flask app
app = Flask(__name__)

# Set up upload folder for images
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Enable CORS for cross-origin requests
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialize model analyzers
behavior_analyzer = BehaviorAnalyzer()  # Analyzes study behavior
text_analyzer = TextAnalyzer()  # Analyzes text sentiment
face_analyzer = FaceAnalyzer()  # Analyzes facial emotions
ensemble_predictor = EnsemblePredictor()  # Combines scores for final prediction
STORAGE_FILE = os.path.join(os.path.dirname(__file__), 'stress_logs.json')  # File to store stress logs

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle prediction requests
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.form
        files = request.files

        student_id = data.get('student_id', 'N/A')
        text_input = data.get('text', None)
        study_behavior = json.loads(data.get('study_behavior', '[]'))
        if not study_behavior or len(study_behavior) != 4:
            return jsonify({'error': 'Invalid study behavior data'}), 400

        # Log the request details
        print("\n=== New Prediction Request ===")
        print(f"Student ID: {student_id}")
        print(f"Text Input: {text_input if text_input else 'Not provided'}")
        print(f"Study Behavior: {study_behavior}")
        print(f"Image: {'Provided' if 'image' in files else 'Not provided'}")

        # Model scores with sentiment/emotion
        text_result = text_analyzer.predict(text_input) if text_input else (0.5, "Neutral")
        text_score, text_sentiment = text_result  # Unpack text score and sentiment

        face_score, face_emotion = 0.5, "Unknown"
        image_path = None
        if 'image' in files:
            file = files['image']
            if file and file.filename:
                filename = secure_filename(file.filename)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(image_path)  # Save the uploaded image
                if os.path.exists(image_path):
                    face_result = face_analyzer.predict(image_path)
                    face_score, face_emotion = face_result  # Unpack face score and emotion
                else:
                    print(f"Warning: Could not save image at {image_path}")

        # Analyze study behavior
        behavior_result = behavior_analyzer.analyze(study_behavior)
        behavior_stress = float(behavior_result["stress_score"])
        behavior_depression = float(behavior_result["depression_score"])

        # Log individual model scores
        print("\n=== Individual Model Scores ===")
        print(f"Text Score: {text_score:.2f} (Sentiment: {text_sentiment})")
        print(f"Face Score: {face_score:.2f} (Emotion: {face_emotion})")
        print(f"Behavior Stress Score: {behavior_stress:.2f}")
        print(f"Behavior Depression Score: {behavior_depression:.2f}")

        # Adjust text score scaling for normal cases
        adjusted_text_score = text_score
        if text_sentiment == "Positive" and behavior_stress < 0.5:
            adjusted_text_score = max(0.2, text_score * 0.5)  # Reduce text score impact

        # Adjust ensemble weights based on behavior scores
        if behavior_stress > 0.9 or behavior_depression > 0.9:
            weights = (0.25, 0.25, 0.5)  # More weight to behavior in extreme cases
        else:
            weights = (0.3, 0.3, 0.4)  # More weight to behavior in normal cases
        
        # Calculate final stress and depression scores
        final_stress = (adjusted_text_score * weights[0] + face_score * weights[1] + behavior_stress * weights[2])
        final_depression = (adjusted_text_score * weights[0] + face_score * weights[1] + behavior_depression * weights[2])

        # Determine stress and depression levels
        stress_level = get_stress_level(final_stress)
        depression_level = get_depression_level(final_depression)

        # Log combined scores
        print("\n=== Combined Voting Fusion Scores ===")
        print(f"Stress Score: {final_stress:.2f} (Level: {stress_level})")
        print(f"Depression Score: {final_depression:.2f} (Level: {depression_level})")
        print("==============================")

        # Generate recommendations and alerts
        recommendations = generate_recommendations(final_stress, final_depression)
        alert_counselor = bool(final_stress > 0.75 or final_depression > 0.75)
        alert_proctor = bool(final_stress > 0.6 or final_depression > 0.6)

        # Prepare response
        response = jsonify({
            'stress_score': final_stress,
            'depression_score': final_depression,
            'stress_level': stress_level,
            'depression_level': depression_level,
            'recommendations': recommendations,
            'alert_counselor': alert_counselor,
            'alert_proctor': alert_proctor,
            'text_sentiment': text_score
        })

        # Clean up: Remove the uploaded image
        if image_path and os.path.exists(image_path):
            os.remove(image_path)

        return response

    except Exception as e:
        # Log any errors
        print(f"\n=== Error ===")
        print(f"Error in prediction: {str(e)}")
        print("==============================")
        return jsonify({'error': str(e)}), 500

# Helper function to determine stress level
def get_stress_level(score):
    if score >= 0.76: return 'Severe'
    elif score >= 0.51: return 'High'
    elif score >= 0.26: return 'Moderate'
    else: return 'Normal'

# Helper function to determine depression level
def get_depression_level(score):
    if score >= 0.76: return 'Severe'
    elif score >= 0.51: return 'High'
    elif score >= 0.24: return 'Moderate'
    else: return 'Normal'

# Helper function to generate recommendations based on scores
def generate_recommendations(stress_score, depression_score):
    recommendations = []
    if stress_score > 0.75:
        recommendations.extend(["Consider speaking with a counselor immediately", "Take regular breaks between study sessions", "Practice deep breathing exercises"])
    elif stress_score > 0.5:
        recommendations.extend(["Try to maintain a balanced study schedule", "Consider reducing social media usage", "Ensure you get enough sleep"])
    if depression_score > 0.75:
        recommendations.extend(["Please seek professional help", "Connect with friends and family", "Maintain a regular daily routine"])
    elif depression_score > 0.5:
        recommendations.extend(["Consider joining student support groups", "Engage in physical activities", "Set achievable daily goals"])
    if not recommendations:
        recommendations = ["Keep maintaining your current healthy routine", "Stay connected with friends and family", "Regular exercise helps maintain mental health"]
    return recommendations

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)