# behavior_analyzer.py
# Module to analyze student behavior and predict stress/depression levels
# Uses rule-based and ML-based (RandomForest) approaches

from sklearn.ensemble import RandomForestRegressor
import numpy as np
import json

class BehaviorAnalyzer:
    def __init__(self):
        # Initialize RandomForest model for stress prediction
        self.model = RandomForestRegressor(n_estimators=300, random_state=42)
        self.initialize_model()

    def initialize_model(self):
        # Training data for RandomForest model
        X = np.array([
            [2, 6, 4, 3],  [7, 2, 8, 1],  [3, 5, 5, 2],  [8, 1, 7, 1],
            [1, 7, 4, 4],  [5, 3, 6, 2],  [9, 1, 8, 2],  [1, 9, 3, 7],
            [6, 4, 7, 3],  [2, 8, 4, 6],  [4, 7, 5, 5],  [10, 1, 9, 2],
            [1, 10, 3, 8], [12, 8, 0, 10], [0, 2, 8, 0]
        ])
        y = np.array([0.85, 0.2, 0.5, 0.15, 0.9, 0.55, 0.1, 0.95, 0.45, 0.88, 0.6, 0.05, 0.98, 1.0, 0.0])
        self.model.fit(X, y)
    
    def analyze(self, behavior_data):
        # Extract behavior data
        study_time, social_media, sleep_hours, deadlines = behavior_data
    
        # Rule-Based Stress Calculation
        rule_stress = 0.0
        if study_time > 9: rule_stress += 0.4
        elif study_time > 7: rule_stress += 0.3
        if social_media > 6: rule_stress += 0.4
        elif social_media > 4: rule_stress += 0.3
        if sleep_hours < 5: rule_stress += 0.4
        elif sleep_hours < 7: rule_stress += 0.3
        if deadlines > 6: rule_stress += 0.4
        elif deadlines > 4: rule_stress += 0.3

        # Rule-Based Depression Calculation
        rule_depression = 0.0
        if sleep_hours < 5: rule_depression += 0.5
        elif sleep_hours < 7: rule_depression += 0.4
        if social_media > study_time: rule_depression += 0.4
        elif social_media > study_time/2: rule_depression += 0.3
        if deadlines > 7: rule_depression += 0.4
        elif deadlines > 5: rule_depression += 0.3

        # Special case: Low workload (study_time and deadlines both 0)
        if study_time == 0 and deadlines == 0:
            rule_stress = min(rule_stress, 0.1)
            rule_depression = min(rule_depression, 0.1)

        # ML-Predicted Stress Score
        X = np.array([behavior_data])
        predicted_stress = self.model.predict(X)[0]

        # Final Score (Weighted + Boosted)
        final_stress_score = (predicted_stress * 0.6 + rule_stress * 0.4) * 1.2
        final_depression_score = rule_depression * 1.2

        # Ensure values between 0 and 1
        final_stress_score = np.clip(final_stress_score, 0, 1)
        final_depression_score = np.clip(final_depression_score, 0, 1)

        return {
            "stress_score": float(final_stress_score),
            "depression_score": float(final_depression_score)
        }

    def analyze_json(self, behavior_data):
        # JSON wrapper for analysis
        try:
            result = self.analyze(behavior_data)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})