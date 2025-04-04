import numpy as np

class EnsemblePredictor:
    def __init__(self):
        # Default weights
        self.default_soft_weight = 0.7
        self.default_hard_weight = 0.3
        self.confidence_threshold = 0.6  # Can be made dynamic

    def combined_voting_predict(self, text_score, face_score, behavior_score):
        # Validate inputs
        if any(score is None or not (0 <= score <= 1) for score in [text_score, face_score, behavior_score]):
            raise ValueError("All scores must be between 0 and 1.")

        # Soft Voting with confidence weighting
        scores = np.array([text_score, face_score, behavior_score])
        confidences = np.abs(scores - 0.5) + 0.5  # Confidence measure
        soft_weights = confidences / np.sum(confidences)
        soft_score = np.sum(scores * soft_weights)

        # Hard Voting with threshold
        hard_votes = (scores > self.confidence_threshold).astype(int)
        hard_score = np.mean(hard_votes)

        # Dynamic weight adjustment based on confidence spread
        confidence_spread = np.std(confidences)
        soft_weight = 0.8 if confidence_spread > 0.2 else self.default_soft_weight
        hard_weight = 1 - soft_weight  # Ensure weights sum to 1

        # Compute final score
        final_score = (soft_score * soft_weight) + (hard_score * hard_weight)

        return np.clip(final_score, 0, 1)  # Ensure score stays in [0,1]
