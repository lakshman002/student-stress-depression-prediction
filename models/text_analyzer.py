# text_analyzer.py
# Module to analyze text sentiment using DistilBERT

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class TextAnalyzer:
    def __init__(self, model_path="distilbert-base-uncased"):
        # Initialize DistilBERT tokenizer and model
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    def preprocess_text(self, text):
        # Preprocess text by converting to lowercase
        cleaned_text = text.lower() if text else ""
        return cleaned_text
    
    @torch.no_grad()
    def predict(self, text):
        # Predict sentiment and stress score from text
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return 0.5, "Neutral"
        
        # DistilBERT prediction
        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        bert_score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][1].item()
        
        # Determine sentiment based on score
        if bert_score < 0.6:
            sentiment = "Positive"
        elif bert_score <= 0.8:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"
        
        # Boost the final score
        final_score = bert_score * 1.2
        final_score = min(1.0, max(0.0, final_score))
        return final_score, sentiment

# Test the analyzer
if __name__ == "__main__":
    analyzer = TextAnalyzer()
    texts = ["I feel great and focused!", "I feel terrible and overwhelmed!"]
    for text in texts:
        score, sentiment = analyzer.predict(text)
        print(f"Text: {text}, Score: {score:.2f}, Sentiment: {sentiment}")