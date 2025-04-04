# text_analyzer.py
# Module to analyze text sentiment using DistilBERT

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

class TextAnalyzer:
    def __init__(self, model_path="distilbert-base-uncased"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)

    def preprocess_text(self, text):
        cleaned_text = text.lower() if text else ""
        return cleaned_text

    @torch.no_grad()
    def predict(self, text):
        self.load_model()  # Load model only when needed
        cleaned_text = self.preprocess_text(text)
        if not cleaned_text:
            return 0.5, "Neutral"

        inputs = self.tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model(**inputs)
        bert_score = torch.nn.functional.softmax(outputs.logits, dim=1)[0][1].item()

        if bert_score < 0.6:
            sentiment = "Positive"
        elif bert_score <= 0.8:
            sentiment = "Neutral"
        else:
            sentiment = "Negative"

        final_score = bert_score * 1.2
        final_score = min(1.0, max(0.0, final_score))
        return final_score, sentiment

if __name__ == "__main__":
    analyzer = TextAnalyzer()
    texts = ["I feel great and focused!", "I feel terrible and overwhelmed!"]
    for text in texts:
        score, sentiment = analyzer.predict(text)
        print(f"Text: {text}, Score: {score:.2f}, Sentiment: {sentiment}")