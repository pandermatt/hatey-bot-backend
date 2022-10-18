from transformers import pipeline


# Input: string, containing the text to be analysed
# Output: string, containing the sentiment of the text (positive, negative or neutral)
class SentimentAnalyser:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", top_k=3)
        self.labels = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

    def predict(self, text):
        result = self.model(text)[0]
        results = {self.labels[res["label"]]: res['score'] for res in result}
        return results