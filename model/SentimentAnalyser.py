from transformers import pipeline


class SentimentAnalyser:
    def __init__(self):
        self.model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment", top_k=3)
        self.labels = {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}

    def predict(self, text):
        """
        Predicts the sentiment of the text
        :param text: text to predict
        :return: dict with the sentiment and the confidence (negative, neutral, positive)
        """
        result = self.model(text)[0]
        results = {self.labels[res["label"]]: res['score'] for res in result}
        return results
