# ToxicityDetection class
from detoxify import Detoxify

class ToxicityPredictorTransformer:
    def __init__(self, filter_setting=['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack'], threshold=0.5):
        self.model = Detoxify('original')
        self.filter_setting = filter_setting
        self.threshold = threshold

    def predict(self, sentence):
        prediction = self.model.predict(sentence)
        prediction['clean'] = 0
        clean = True
        for key in prediction.keys():
            if prediction[key] > self.threshold:
                prediction[key] = 1
                clean = False
            else:
                prediction[key] = 0
        if clean:
            prediction['clean'] = 1
        return prediction

    def applyFilter(self, prediction):
        for key in self.filter_setting:
            if prediction[key] == 1:
                return False
        return True

    def isThisSentenceClean(self, sentence):
        prediction = self.predict(sentence)
        return self.applyFilter(prediction)