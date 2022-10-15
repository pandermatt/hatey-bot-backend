from detoxify import Detoxify

class ToxicityPredictorTransformer:
    def __init__(self):
        self.model = Detoxify('original')
        self.threshold = 0.5


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