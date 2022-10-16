from detoxify import Detoxify

FILTER_SETTING = ['toxicity', 'severe_toxicity', 'obscene', 'threat', 'insult', 'identity_attack']


class ToxicityPredictorTransformer:
    def __init__(self, filter_setting=None, threshold=0.5):
        if filter_setting is None:
            filter_setting = FILTER_SETTING
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

    def apply_filter(self, prediction):
        for key in self.filter_setting:
            if prediction[key] == 1:
                return False
        return True

    def is_this_sentence_clean(self, sentence):
        prediction = self.predict(sentence)
        return self.apply_filter(prediction)
