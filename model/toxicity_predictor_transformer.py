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
        clean = True
        for key in prediction.keys():
            if prediction[key] > self.threshold:
                prediction[key] = 1
                clean = False
            else:
                prediction[key] = 0
        prediction['clean'] = 1 if clean else 0
        return prediction

    def is_sentence_clean(self, sentence):
        prediction = self.predict(sentence)
        return prediction['clean'] == 1
