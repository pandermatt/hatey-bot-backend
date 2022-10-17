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
                clean = False
        prediction['non-toxic'] = 1 if clean else 0
        return prediction

    def apply_filter(self, prediction):
        for key in self.filter_setting:
            if prediction[key] > self.threshold:
                return False
        return True

    def is_sentence_clean(self, sentence):
        prediction = self.predict(sentence)
        return self.apply_filter(prediction)

    def reasons(self, text):
        prediction = self.predict(text)
        return [key for key in self.filter_setting if prediction[key] > self.threshold]

    def problematic_words(self, text):
        return [word for word in text.split() if not self.is_sentence_clean(word)]