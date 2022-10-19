import stringcase as stringcase
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, \
    GradientBoostingClassifier

from core.model_trainer import generate_and_train_model
from model.sentiment_analyser import SentimentAnalyser
from model.tokenizer import NltkTokenizer
from model.toxicity_predictor_transformer import ToxicityPredictorTransformer
from util.file_io import FileIo
from util.logger import log


class HateyPredictor:
    """
    This class is a wrapper for the toxicity predictor and the ensemble classifier.
    """

    def __init__(self):
        log.info("Initializing HateyPredictor...")
        self.transformer_model = ToxicityPredictorTransformer()
        self.classifiers = {}
        self.tokenizer = NltkTokenizer()
        self.sentiment_analyser = SentimentAnalyser()

        for name, classifier in self._base_classifiers().items():
            cached_model = FileIo.cached(f"train_{stringcase.snakecase(name)}.pkl",
                                         lambda x: generate_and_train_model(base_classifier=x,
                                                                            tokenizer=self.tokenizer))
            self.classifiers[name] = cached_model(classifier)
        log.info("HateyPredictor initialized.")

    def _base_classifiers(self):
        return {
            'BaggingClassifier': BaggingClassifier(base_estimator=RandomForestClassifier()),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'GradientBoostingClassifier': GradientBoostingClassifier()
        }

    def reasons(self, text):
        reasons = self.transformer_model.reasons(text)
        return [stringcase.sentencecase(reason) for reason in reasons]

    def reasons_as_text(self, text):
        return ", ".join(self.reasons(text))

    def problematic_words(self, text):
        return [word for word in self.transformer_model.problematic_words(text)]

    def predictions(self, text):
        tokens = self.tokenizer.tokenize([text])
        result = {
            'Transformer': self.clean(self.transformer_model.predict(text)),
        }
        for name, classifier in self.classifiers.items():
            result[name] = self.clean(classifier.predict_one_with_labels(tokens))

        return result

    def sentiment(self, text):
        return self.clean(self.sentiment_analyser.predict(text))

    def is_hate_speech(self, text):
        return not self.transformer_model.is_sentence_clean(text)

    def clean(self, param):
        return {stringcase.sentencecase(key): str(value) for key, value in param.items()}


hatey_predictor_singleton = HateyPredictor()
