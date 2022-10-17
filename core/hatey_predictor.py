import stringcase as stringcase

from core.model_trainer import generate_and_train_model
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

        cached_model = FileIo.cached('ensemble_classifier.pkl', generate_and_train_model)
        self.ensemble_classifier, self.ensemble_tokenizer = cached_model()
        log.info("HateyPredictor initialized.")

    def reasons(self, text):
        reasons = self.transformer_model.reasons(text)
        return ", ".join([stringcase.sentencecase(reason) for reason in reasons])

    def problematic_words(self, text):
        return [word for word in self.transformer_model.problematic_words(text)]

    def predictions(self, text):
        tokens = self.ensemble_tokenizer.tokenize([text])

        return {
            'Transformer': self.clean(self.transformer_model.predict(text)),
            'Classifier': self.clean(self.ensemble_classifier.predict_one_with_labels(tokens))
        }

    def is_hate_speech(self, text):
        return not self.transformer_model.is_sentence_clean(text)

    def clean(self, param):
        return {stringcase.sentencecase(key): str(value) for key, value in param.items()}


hatey_predictor_singleton = HateyPredictor()
