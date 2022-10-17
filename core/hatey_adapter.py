from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier

from core.model_trainer import generate_and_train_model
from model.toxicity_predictor_transformer import ToxicityPredictorTransformer
from util.file_io import FileIo
from util.logger import log


class HateyPredictor:
    def __init__(self):
        log.info("Initializing HateyPredictor...")
        self.transformer_model = ToxicityPredictorTransformer()

        cached_model = FileIo.cached('ensemble_classifier.pkl', generate_and_train_model)
        self.ensemble_classifier, self.ensemble_tokenizer = cached_model()
        log.info("HateyPredictor initialized.")

    def reasons(self, text):
        reasons = self.transformer_model.predict(text)
        reasons = [key for key, value in reasons.items() if value == 1]
        return ', '.join(reasons)

    def predictions(self, text):
        tokens = self.ensemble_tokenizer.tokenize([text])
        return {
            'transformer': self.transformer_model.predict(text),
            'ensemble': self.ensemble_classifier.predict_one_with_labels(tokens)
        }

    def is_hate_speech(self, text):
        return not self.transformer_model.is_sentence_clean(text)


hatey_predictor_singleton = HateyPredictor()

if __name__ == '__main__':
    print(hatey_predictor_singleton.predictions('I hate all of you.'))