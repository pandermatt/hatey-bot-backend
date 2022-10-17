import stringcase
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier

from data_model.toxic_comment_data import ToxicCommentData
from model.ensemble_classifier import EnsembleClassifier
from model.tokenizer import NLTKTokenizer
from util.file_io import FileIo
from util.logger import log


def generate_and_train_model(dataset=None, tokenizer=None, base_classifier=None):
    """
    This function generates a model and trains it on the given dataset.
    :param dataset: The dataset to train the model on.
    :param tokenizer: The tokenizer to use for the model.
    :param base_classifier: The base classifier to use for the EnsembleClassifier.
    :return: The trained model and tokenizer.
    """
    if dataset is None:
        dataset = ToxicCommentData()
    if tokenizer is None:
        tokenizer = NLTKTokenizer()

    X = dataset.get_data()
    Y = dataset.get_label()
    cached_tokens = FileIo.cached(f'train_nltk_tokens_{stringcase.snakecase(dataset.__class__.__name__)}.pkl',
                                  lambda x: tokenizer.tokenize(x))
    X = cached_tokens(X)

    log.info("Training model...")
    if base_classifier is None:
        base_classifier = BaggingClassifier(base_estimator=ExtraTreesClassifier(),
                                            n_estimators=10, max_samples=0.5,
                                            max_features=0.5)
    model = EnsembleClassifier(classifier=base_classifier, label_names=dataset.get_label_names())
    model.train(X, Y)
    log.info("Model trained.")

    return model, tokenizer
