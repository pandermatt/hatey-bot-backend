from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier

from data_model.toxic_comment_data import ToxicCommentData
from model.ensemble_classifier import EnsembleClassifier
from model.tokenizer import NLTKTokenizer
from util.file_io import FileIo


def generate_and_train_model(dataset=None, tokenizer=None, base_classifier=None):
    if dataset is None:
        dataset = ToxicCommentData()
    if tokenizer is None:
        tokenizer = NLTKTokenizer()

    X = dataset.get_data()
    Y = dataset.get_label()
    cached_tokens = FileIo.cached(f'train_nltk_tokens_{dataset.__class__.__name__}.pkl',
                                  lambda x: tokenizer.tokenize(x))
    X = cached_tokens(X)

    if base_classifier is None:
        base_classifier = BaggingClassifier(base_estimator=ExtraTreesClassifier(),
                                            n_estimators=10, max_samples=0.5,
                                            max_features=0.5)
    model = EnsembleClassifier(classifier=base_classifier, label_names=dataset.get_label_names())
    model.train(X, Y)

    return model, tokenizer
