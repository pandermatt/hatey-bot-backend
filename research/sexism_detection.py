import numpy as np
import sklearn
import stringcase

from core.hatey_predictor import hatey_predictor_singleton
from data_model.sexist_data import SexistData
from util.file_io import FileIo

if __name__ == '__main__':
    dataset = SexistData()

    X = dataset.get_data()
    Y = dataset.get_label()
    print(len(X))
    print(len(Y))
    print(np.unique(Y, return_counts=True))

    tokenizer = hatey_predictor_singleton.tokenizer
    cached_tokens = FileIo.cached(f'train_nltk_tokens_{stringcase.snakecase(dataset.__class__.__name__)}.pkl',
                                    lambda x: tokenizer.tokenize(x))
    X = cached_tokens(X)

    classifier = hatey_predictor_singleton.classifiers['BaggingClassifier']

    Y_pred = classifier.predict(X)
    Y_pred = [0 if y == 0 else 1 for y in Y_pred]

    confusion_matrix = sklearn.metrics.confusion_matrix(Y, Y_pred)
    confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    print(confusion_matrix)
    print(sklearn.metrics.classification_report(Y, Y_pred))
