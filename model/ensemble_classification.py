import numpy as np
import seaborn as sns
from keras.layers import Dense, Dropout
from keras.models import Sequential
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import tensorflow as tf

from config import config


class EnsembleClassification:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        # self.classifier = ExtraTreesClassifier()
        # self.classifier = RandomForestClassifier()
        self.classifier = BaggingClassifier(base_estimator=ExtraTreesClassifier(), n_estimators=10, max_samples=0.5,
                                            max_features=0.5)

    def train(self, X, Y):
        x_tfidf = self.tfidf_vectorizer.fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_tfidf, Y,
                                                                                test_size=0.2,
                                                                                random_state=42,
                                                                                stratify=Y)

        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, text):
        X = self.tfidf_vectorizer.transform(text)
        return self.classifier.predict(X)

    def predict_with_probability(self, text):
        X = self.tfidf_vectorizer.transform(text)
        return self.classifier.predict_proba(X)

    def classification_report(self):
        return classification_report(self.y_test, self.classifier.predict(self.X_test))

    def confusion_matrix(self, labels=None):
        cm = confusion_matrix(self.y_test, self.classifier.predict(self.X_test))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # save confusion matrix as plot image
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        if labels is not None:
            plt.xticks(np.arange(len(labels)), labels)
            plt.yticks(np.arange(len(labels)), labels)
        plt.savefig(config.result_file('confusion_matrix.png'))

        return cm
