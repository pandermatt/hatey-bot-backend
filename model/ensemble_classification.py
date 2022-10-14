import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import config


class EnsembleClassification:
    def __init__(self):
        self.classifier = GradientBoostingClassifier()
        self.tfidf_vectorizer = TfidfVectorizer(analyzer='word',
                                                tokenizer=(lambda x: x),
                                                preprocessor=(lambda x: x))

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

    def confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.classifier.predict(self.X_test))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # save confusion matrix as plot image
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.savefig(config.result_file('confusion_matrix.png'))

        return cm
