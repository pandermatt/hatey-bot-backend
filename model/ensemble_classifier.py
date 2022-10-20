import numpy as np
import seaborn as sns
import stringcase
from matplotlib import pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from config import config


class EnsembleClassifier:
    def __init__(self, classifier=ExtraTreesClassifier(), label_names=None):
        """
        :param classifier: classifier to use
        - ExtraTreesClassifier
        - BaggingClassifier
        - RandomForestClassifier
        - etc.
        """
        self.tfidf_vectorizer = TfidfVectorizer(tokenizer=lambda x: x, lowercase=False)
        self.classifier = classifier
        self.label_names = label_names

    def train(self, X, Y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y,
                                                                                test_size=0.2,
                                                                                random_state=42,
                                                                                stratify=Y)

        self.X_train = self.tfidf_vectorizer.fit_transform(self.X_train)
        self.X_test = self.tfidf_vectorizer.transform(self.X_test)

        self.classifier.fit(self.X_train, self.y_train)

    def predict(self, text):
        X = self.tfidf_vectorizer.transform(text)
        return self.classifier.predict(X)

    def predict_with_probability(self, text):
        X = self.tfidf_vectorizer.transform(text)
        return self.classifier.predict_proba(X)

    def predict_one_with_labels(self, text):
        proba = self.predict_with_probability(text)[0]
        return {label: proba[i] for i, label in enumerate(self.label_names)}

    def get_label_names(self):
        return self.label_names

    def classification_report(self):
        return classification_report(self.y_test, self.classifier.predict(self.X_test))

    def confusion_matrix(self, labels=None):
        cm = confusion_matrix(self.y_test, self.classifier.predict(self.X_test))
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels, cmap='Blues')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        class_name = self.classifier.__class__.__name__

        plt.title(f"Confusion Matrix for {stringcase.titlecase(class_name)}")
        plt.savefig(
            config.result_file(f'confusion_matrix_{stringcase.snakecase(class_name)}.pdf'),
            bbox_inches='tight'
        )

        return cm
