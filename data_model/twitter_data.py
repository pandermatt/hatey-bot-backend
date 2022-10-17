import numpy as np

from data_model.abstract_data import AbstractData


# Source: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
class TwitterData(AbstractData):
    FILE_NAME = 'labeled_data.csv'
    FILE_URL = None

    def get_data(self):
        return self.data['tweet'].to_numpy()

    def get_label(self):
        return np.array([0 if label == 0 else 1 for label in self.data['hate_speech']])

    def get_label_names(self):
        return ['non-hate', 'hate']
