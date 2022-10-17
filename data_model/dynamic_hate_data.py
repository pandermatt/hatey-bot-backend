import numpy as np
import pandas as pd

from data_model.abstract_data import AbstractData


class DynamicHateData(AbstractData):
    FILE_NAME = 'dymamic_hate_data.csv'
    FILE_URL = 'https://raw.githubusercontent.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset/main/Dynamically%20Generated%20Hate%20Dataset%20v0.2.3.csv'

    def get_data(self):
        return self.data['text'].to_numpy()

    def get_label(self):
        return np.array([1 if label == 'hate' else 0 for label in self.data['label']]).to_numpy()
