import pandas as pd

from config import config
from data_model.abstract_data import AbstractData


class EthosData(AbstractData):
    FILE_NAME = 'Ethos_Dataset_Binary.csv'
    FILE_URL = 'https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv'

    def _preprocess_data(self, file):
        return pd.read_csv(file, sep=';')

    def get_data(self):
        return self.data['comment']

    def get_label(self):
        return self.data['isHate']
