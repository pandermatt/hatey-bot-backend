from data_model.abstract_data import AbstractData


# Source: https://www.kaggle.com/datasets/tovarischsukhov/southparklines
class SouthParkData(AbstractData):
    FILE_NAME = 'All-seasons.csv'
    FILE_URL = None

    def _preprocess_data(self, file):
        data = super()._preprocess_data(file)
        data = data.groupby('Character').filter(lambda x: len(x) > 100)
        return data

    def get_data(self):
        return self.data['Line'].to_numpy()

    def get_label(self):
        label_names = self.get_label_names()
        return self.data['Character'].apply(lambda x: label_names.tolist().index(x)).to_numpy()

    def get_label_names(self):
        return self.data['Character'].unique()
