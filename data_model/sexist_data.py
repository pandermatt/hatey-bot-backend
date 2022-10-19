from data_model.abstract_data import AbstractData


# Source: https://www.kaggle.com/datasets/dgrosz/sexist-workplace-statements
class SexistData(AbstractData):
    FILE_NAME = 'ISEP_sexist_data_labeling.csv'
    FILE_URL = None

    def get_data(self):
        return self.data['Sentences'].to_numpy()

    def get_label(self):
        return self.data['Label'].to_numpy()

    def get_label_names(self):
        return ['non-sexism', 'sexism']

