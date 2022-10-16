from data_model.abstract_data import AbstractData


# Source: https://www.kaggle.com/datasets/tovarischsukhov/southparklines
class SouthParkData(AbstractData):
    FILE_NAME = 'All-seasons.csv'

    def get_data(self):
        return self.data['Line']

    def get_label(self):
        pass
