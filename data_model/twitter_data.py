from data_model.abstract_data import AbstractData


class TwitterData(AbstractData):
    FILE_NAME = 'labeled_data.csv'

    def get_data(self):
        return self.data['tweet']

    def get_label(self):
        return self.data['hate_speech']
