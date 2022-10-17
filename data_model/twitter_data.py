from data_model.abstract_data import AbstractData


# Source: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset
class TwitterData(AbstractData):
    FILE_NAME = 'labeled_data.csv'

    def get_data(self):
        return self.data['tweet'].to_numpy()

    def get_label(self):
        return self.data['hate_speech'].to_numpy()
