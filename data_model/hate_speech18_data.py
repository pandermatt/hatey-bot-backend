from data_model.abstract_data import AbstractData
from datasets import load_dataset


class HateSpeech18Data(AbstractData):
    FILE_NAME = None
    FILE_URL = None

    def _load_file(self):
        return "hate_speech18"

    def _preprocess_data(self, file):
        data = load_dataset("hate_speech18")
        return data['train'].to_pandas()

    def get_data(self):
        return self.data['text']

    def get_label(self):
        return self.data['label']
