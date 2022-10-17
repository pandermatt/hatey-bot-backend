from data_model.abstract_data import AbstractData
from datasets import load_dataset


# Source: https://huggingface.co/datasets/hate_speech18
class HateSpeech18Data(AbstractData):
    FILE_NAME = None
    FILE_URL = None

    def _load_file(self):
        return "hate_speech18"

    def _preprocess_data(self, file):
        data = load_dataset("hate_speech18")
        return data['train'].to_pandas()

    def get_data(self):
        return self.data['text'].to_numpy()

    def get_label(self):
        return self.data['label'].to_numpy()

    def get_label_names(self):
        """
        0 - hate
        1 - noHate
        2 - relation (doesn't contain hate speech on their own, but combination of serveral sentences does)
        3 - idk/skip (not written in English or not sure)
        """
        return ['hate', 'noHate', 'relation', 'skip']
