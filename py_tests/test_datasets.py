from data_model.dynamic_hate_data import DynamicHateData
from data_model.ethos_data import EthosData
from data_model.hate_speech18_data import HateSpeech18Data
from data_model.south_park_data import SouthParkData
from data_model.toxic_comment_data import ToxicCommentData
from data_model.twitter_data import TwitterData

datasets = [
    DynamicHateData,
    EthosData,
    HateSpeech18Data,
    SouthParkData,
    ToxicCommentData,
    TwitterData
]


def test_dataset():
    for dataset_class in datasets:
        print(f"Testing dataset: {dataset_class.__name__}")
        dataset = dataset_class()
        X = dataset.get_data()
        Y = dataset.get_label()

        assert len(X) == len(Y)

        classes = len(dataset.get_label_names())
        assert classes == len(set(Y))
        for i in range(len(X)):
            assert 0 <= Y[i] < classes
