from core.hatey_predictor import hatey_predictor_singleton
from data_model.south_park_data import SouthParkData
from data_model.twitter_data import TwitterData

if __name__ == '__main__':
    datasets = [
        SouthParkData,
        TwitterData
    ]

    for dataset_class in datasets:
        hate_counter = {}
        print(f"Testing dataset: {dataset_class.__name__}")
        dataset = dataset_class()
        X = dataset.get_data()
        Y = dataset.get_label()
        names = dataset.get_label_names()

        for i in range(len(X)):
            if hatey_predictor_singleton.is_hate_speech(X[i]):
                if Y[i] not in hate_counter:
                    hate_counter[Y[i]] = 0
                hate_counter[Y[i]] += 1

        print(hate_counter)

        hate_by_percent = {
            names[key]: (hate_counter[key] / len(X[Y == key])) * 100 for key in hate_counter
        }
        hate_by_percent = {k: v for k, v in sorted(hate_by_percent.items(), key=lambda item: item[1], reverse=True)}
        print(hate_by_percent)
