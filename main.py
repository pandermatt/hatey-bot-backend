import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

from config import config
from data_analysis import word_cloud_generator
from data_model.dynamic_hate_data import DynamicHateData
from data_model.ethos_data import EthosData
from model.ensemble_classification import EnsembleClassification
from util.file_io import FileIo

if __name__ == '__main__':
    data = DynamicHateData().get_data()
    labels = DynamicHateData().get_label()

    all_text = ' '.join(data[labels == 0])
    word_cloud_generator.generate(all_text, 'dynamic_hate_data_non_hate.png')

    all_text = ' '.join(data[labels == 1])
    word_cloud_generator.generate(all_text, 'dynamic_hate_data_hate.png')

    model = EnsembleClassification()
    model.train(data, labels)

    FileIo.save_model('dynamic_hate_data_model', model)

    print(model.classification_report())
    print(model.confusion_matrix())

    loaded_model = FileIo.load_model('dynamic_hate_data_model')
    X = ['I hate you', 'I love you']
    Y = loaded_model.predict(X)
    print(Y)
