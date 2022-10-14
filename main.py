import numpy as np
import pandas as pd

from config import config
from data_analysis import word_cloud_generator
from data_model.toxic_comment_data import ToxicCommentData
from model.ensemble_classification import EnsembleClassification
from model.spacy_tokenizer import SpacyTokenizer
from util.file_io import FileIo
from util.output_writer import OutputWriter

if __name__ == '__main__':
    dataset = ToxicCommentData()
    data = dataset.get_data()
    labels = dataset.get_label()

    tokenizer = SpacyTokenizer()
    data = data.apply(tokenizer)

    for i in range(7):
        print(f'Label {i}: {len(labels[labels == i])}')
        all_words = ' '.join(data[labels == i].apply(lambda x: ' '.join(x)))
        word_cloud_generator.generate(all_words, f'label_{i}.png')

    model = EnsembleClassification()
    model.train(data, labels)

    FileIo.save_model('model', model)
    FileIo.save_model('tokenizer', tokenizer)

    print(model.classification_report())
    print(model.confusion_matrix())

    test_csv = pd.read_csv(config.input_file('toxic_comment_verify.csv'))
    X = test_csv['comment_text'].apply(tokenizer)
    ids = test_csv['id']

    loaded_model = FileIo.load_model('model')
    loaded_tokenizer = FileIo.load_model('tokenizer')
    Y = loaded_model.predict_with_probability(X)
    print(np.argmax(Y, axis=1))

    OutputWriter('submission.csv').write(ids, Y)
