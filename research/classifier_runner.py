import numpy as np
import pandas as pd

from config import config
from data_analysis import word_cloud_generator
from data_model.toxic_comment_data import ToxicCommentData
from model.ensemble_classification import EnsembleClassification
from model.tokenizer import NLTKTokenizer
from util.file_io import FileIo
from util.output_writer import OutputWriter

if __name__ == '__main__':
    dataset = ToxicCommentData()
    data = dataset.get_data()
    labels = dataset.get_label()
    label_names = dataset.get_label_names()

    X = data.to_numpy()
    Y = labels.to_numpy()

    tokenizer = NLTKTokenizer()
    cached_tokens = FileIo.cached('train_nltk_tokens.pkl', lambda x: tokenizer.tokenize(x))
    X = cached_tokens(X)

    for i in range(7):
        print(f'Label {i}: {len(X[Y == i])}')
    #     all_words = [text for subtext in X[Y == i] for text in subtext]
    #     word_cloud_generator.generate(' '.join(all_words), f'label_{label_names[i]}.png')

    model = EnsembleClassification()
    model.train(X, Y)

    FileIo.save_obj('model', model)
    FileIo.save_obj('tokenizer', tokenizer)

    # print(model.classification_report())
    # print(model.confusion_matrix(labels=label_names))

    loaded_model = FileIo.load_obj('model')
    loaded_tokenizer = FileIo.load_obj('tokenizer')
    cached_tokens = FileIo.cached('test_nltk_tokens.pkl', lambda x: loaded_tokenizer.tokenize(x))
    test_csv = pd.read_csv(config.input_file('toxic_comment_verify.csv'))

    X = cached_tokens(test_csv['comment_text'].to_numpy())
    # Y = loaded_model.predict_with_probability(X)
    Y = loaded_model.predict(X)
    print(Y)
    print(np.argmax(Y, axis=1))

    ids = test_csv['id']
    OutputWriter('submission.csv').write(ids, Y)
