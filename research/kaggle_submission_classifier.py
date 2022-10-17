import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier

from config import config
from core.model_trainer import generate_and_train_model
from data_model.toxic_comment_data import ToxicCommentData
from model.tokenizer import NLTKTokenizer
from util.file_io import FileIo
from util.output_writer import OutputWriter

if __name__ == '__main__':
    bagging_classifier = BaggingClassifier(base_estimator=ExtraTreesClassifier(), n_estimators=10, max_samples=0.5,
                                           max_features=0.5)
    model, tokenizer = generate_and_train_model(dataset=ToxicCommentData(),
                                                tokenizer=NLTKTokenizer(),
                                                base_classifier=bagging_classifier)

    print(model.classification_report())
    print(model.confusion_matrix(labels=model.get_label_names()))

    cached_tokens = FileIo.cached('test_nltk_tokens.pkl', lambda x: tokenizer.tokenize(x))
    test_csv = pd.read_csv(config.input_file('toxic_comment_verify.csv'))

    X = cached_tokens(test_csv['comment_text'].to_numpy())
    Y = model.predict_with_probability(X)
    print(np.argmax(Y, axis=1))

    ids = test_csv['id']
    OutputWriter('submission.csv').write(ids, Y)
