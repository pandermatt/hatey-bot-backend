import os
from datetime import datetime

import pandas as pd
import stringcase

from config import config
from core.hatey_predictor import hatey_predictor_singleton
from model.tokenizer import NltkTokenizer, SpacyTokenizer
from util.file_io import FileIo
from util.output_writer import OutputWriter

if __name__ == '__main__':
    tokenizers = [NltkTokenizer(), SpacyTokenizer()]
    test_csv = pd.read_csv(config.input_file('toxic_comment_verify.csv'))
    ids = test_csv['id']

    for _, model in hatey_predictor_singleton.classifiers.items():
        print(model.classification_report())
        print(model.confusion_matrix(labels=model.get_label_names()))
        model_name = stringcase.snakecase(model.classifier.__class__.__name__)

        for tokenizer in tokenizers:
            tokenizer_name = stringcase.snakecase(tokenizer.__class__.__name__)
            cached_tokens = FileIo.cached(f'test_{tokenizer_name}_tokens.pkl', lambda x: tokenizer.tokenize(x))

            X = cached_tokens(test_csv['comment_text'].to_numpy())
            Y = model.predict_with_probability(X)

            writer = OutputWriter(f'submission_{model_name}.csv')
            writer.write(ids, Y)

            command = f'kaggle competitions submit -c jigsaw-toxic-comment-classification-challenge -f {writer.path} -m "{model_name} / {tokenizer_name} / {datetime.now()}"'
            print(command)
            os.system(command)
