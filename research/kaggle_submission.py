import os
from datetime import datetime

import numpy as np
import pandas as pd
import stringcase
from matplotlib import pyplot as plt

from config import config
from core.hatey_predictor import hatey_predictor_singleton
from model.tokenizer import NltkTokenizer, SpacyTokenizer
from util.file_io import FileIo
from util.output_writer import OutputWriter


def plot_results():
    labels = ['Gradient Boosting', 'Ada Boost', 'Random Forest', 'Extra Trees', 'Bagging', 'Transformer']
    # results from https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/submissions
    nltk = [0.69535, 0.63516, 0.79300, 0.79387, 0.83017, 0]
    spacy = [0.67112, 0.64179, 0.76685, 0.76247, 0.80164, 0]
    other = [0, 0, 0, 0, 0, 0.95]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(x - width / 2, nltk, width, label='NLTK')
    ax.bar(x + width / 2, spacy, width, label='SpaCy')
    ax.bar(x + width / 2, other, width, label='Embeddings')

    ax.set_ylabel('Score')
    ax.set_title('Scores by model and tokenizer')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0.5, 1.0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.tight_layout()

    plt.savefig(config.result_file('model_comparison.pdf'), bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    plot_results()

    # https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge
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
