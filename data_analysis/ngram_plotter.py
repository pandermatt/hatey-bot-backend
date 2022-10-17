import nltk
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from config import config


class NgramPlotter:
    def __init__(self, ngram_size=2, top_n=30):
        self.ngram_size = ngram_size
        self.top_n = top_n

    def _get_ngrams(self, tokens):
        ngrams = nltk.ngrams(tokens, self.ngram_size)
        return [' '.join(grams) for grams in ngrams]

    def plot_histogram(self, text, filename):
        ngrams = self._get_ngrams(text)
        all_fdist = nltk.FreqDist(ngrams).most_common(self.top_n)
        all_fdist = pd.Series(dict(all_fdist))
        fig, ax = plt.subplots(figsize=(10, 10))
        all_plot = sns.barplot(x=all_fdist.index, y=all_fdist.values, ax=ax)
        plt.xticks(rotation=90)
        # only use integer for y axis
        ax.get_yaxis().set_major_locator(plt.MaxNLocator(integer=True))
        fig.savefig(config.result_file(filename), bbox_inches='tight')
