import numpy as np

from data_analysis import word_cloud_generator
from data_analysis.ngram_plotter import NgramPlotter
from data_model.south_park_data import SouthParkData
from model.tokenizer import NltkTokenizer

if __name__ == '__main__':
    dataset = SouthParkData()
    data = dataset.get_data()
    labels = dataset.get_label()
    label_names = dataset.get_label_names()

    X = data.to_numpy()
    Y = labels.to_numpy()

    classes = len(np.unique(Y))
    print(X.shape, Y.shape, classes)

    tokenizer = NltkTokenizer()
    X = tokenizer.tokenize(X)

    ngram_plotter = NgramPlotter(ngram_size=2)

    for i in range(classes):
        print(f'Label {i}: {len(X[Y == i])}')
        all_words = [text for subtext in X[Y == i] for text in subtext]
        word_cloud_generator.generate(' '.join(all_words), f'word_cloud_{i}_{label_names[i]}.pdf')
        ngram_plotter.plot_histogram(all_words, f'ngram_{i}_{label_names[i]}.pdf')
