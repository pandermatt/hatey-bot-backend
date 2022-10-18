import numpy as np

from data_analysis import word_cloud_generator
from data_analysis.ngram_plotter import NgramPlotter
from data_model.toxic_comment_data import ToxicCommentData
from model.tokenizer import NltkTokenizer

if __name__ == '__main__':
    dataset = ToxicCommentData()
    X = dataset.get_data()
    Y = dataset.get_label()
    label_names = dataset.get_label_names()

    classes = len(np.unique(Y))
    print(X.shape, Y.shape, classes)

    tokenizer = NltkTokenizer()
    X = tokenizer.tokenize(X)

    ngram_plotter = NgramPlotter(ngram_size=2)

    data_set_length = {}

    for i in range(classes):
        print(f'Label {label_names[i]}: {len(X[Y == i])}')
        data_set_length[label_names[i]] = len(X[Y == i])
        all_words = [text for subtext in X[Y == i] for text in subtext]
        word_cloud_generator.generate(' '.join(all_words), f'word_cloud_{i}_{label_names[i]}.pdf')
        ngram_plotter.plot_histogram(all_words, f'ngram_{i}_{label_names[i]}.pdf')

    data_set_length = {k: v for k, v in sorted(data_set_length.items(), key=lambda item: item[1])}
    import matplotlib.pyplot as plt
    plt.figure(figsize=(5, 4))
    plt.grid(axis='x', linestyle='--')
    plt.barh(range(len(data_set_length)), list(data_set_length.values()), align='center')
    plt.yticks(range(len(data_set_length)), list(data_set_length.keys()))
    for i, v in enumerate(data_set_length.values()):
        plt.text(v + 300, i, str(v), color='blue', fontweight='bold')
    plt.savefig('data_set_length.pdf', bbox_inches='tight')
    plt.show()


