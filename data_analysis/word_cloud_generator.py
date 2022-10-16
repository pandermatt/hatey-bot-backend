import base64

from matplotlib import pyplot as plt
from wordcloud import WordCloud

from config import config


def generate(corpus, file_name):
    wordcloud = WordCloud(width=1600, height=800, background_color="white", collocations=False).generate(corpus)
    plt.figure(figsize=(20, 10))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(config.result_file(file_name), bbox_inches='tight')

    return base64.b64encode(open(config.result_file(file_name), 'rb').read()).decode('utf-8')
