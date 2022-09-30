import pandas as pd

from config import config
from data_analysis import word_cloud_generator
from data_model.south_park_data import SouthParkData
from data_model.twitter_data import TwitterData

if __name__ == '__main__':
    data = SouthParkData().get_data()
    all_text = ' '.join(data)
    word_cloud_generator.generate(all_text, 'south_park_word_cloud.png')

    data = TwitterData().get_data()
    labels = TwitterData().get_label()

    all_text = ' '.join(data[labels == 0])
    word_cloud_generator.generate(all_text, 'twitter_word_cloud_label_0.png')

    all_text = ' '.join(data[labels == 1])
    print(word_cloud_generator.generate(all_text, 'twitter_word_cloud_label_1.png'))

