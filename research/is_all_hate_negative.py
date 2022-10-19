from config import config
from data_model.toxic_comment_data import ToxicCommentData
from model.sentiment_analyser import SentimentAnalyser

if __name__ == '__main__':
    # is_all_hate_negative?
    dataset = ToxicCommentData()
    X = dataset.get_data()
    Y = dataset.get_label()
    label_names = dataset.get_label_names()

    analyser = SentimentAnalyser()
    positive_sentiment_counter = {}
    output_csv = open(config.result_file('is_all_hate_negative.csv'), 'w')
    for x, y in zip(X, Y):
        if y == 0:
            continue
        if y not in positive_sentiment_counter:
            positive_sentiment_counter[y] = 0

        sentiment = analyser.predict(x[:512])
        output_csv.write(f"{y},{sentiment['positive']},{sentiment['negative']},{sentiment['neutral']}\n")
        if sentiment['positive'] >= 0.8:
            positive_sentiment_counter[y] += 1
            print(f"'{x}' is '{label_names[y]}' and has positive sentiment of {sentiment['positive']}")
            print('-------------------------')

    output_csv.close()

    print(positive_sentiment_counter)
