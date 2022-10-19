from config import config
from data_model.toxic_comment_data import ToxicCommentData

if __name__ == '__main__':
    # is_all_hate_negative?
    dataset = ToxicCommentData()
    label_names = dataset.get_label_names()
    content = open(config.result_file('is_all_hate_negative.csv'), 'r').readlines()
    content = [line.strip().split(',') for line in content]
    content = [[int(line[0]), float(line[1]), float(line[2]), float(line[3])] for line in content]

    total = {}
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        positive_sentiment_counter = {}
        for line in content:
            y = line[0]

            if y not in positive_sentiment_counter:
                positive_sentiment_counter[y] = 0
                total[y] = 0

            total[y] += 1

            if line[1] >= threshold:
                positive_sentiment_counter[y] += 1

        print(f"Threshold: {threshold}")
        print('-------------------------')
        positive_sentiment_counter = {k: v for k, v in
                                      sorted(positive_sentiment_counter.items(), key=lambda item: item[1],
                                             reverse=True)}
        for key in positive_sentiment_counter:
            print(f"{label_names[key]}: {positive_sentiment_counter[key]} / {total[key]}")
        print('=========================')
