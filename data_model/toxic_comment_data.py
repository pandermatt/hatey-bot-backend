from data_model.abstract_data import AbstractData


# Source: https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/overview/evaluation
class ToxicCommentData(AbstractData):
    FILE_NAME = 'toxic_comment.csv'
    FILE_URL = None

    def _preprocess_data(self, file):
        data = super()._preprocess_data(file)

        data['label'] = 0
        data.loc[data['toxic'] == 1, 'label'] = 1
        data.loc[data['severe_toxic'] == 1, 'label'] = 2
        data.loc[data['obscene'] == 1, 'label'] = 3
        data.loc[data['threat'] == 1, 'label'] = 4
        data.loc[data['insult'] == 1, 'label'] = 5
        data.loc[data['identity_hate'] == 1, 'label'] = 6
        data = data.groupby('label').head(10000)
        return data

    def get_data(self):
        return self.data['comment_text'].to_numpy()

    def get_label(self):
        return self.data['label'].to_numpy()

    def get_label_names(self):
        return ['non-toxic', 'toxicity', 'severe-toxicity', 'obscene', 'threat', 'insult', 'identity-attack']


class ToxicCommentBalancedData(ToxicCommentData):
    # generate this data set by running the oversampling.ipynb jupyter notebook
    FILE_NAME = 'toxic_comment_balanced.csv'
