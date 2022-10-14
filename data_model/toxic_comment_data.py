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
        data = data.sample(frac=1).groupby('label').head(10000)
        return data

    def get_data(self):
        return self.data['comment_text']

    # 0 - non-toxic
    # 1 - toxic
    # 2 - severe toxic
    # 3 - obscene
    # 4 - threat
    # 5 - insult
    # 6 - identity hate
    def get_label(self):
        return self.data['label']

    def get_label_names(self):
        return ['non-toxic', 'toxic', 'severe toxic', 'obscene', 'threat', 'insult', 'identity hate']
