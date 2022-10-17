from data_model.toxic_comment_data import ToxicCommentData

dataset = ToxicCommentData()


def test_dataset():
    X = dataset.get_data()
    Y = dataset.get_label()

    assert len(X) == len(Y)

    classes = len(dataset.get_label_names())
    for i in range(len(X)):
        assert 0 <= Y[i] < classes
