from model.tokenizer import SpacyTokenizer, NltkTokenizer

sentence = ["This is a sentence. You are cute cute cute, seriously you so cute."]


def test_spacy():
    tokens = SpacyTokenizer().tokenize(sentence)[0]
    assert list(tokens) == ['sentence', 'cute', 'cute', 'cute', 'seriously', 'cute']


def test_nltk():
    tokens = NltkTokenizer(remove_repeated_ngrams=False).tokenize(sentence)[0]
    assert list(tokens) == ['this', 'sentence', 'you', 'cute', 'cute', 'cute', 'seriously', 'cute']


def test_remove_repeated_ngrams():
    tokens = NltkTokenizer(remove_repeated_ngrams=True).tokenize(sentence)[0]
    assert list(tokens) == ['this', 'sentence', 'you', 'cute', 'seriously', 'cute']
