import nltk
import numpy as np
import spacy
from nltk import WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords


class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def tokenize(self, texts):
        return np.array([self._tokenize(text) for text in texts])

    def _tokenize(self, text):
        return [token.lemma_ for token in self.nlp(text) if not token.is_stop and not token.is_punct]


class NltkTokenizer:
    def __init__(self, remove_repeated_ngrams=True):
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('omw-1.4', quiet=True)

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.remove_repeated_ngrams = remove_repeated_ngrams

    def tokenize(self, texts):
        return np.array([self._tokenize(text) for text in texts])

    def _tokenize(self, text):
        words = word_tokenize(text)
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word).lower() for word in words]
        if self.remove_repeated_ngrams:
            words = [word for i, word in enumerate(words) if i < 2 or (word != words[i - 2] and word != words[i - 1])]

        return words