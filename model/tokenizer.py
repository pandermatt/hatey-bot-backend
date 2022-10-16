import numpy as np
import spacy
from nltk import PorterStemmer, WordNetLemmatizer, word_tokenize
from nltk.corpus import stopwords


class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def tokenize(self, texts):
        return [self._tokenize(text) for text in texts]

    def _tokenize(self, text):
        return [token.lemma_ for token in self.nlp(text) if not token.is_stop and not token.is_punct]



class NLTKTokenizer:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def tokenize(self, texts):
        return np.array([self._tokenize(text) for text in texts])

    def _tokenize(self, text):
        words = word_tokenize(text)
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        words = [self.lemmatizer.lemmatize(word).lower() for word in words]
        return words
