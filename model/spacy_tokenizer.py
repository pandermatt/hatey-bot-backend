import spacy


class SpacyTokenizer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    def __call__(self, text):
        print('.', end='')
        return [token.lemma_ for token in self.nlp(text)
                if not token.is_stop and not token.is_punct and token.is_alpha]
