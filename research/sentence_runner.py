import numpy as np

from data_model.toxic_comment_data import ToxicCommentData
from model.toxicity_predictor_transformer import ToxicityPredictorTransformer
from util.file_io import FileIo

if __name__ == '__main__':
    loaded_model = FileIo.load_obj('model')
    loaded_tokenizer = FileIo.load_obj('tokenizer')
    label_names = ToxicCommentData().get_label_names()

    sentences = ["I am a student at the University of Toronto",
                 "That bitch stupid af",
                 "I love the islamic culture",
                 "i would not mind to enjoy his physical assets although he doesn't have much else to offer.",
                 "all niggas are ugly",
                 "please dont talk to me, your breath smells very displeasing. brush your teeth!"]
    X = loaded_tokenizer.tokenize(sentences)
    Y = loaded_model.predict_with_probability(X)
    print(np.argmax(Y, axis=1))

    transformer = ToxicityPredictorTransformer()

    for text, label in zip(sentences, Y):
        label_name = label_names[np.argmax(label)]
        print(f"{text} -> {label_name}")
        print(transformer.predict(text))
