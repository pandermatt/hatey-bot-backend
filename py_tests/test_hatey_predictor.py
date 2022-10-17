import pytest

from core.hatey_predictor import HateyPredictor

hatey_predictor = HateyPredictor()


def test_is_hate_speech():
    assert hatey_predictor.is_hate_speech("You are a stupid")
    assert not hatey_predictor.is_hate_speech("You are a nice person")


def test_reasons():
    assert hatey_predictor.reasons("You are a stupid") == "Toxicity, Obscene, Insult"
    assert hatey_predictor.reasons("You are a nice person") == ""


def test_problematic_words():
    assert hatey_predictor.problematic_words("You are an idiot") == ['Idiot']
    assert hatey_predictor.problematic_words("You are a nice person") == []


def test_predictions():
    predictions = hatey_predictor.predictions("You are an idiot")
    assert predictions['Transformer']['Insult'] > 0.5
    assert predictions['Classifier']['Insult'] > 0.5
