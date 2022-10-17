from core.hatey_predictor import hatey_predictor_singleton


def test_is_hate_speech():
    assert hatey_predictor_singleton.is_hate_speech("You are a stupid")
    assert not hatey_predictor_singleton.is_hate_speech("You are a nice person")


def test_reasons_as_text():
    assert hatey_predictor_singleton.reasons_as_text("You are a stupid") == "Toxicity, Obscene, Insult"
    assert hatey_predictor_singleton.reasons_as_text("You are a nice person") == ""


def test_reasons():
    assert hatey_predictor_singleton.reasons("You are a stupid") == ["Toxicity", "Obscene", "Insult"]
    assert hatey_predictor_singleton.reasons("You are a nice person") == []


def test_problematic_words():
    assert hatey_predictor_singleton.problematic_words("You are an idiot") == ['idiot']
    assert hatey_predictor_singleton.problematic_words("You are a nice person") == []


def test_predictions():
    predictions = hatey_predictor_singleton.predictions("You are an idiot")
    assert float(predictions['Transformer']['Insult']) > 0.5
    assert float(predictions['BaggingClassifier']['Insult']) > 0.5
    assert float(predictions['RandomForestClassifier']['Insult']) > 0.5
