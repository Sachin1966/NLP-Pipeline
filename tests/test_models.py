import pytest
from src.models.sentiment import SentimentAnalyzer
from src.models.classification import MultiClassClassifier
from src.models.keywords import KeywordExtractor
from src.models.summarizer import TextSummarizer

# Use CPU-only devices for testing speed
@pytest.fixture(scope="module")
def sentiment_analyzer():
    return SentimentAnalyzer()

@pytest.fixture(scope="module")
def classifier():
    return MultiClassClassifier()

@pytest.fixture(scope="module")
def extractor():
    return KeywordExtractor()

@pytest.fixture(scope="module")
def summarizer():
    return TextSummarizer()

def test_sentiment_score(sentiment_analyzer):
    res = sentiment_analyzer.analyze_text("This keyboard is wonderful and typing is super quiet.")
    assert res["label"] == "POSITIVE"
    assert res["score"] > 0.5

def test_aspects_sentiment(sentiment_analyzer):
    # Splits on 'but'
    text = "The screen resolution is stunning, but the battery life is terrible."
    ress = sentiment_analyzer.analyze_aspects(text)
    
    aspects = {r["aspect"]: r["sentiment"] for r in ress}
    assert "UI" in aspects
    assert "Battery" in aspects
    assert aspects["UI"] == "POSITIVE"
    assert aspects["Battery"] == "NEGATIVE"

def test_classifier_emotion(classifier):
    res = classifier.detect_emotion("I am so furious and angry at this double charge!")
    assert res == "Angry"

def test_classifier_toxicity(classifier):
    res = classifier.detect_toxicity("You are a stupid idiot and your software is garbage.")
    assert res["is_toxic"] is True
    assert res["toxicity_score"] > 0.5

def test_classifier_category(classifier):
    res = classifier.classify_category("Can you please add a dark mode toggler and mobile widget to the dashboard?")
    assert res == "Feature Request"

def test_keywords_rake(extractor):
    text = "Machine learning and natural language processing pipelines are changing software services."
    res = extractor.extract_rake(text, top_n=3)
    assert len(res) > 0

def test_summarizer_extractive(summarizer):
    text = (
        "The laptop was delivered yesterday. The design is sleek and lightweight. "
        "The screen resolution is bright. However, the battery only lasts 2 hours. "
        "Customer service offered a full refund."
    )
    res = summarizer.summarize_extractive(text, num_sentences=2)
    assert len(res.split(".")) >= 2
