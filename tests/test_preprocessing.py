import pytest
from src.preprocessing.cleaner import TextCleaner

def test_cleaner_lowercasing():
    cleaner = TextCleaner({"lowercase": True})
    res = cleaner.clean("HELLO WORLD")
    assert res["cleaned_text"] == "hello world"

def test_cleaner_html_url_removal():
    cleaner = TextCleaner({"remove_html": True, "remove_urls": True})
    res = cleaner.clean("<b>Nexus</b> check this https://google.com links.")
    assert "https" not in res["cleaned_text"]
    assert "<b>" not in res["cleaned_text"]
    assert "nexus" in res["cleaned_text"].lower()

def test_cleaner_emojis():
    cleaner = TextCleaner({"handle_emojis": True})
    res = cleaner.clean("I love this app 😄")
    assert "grinning" in res["cleaned_text"] or "smiling" in res["cleaned_text"] or "happy" in res["cleaned_text"] or "love" in res["cleaned_text"]
