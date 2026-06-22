import sys
import numpy as np
import unittest.mock as mock

# 1. Mock sentence_transformers SentenceTransformer class to avoid heavy weights downloads
class MockSentenceTransformer:
    def __init__(self, model_name=None, *args, **kwargs):
        self.model_name = model_name

    def encode(self, sentences, **kwargs):
        if isinstance(sentences, str):
            return np.random.randn(384)
        return np.random.randn(len(sentences), 384)

mock_st = mock.MagicMock()
mock_st.SentenceTransformer = MockSentenceTransformer
sys.modules['sentence_transformers'] = mock_st

# 2. Mock transformers pipeline function to intercept heavy pipelines
class MockPipeline:
    def __init__(self, task, model=None, device=None, *args, **kwargs):
        self.task = task
        self.model = model
        self.device = device

    def __call__(self, *args, **kwargs):
        if self.task == "sentiment-analysis":
            text = args[0] if args else kwargs.get("text", "")
            if any(w in text.lower() for w in ["wonderful", "love", "great", "excellent", "good", "stunning"]):
                return [{"label": "POSITIVE", "score": 0.95}]
            elif any(w in text.lower() for w in ["terrible", "bad", "worst", "disappointed"]):
                return [{"label": "NEGATIVE", "score": 0.95}]
            return [{"label": "POSITIVE", "score": 0.8}]
            
        elif self.task == "text-classification":
            text = args[0] if args else kwargs.get("text", "")
            # Check for emotion model (j-hartmann/emotion-english-distilroberta-base)
            if self.model == "j-hartmann/emotion-english-distilroberta-base" or "emotion" in str(self.model):
                if any(w in text.lower() for w in ["furious", "angry"]):
                    return [{"label": "anger", "score": 0.95}]
                return [{"label": "joy", "score": 0.95}]
            # Check for toxicity model (unitary/toxic-bert)
            elif self.model == "unitary/toxic-bert" or "toxic" in str(self.model):
                if any(w in text.lower() for w in ["stupid", "idiot", "garbage"]):
                    return [{"label": "toxic", "score": 0.95}]
                return [{"label": "severe_toxic", "score": 0.01}]
            return [{"label": "neutral", "score": 0.95}]
            
        elif self.task == "zero-shot-classification":
            text = args[0] if args else kwargs.get("text", "")
            candidate_labels = kwargs.get("candidate_labels", [])
            # Return Feature Request if text contains widget/add
            if any(w in text.lower() for w in ["feature", "add", "widget"]):
                labels = ["Feature Request"] + [l for l in candidate_labels if l != "Feature Request"]
                return {"labels": labels, "scores": [0.9] + [0.1 / (len(candidate_labels) - 1)] * (len(candidate_labels) - 1)}
            return {"labels": candidate_labels, "scores": [1.0 / len(candidate_labels)] * len(candidate_labels)}
            
        elif self.task == "summarization":
            return [{"summary_text": "This is a mocked summary of the customer review."}]
            
        return [{"label": "LABEL_0", "score": 0.5}]

mock_trans = mock.MagicMock()
mock_trans.pipeline = MockPipeline
sys.modules['transformers'] = mock_trans

# 3. Mock check_huggingface_connectivity to always return True during test execution
import src.embeddings.embedder
src.embeddings.embedder.check_huggingface_connectivity = lambda: True
