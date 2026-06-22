import re
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class MultiClassClassifier:
    def __init__(self, device=-1):
        self.device = device
        self.emotion_pipe = None
        self.toxic_pipe = None
        self.zeroshot_pipe = None
        
        import os
        low_mem_mode = os.getenv("LOW_MEMORY_MODE", "false").lower() == "true"
        
        from src.embeddings.embedder import check_huggingface_connectivity
        is_online = check_huggingface_connectivity() if not low_mem_mode else False
        
        # 1. Load Emotion Model
        if low_mem_mode:
            logger.info("LOW_MEMORY_MODE is active. Skipping Emotion detection model and falling back to heuristic.")
            self.emotion_pipe = None
        elif is_online:
            try:
                logger.info("Loading Emotion detection pipeline (distilroberta)...")
                self.emotion_pipe = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", device=device)
            except Exception as e:
                logger.warning(f"Failed to load Emotion model ({e}). Using local heuristic.")
        else:
            logger.warning("Hugging Face is unreachable. Emotion Classifier immediately falling back to local heuristics.")
            
        # 2. Load Toxicity Model
        if low_mem_mode:
            logger.info("LOW_MEMORY_MODE is active. Skipping Toxicity detection model and falling back to heuristic.")
            self.toxic_pipe = None
        elif is_online:
            try:
                logger.info("Loading Toxicity detection pipeline (toxic-bert)...")
                self.toxic_pipe = pipeline("text-classification", model="unitary/toxic-bert", device=device)
            except Exception as e:
                logger.warning(f"Failed to load Toxicity model ({e}). Using local heuristic.")
        else:
            logger.warning("Hugging Face is unreachable. Toxicity Classifier immediately falling back to local heuristics.")
            
        # 3. Load Zero-shot Category Model
        if low_mem_mode:
            logger.info("LOW_MEMORY_MODE is active. Skipping Zero-Shot text classification model and falling back to heuristic.")
            self.zeroshot_pipe = None
        elif is_online:
            try:
                logger.info("Loading Zero-Shot text classification pipeline (distilbert)...")
                self.zeroshot_pipe = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli", device=device)
            except Exception as e:
                logger.warning(f"Failed to load Zero-shot model ({e}). Using local heuristic.")
        else:
            logger.warning("Hugging Face is unreachable. Zero-Shot Classifier immediately falling back to local heuristics.")

        # Keywords for heuristics fallbacks
        self.emotion_lexicon = {
            "happy": {"happy", "love", "joy", "glad", "stunning", "perfect", "thrilled", "great", "excellent", "pleased"},
            "excited": {"excited", "awesome", "incredible", "cannot wait", "amazing", "wow", "eager", "fantastic"},
            "angry": {"angry", "mad", "furious", "pissed", "hate", "rage", "terrible", "worst", "screwed"},
            "frustrated": {"frustrated", "annoying", "annoyed", "useless", "laggy", "crashes", "slow", "waste", "buggy", "fix"},
            "disappointed": {"disappointed", "sad", "unfortunate", "letdown", "overpriced", "poor", "expected better"},
        }
        
        self.toxic_lexicon = {"hate", "suck", "sucks", "stupid", "idiot", "kill", "abuse", "garbage", "trash", "crap", "worst", "hell", "jerk", "moron"}
        
        self.category_lexicon = {
            "Billing Issue": {"billing", "invoice", "charge", "refund", "price", "overcharge", "subscription", "cost", "payment", "card"},
            "Feature Request": {"request", "add", "feature", "widget", "integration", "enhancement", "option", "theme", "custom"},
            "Bug Report": {"bug", "crash", "crashed", "crashes", "error", "fails", "glitch", "broken", "failing", "exception", "freeze"},
            "Complaint": {"terrible", "slow", "worst", "unhelpful", "poor", "frustrated", "waste", "refund", "bad", "disappointed"},
            "Appreciation": {"love", "thanks", "thank you", "great", "excellent", "awesome", "stunning", "perfect", "good", "helpful"},
            "Technical Support": {"sso", "login", "server", "credentials", "password", "reset", "oauth", "access", "config", "setup"},
            "Account Issue": {"account", "sso", "profile", "access", "register", "login", "username", "password", "deactivate"}
        }

    def detect_emotion(self, text: str) -> str:
        """
        Detects emotion: Happy, Angry, Frustrated, Excited, Neutral, Disappointed.
        """
        if not text or not text.strip():
            return "Neutral"
            
        if self.emotion_pipe:
            try:
                # Map model labels to standard capitalizations
                res = self.emotion_pipe(text[:512])[0]
                label = res["label"].lower()
                mapping = {
                    "joy": "Happy",
                    "sadness": "Disappointed",
                    "anger": "Angry",
                    "fear": "Frustrated",
                    "surprise": "Excited",
                    "love": "Happy",
                    "neutral": "Neutral"
                }
                return mapping.get(label, label.capitalize())
            except Exception:
                pass
                
        # Heuristic Fallback
        text_lower = text.lower()
        scores = {emo: 0 for emo in self.emotion_lexicon}
        for emo, words in self.emotion_lexicon.items():
            for w in words:
                if w in text_lower:
                    scores[emo] += 1
                    
        max_emo = max(scores, key=scores.get)
        if scores[max_emo] > 0:
            # Map keys to match prompt spec
            mapping = {
                "happy": "Happy",
                "excited": "Excited",
                "angry": "Angry",
                "frustrated": "Frustrated",
                "disappointed": "Disappointed"
            }
            return mapping[max_emo]
        return "Neutral"

    def detect_toxicity(self, text: str) -> dict:
        """
        Detects toxicity scores.
        Returns: {"toxicity_score": float, "is_toxic": bool}
        """
        if not text or not text.strip():
            return {"toxicity_score": 0.0, "is_toxic": False}
            
        if self.toxic_pipe:
            try:
                res = self.toxic_pipe(text[:512])[0]
                # toxic-bert usually outputs toxicity score
                score = float(res["score"])
                is_toxic = score > 0.5 if res["label"] == "toxic" else score < 0.5
                if res["label"] != "toxic":
                    score = 1.0 - score
                return {"toxicity_score": score, "is_toxic": score > 0.5}
            except Exception:
                pass
                
        # Heuristic Fallback
        text_lower = text.lower()
        toxic_count = sum(1 for w in self.toxic_lexicon if w in text_lower)
        score = min(0.95, (toxic_count / 3.0)) if toxic_count > 0 else 0.05
        return {"toxicity_score": score, "is_toxic": score > 0.5}

    def classify_category(self, text: str, candidate_labels=None) -> str:
        """
        Classifies feedback into categories: Billing Issue, Feature Request, Bug Report, etc.
        """
        default_labels = candidate_labels or [
            "Billing Issue", "Feature Request", "Bug Report", "Complaint", 
            "Appreciation", "Technical Support", "Account Issue"
        ]
        
        if not text or not text.strip():
            return default_labels[0]
            
        if self.zeroshot_pipe:
            try:
                res = self.zeroshot_pipe(text[:512], candidate_labels=default_labels)
                return res["labels"][0]
            except Exception:
                pass
                
        # Heuristic Fallback
        text_lower = text.lower()
        scores = {cat: 0 for cat in default_labels}
        for cat in default_labels:
            if cat in self.category_lexicon:
                for keyword in self.category_lexicon[cat]:
                    if keyword in text_lower:
                        scores[cat] += 1
                        
        max_cat = max(scores, key=scores.get)
        if scores[max_cat] > 0:
            return max_cat
            
        # Default fallback logic based on overall negative vs positive words
        if "charge" in text_lower or "bill" in text_lower or "invoice" in text_lower:
            return "Billing Issue"
        elif "request" in text_lower or "could you" in text_lower or "can we" in text_lower:
            return "Feature Request"
        elif "bug" in text_lower or "crash" in text_lower or "freeze" in text_lower:
            return "Bug Report"
        elif "support" in text_lower or "help" in text_lower or "reset" in text_lower:
            return "Technical Support"
        
        return "Complaint" if any(w in text_lower for w in ["bad", "slow", "worst"]) else "Appreciation"
