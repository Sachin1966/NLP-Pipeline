import re
import logging
from transformers import pipeline
import spacy

logger = logging.getLogger(__name__)

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class SentimentAnalyzer:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english", device=-1):
        self.device = device
        self.model_name = model_name
        self.sentiment_pipe = None
        self.nlp = nlp
        
        # Heuristic Lexicons for fallback
        self.pos_words = {"love", "great", "excellent", "good", "amazing", "stunning", "perfect", "flawless", "satisfied", "satisfying", "beautiful", "fast", "helpful", "friendly", "best", "value"}
        self.neg_words = {"bad", "worst", "terrible", "disappointed", "disappointing", "slow", "buggy", "crash", "crashes", "overpriced", "useless", "laggy", "fail", "broken", "hate", "charge", "expensive"}
        
        import os
        low_mem_mode = os.getenv("LOW_MEMORY_MODE", "false").lower() == "true"
        
        from src.embeddings.embedder import check_huggingface_connectivity
        if low_mem_mode:
            logger.info("LOW_MEMORY_MODE is active. Skipping Hugging Face model and falling back to heuristic lexicon engine.")
            self.sentiment_pipe = None
        elif not check_huggingface_connectivity():
            logger.warning("Hugging Face is unreachable. SentimentAnalyzer immediately falling back to lexicon engine.")
            self.sentiment_pipe = None
        else:
            try:
                logger.info(f"Loading sentiment analysis pipeline: {model_name}")
                self.sentiment_pipe = pipeline("sentiment-analysis", model=model_name, device=device)
            except Exception as e:
                logger.warning(f"Failed to load Hugging Face sentiment model ({e}). Using local dictionary lexicon engine.")

    def analyze_text(self, text: str) -> dict:
        """
        Runs global sentiment analysis on the raw text.
        Returns label (POSITIVE, NEGATIVE, NEUTRAL) and confidence score.
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 1.0}
            
        if self.sentiment_pipe:
            try:
                res = self.sentiment_pipe(text[:512])[0] # Truncate to safety limits
                label = res["label"]
                score = float(res["score"])
                
                # Model labels might be POSITIVE/NEGATIVE or LABEL_0/LABEL_1
                if label in ["LABEL_1", "POSITIVE"]:
                    return {"label": "POSITIVE", "score": score}
                elif label in ["LABEL_0", "NEGATIVE"]:
                    # Distilbert is binary; return negative.
                    return {"label": "NEGATIVE", "score": score}
                else:
                    return {"label": label, "score": score}
            except Exception as e:
                logger.debug(f"HF Sentiment inference error ({e}). Using local fallback.")
                
        # Heuristic Fallback
        words = re.findall(r'\b\w+\b', text.lower())
        pos_count = sum(1 for w in words if w in self.pos_words)
        neg_count = sum(1 for w in words if w in self.neg_words)
        
        if pos_count > neg_count:
            score = 0.5 + (pos_count / (pos_count + neg_count + 1)) * 0.49
            return {"label": "POSITIVE", "score": round(score, 4)}
        elif neg_count > pos_count:
            score = 0.5 + (neg_count / (pos_count + neg_count + 1)) * 0.49
            return {"label": "NEGATIVE", "score": round(score, 4)}
        else:
            return {"label": "NEUTRAL", "score": 0.5}

    def analyze_aspects(self, text: str, allowed_aspects=None) -> list:
        """
        Splits text by clause and attributes sentiment scores to specific aspects.
        allowed_aspects defaults to: ["UI", "Battery", "Performance", "Pricing", "Support", "Security", "Features"]
        """
        allowed_aspects = allowed_aspects or ["UI", "Battery", "Performance", "Pricing", "Support", "Security", "Features"]
        allowed_aspects_lower = [a.lower() for a in allowed_aspects]
        
        # Clause segmenter: split on conjunctions & punctuation
        clauses = re.split(r'\.|\bbut\b|\bhowever\b|\band\b|;|,', text, flags=re.IGNORECASE)
        aspect_results = []
        
        # Map aspect synonyms to standard names
        aspect_mapping = {
            "ui": "UI", "interface": "UI", "layout": "UI", "screen": "UI", "display": "UI", "graphics": "UI", "buttons": "UI",
            "battery": "Battery", "charging": "Battery", "power": "Battery", "charger": "Battery",
            "performance": "Performance", "speed": "Performance", "lag": "Performance", "reliability": "Performance", "crash": "Performance", "crashed": "Performance", "laggy": "Performance", "slow": "Performance",
            "pricing": "Pricing", "cost": "Pricing", "price": "Pricing", "billing": "Pricing", "bill": "Pricing", "expensive": "Pricing", "cheap": "Pricing", "value": "Pricing",
            "support": "Support", "service": "Support", "help desk": "Support", "assistance": "Support", "agent": "Support",
            "security": "Security", "login": "Security", "sso": "Security", "encryption": "Security", "privacy": "Security", "auth": "Security",
            "features": "Features", "options": "Features", "dark mode": "Features", "integration": "Features", "widget": "Features"
        }
        
        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 5:
                continue
                
            # Parse clause with spaCy
            doc = self.nlp(clause)
            detected_aspects_in_clause = set()
            
            # Find noun chunks or single tokens matching synonyms
            for chunk in doc.noun_chunks:
                for token in chunk:
                    lemma = token.lemma_.lower()
                    if lemma in aspect_mapping:
                        detected_aspects_in_clause.add(aspect_mapping[lemma])
                        
            # Check individual tokens in case noun chunks misses it
            for token in doc:
                lemma = token.lemma_.lower()
                if lemma in aspect_mapping and aspect_mapping[lemma] not in detected_aspects_in_clause:
                    detected_aspects_in_clause.add(aspect_mapping[lemma])
            
            # If aspects are detected, run sentiment on this specific clause
            if detected_aspects_in_clause:
                clause_sent = self.analyze_text(clause)
                for asp in detected_aspects_in_clause:
                    if asp in allowed_aspects:
                        aspect_results.append({
                            "aspect": asp,
                            "sentiment": clause_sent["label"],
                            "score": clause_sent["score"],
                            "clause": clause
                        })
                        
        return aspect_results
