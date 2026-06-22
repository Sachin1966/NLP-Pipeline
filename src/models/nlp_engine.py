import logging
import re
import spacy
from src.preprocessing.cleaner import TextCleaner
from src.models.sentiment import SentimentAnalyzer
from src.models.classification import MultiClassClassifier
from src.models.keywords import KeywordExtractor
from src.models.summarizer import TextSummarizer

logger = logging.getLogger(__name__)

# Load SpaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class NLPPipelineEngine:
    def __init__(self, device=-1):
        self.cleaner = TextCleaner()
        self.sentiment_analyzer = SentimentAnalyzer(device=device)
        self.classifier = MultiClassClassifier(device=device)
        self.keyword_extractor = KeywordExtractor()
        self.summarizer = TextSummarizer(device=device)
        self.nlp = nlp
        
        # Auxiliary NER vocabulary mapping
        self.competitor_keywords = {"apple", "samsung", "sony", "bose", "microsoft", "google", "aws", "azure", "slack", "zoom", "logitech", "hp", "dell", "lenovo"}
        self.tech_keywords = {"sso", "oauth", "api", "database", "sql", "cloud", "saas", "ai", "nlp", "llm", "docker", "kubernetes", "git"}
        self.company_keywords = {"nexus inc", "nova corp", "aero llc", "apex ltd", "zenith corp", "cloudhosting", "expressdelivery"}

    def extract_custom_entities(self, text: str) -> list:
        """
        Extracts Named Entities: Products, Companies, Locations, Dates, Technologies, Competitors.
        Uses a combination of SpaCy NER and dictionary-based heuristics.
        """
        doc = self.nlp(text)
        entities = []
        seen = set()
        
        # Standard SpaCy Entities mapping
        for ent in doc.ents:
            ent_text = ent.text.strip()
            ent_label = ent.label_
            
            # Map standard SpaCy labels to target entity categories
            mapped_label = None
            if ent_label == "GPE":
                mapped_label = "Location"
            elif ent_label == "DATE":
                mapped_label = "Date"
            elif ent_label == "ORG":
                # Determine if ORG is Competitor, Company or Technology
                ent_lower = ent_text.lower()
                if ent_lower in self.competitor_keywords:
                    mapped_label = "Competitor"
                elif ent_lower in self.company_keywords or "corp" in ent_lower or "inc" in ent_lower or "ltd" in ent_lower:
                    mapped_label = "Company"
                elif ent_lower in self.tech_keywords:
                    mapped_label = "Technology"
                else:
                    mapped_label = "Company"
            elif ent_label == "PRODUCT":
                mapped_label = "Product"
                
            if mapped_label and ent_text.lower() not in seen and len(ent_text) > 1:
                seen.add(ent_text.lower())
                entities.append({"text": ent_text, "label": mapped_label})
                
        # Regex and keyword dictionary check fallbacks (to catch entities missed by standard models)
        words = text.split()
        text_lower = text.lower()
        
        # Check Competitors
        for comp in self.competitor_keywords:
            # Match whole words only
            pattern = r'\b' + re.escape(comp) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for m in matches:
                start, end = m.span()
                matched_text = text[start:end]
                if matched_text.lower() not in seen:
                    seen.add(matched_text.lower())
                    entities.append({"text": matched_text, "label": "Competitor"})
                    
        # Check Technologies
        for tech in self.tech_keywords:
            pattern = r'\b' + re.escape(tech) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for m in matches:
                start, end = m.span()
                matched_text = text[start:end]
                if matched_text.lower() not in seen:
                    seen.add(matched_text.lower())
                    entities.append({"text": matched_text, "label": "Technology"})
                    
        # Check Company names
        for comp_name in self.company_keywords:
            pattern = r'\b' + re.escape(comp_name) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for m in matches:
                start, end = m.span()
                matched_text = text[start:end]
                if matched_text.lower() not in seen:
                    seen.add(matched_text.lower())
                    entities.append({"text": matched_text, "label": "Company"})
                    
        # Check Products from our synthetic vocabulary
        products_list = ["nexus laptop", "nova smartwatch", "aero earbuds", "apex keyboard", "zenith tablet"]
        for prod in products_list:
            pattern = r'\b' + re.escape(prod) + r'\b'
            matches = re.finditer(pattern, text_lower)
            for m in matches:
                start, end = m.span()
                matched_text = text[start:end]
                if matched_text.lower() not in seen:
                    seen.add(matched_text.lower())
                    entities.append({"text": matched_text, "label": "Product"})

        return entities

    def process_document(self, text: str, embedder=None, skip_summary=False) -> dict:
        """
        Orchestrates full NLP analysis pipeline for a document.
        """
        # 1. Clean Text
        cleaned_res = self.cleaner.clean(text)
        cleaned_text = cleaned_res["cleaned_text"]
        
        # 2. Global Sentiment
        sent_res = self.sentiment_analyzer.analyze_text(text)
        
        # 3. Aspect-Based Sentiment
        aspect_res = self.sentiment_analyzer.analyze_aspects(text)
        
        # 4. Named Entities
        entities_res = self.extract_custom_entities(text)
        
        # 5. Emotion
        emotion_res = self.classifier.detect_emotion(text)
        
        # 6. Toxicity
        toxic_res = self.classifier.detect_toxicity(text)
        
        # 7. Category Text Classification
        category_res = self.classifier.classify_category(text)
        
        # 8. Keywords
        if embedder:
            keywords_res = self.keyword_extractor.extract_keybert(text, embedder, top_n=5)
        else:
            keywords_res = self.keyword_extractor.extract_rake(text, top_n=5)
            
        # 9. Summary
        summary_res = "" if skip_summary else self.summarizer.summarize_abstractive(text)
        
        return {
            "cleaned_text": cleaned_text,
            "global_sentiment": sent_res["label"],
            "global_sentiment_score": sent_res["score"],
            "aspects": aspect_res,
            "entities": entities_res,
            "emotion": emotion_res,
            "toxicity_score": toxic_res["toxicity_score"],
            "is_toxic": toxic_res["is_toxic"],
            "category": category_res,
            "keywords": keywords_res,
            "summary": summary_res
        }
