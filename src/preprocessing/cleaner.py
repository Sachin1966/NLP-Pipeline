import re
import os
import yaml
import logging
import nltk
import emoji
import spacy

logger = logging.getLogger(__name__)

# Ensure NLTK packages are present
for package in ['stopwords', 'punkt']:
    try:
        nltk.data.find(f'corpora/{package}' if package == 'stopwords' else f'tokenizers/{package}')
    except LookupError:
        logger.info(f"Downloading NLTK package: {package}")
        nltk.download(package, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load SpaCy model, downloading it dynamically if not present
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.info("SpaCy model 'en_core_web_sm' not found. Downloading dynamically...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class TextCleaner:
    def __init__(self, config_dict=None):
        self.config = {
            "lowercase": True,
            "remove_stopwords": True,
            "lemmatize": True,
            "stem": False,
            "handle_emojis": True,
            "remove_urls": True,
            "remove_html": True,
            "remove_punctuation": True,
            "normalize_chars": True,
            "segment_sentences": True
        }
        
        # Load from config file if none provided
        if not config_dict:
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "pipeline_config.yaml")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        cfg = yaml.safe_load(f)
                        if cfg and 'preprocessing' in cfg:
                            self.config.update(cfg['preprocessing'])
                except Exception as e:
                    logger.warning(f"Failed to load preprocessing config: {e}")
        else:
            self.config.update(config_dict)
            
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception:
            self.stop_words = {"the", "a", "an", "and", "but", "if", "or", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both", "through", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very"}
            
        self.stemmer = PorterStemmer()
        self.nlp = nlp

    def clean(self, text: str) -> dict:
        """
        Runs text through the configured cleaning pipeline.
        Returns a dictionary containing the clean string, tokens, and sentences.
        """
        if not text or not isinstance(text, str):
            return {"cleaned_text": "", "tokens": [], "sentences": []}
            
        cleaned = text
        
        # 1. HTML tag removal
        if self.config.get("remove_html"):
            cleaned = re.sub(r"<[^>]*>", " ", cleaned)
            
        # 2. URL removal
        if self.config.get("remove_urls"):
            cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
            
        # 3. Emoji handling
        if self.config.get("handle_emojis"):
            # Translate emoji into readable text (e.g. 😄 -> :grinning_face_with_smiling_eyes:)
            # and replace the colons with space to integrate into text
            demojized = emoji.demojize(cleaned)
            cleaned = re.sub(r':([a-z_]+):', r' \1 ', demojized)
            
        # 4. Special character normalization (accents, curly quotes)
        if self.config.get("normalize_chars"):
            cleaned = cleaned.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
        # Let's perform sentence segmentation before we lowercase or remove punctuation
        sentences = []
        if self.config.get("segment_sentences"):
            doc = self.nlp(cleaned)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        else:
            sentences = [cleaned]
            
        # 5. Lowercasing
        if self.config.get("lowercase"):
            cleaned = cleaned.lower()
            
        # 6. Punctuation removal
        if self.config.get("remove_punctuation"):
            cleaned = re.sub(r'[^\w\s-]', ' ', cleaned)
            # Remove double spaces
            cleaned = re.sub(r'\s+', ' ', cleaned)
            
        # 7. Tokenization and advanced linguistic preprocessing (Lemmatization/Stemming/Stopwords)
        tokens = []
        doc = self.nlp(cleaned)
        
        for token in doc:
            token_text = token.text.strip()
            if not token_text:
                continue
                
            # Stopword removal
            if self.config.get("remove_stopwords") and token_text.lower() in self.stop_words:
                continue
                
            # Lemmatization or Stemming
            if self.config.get("lemmatize"):
                processed_token = token.lemma_
            elif self.config.get("stem"):
                processed_token = self.stemmer.stem(token_text)
            else:
                processed_token = token_text
                
            tokens.append(processed_token)
            
        cleaned_text = " ".join(tokens)
        
        return {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "tokens": tokens,
            "sentences": sentences
        }

if __name__ == "__main__":
    cleaner = TextCleaner()
    sample = "Hello World! Check out https://google.com <b>Great</b> stuff! 😄"
    print("Testing Preprocessing Cleaner:")
    print(cleaner.clean(sample))
