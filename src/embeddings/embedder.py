import os
import hashlib
import pickle
import sqlite3
import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

import threading

_HF_CONNECTED = None
_HF_CONNECTED_RESULT = None

def _check_url():
    import urllib.request
    endpoint = os.getenv("HF_ENDPOINT", "https://huggingface.co")
    try:
        urllib.request.urlopen(endpoint, timeout=1.0)
        global _HF_CONNECTED_RESULT
        _HF_CONNECTED_RESULT = True
    except Exception:
        pass

def check_huggingface_connectivity() -> bool:
    global _HF_CONNECTED, _HF_CONNECTED_RESULT
    if _HF_CONNECTED is not None:
        return _HF_CONNECTED
        
    if _HF_CONNECTED_RESULT is True:
        _HF_CONNECTED = True
        return True
        
    t = threading.Thread(target=_check_url)
    t.daemon = True
    t.start()
    t.join(timeout=3.0)
    
    if _HF_CONNECTED_RESULT is True:
        _HF_CONNECTED = True
        return True
        
    _HF_CONNECTED = False
    return False

class SentenceEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", cache_path="data/embeddings_cache.db"):
        self.model_name = model_name
        self.cache_path = cache_path
        
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        self.init_cache_db()
        
        self.is_offline = not check_huggingface_connectivity()
        if self.is_offline:
            logger.warning("Hugging Face is unreachable. Using offline/local fallback embedding generator.")
            self.model = None
        else:
            try:
                logger.info(f"Loading embedding model: {model_name}...")
                self.model = SentenceTransformer(model_name)
                logger.info("Embedding model loaded successfully.")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer ({e}). Using offline/local fallback.")
                self.model = None

    def _generate_fallback_embedding(self, text: str) -> np.ndarray:
        # Generate a deterministic pseudo-random unit vector based on text content hash
        import hashlib
        h = int(hashlib.md5(text.encode('utf-8')).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(h)
        vec = rng.standard_normal(384)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def init_cache_db(self):
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            text_hash TEXT PRIMARY KEY,
            embedding BLOB NOT NULL
        )
        """)
        conn.commit()
        conn.close()

    def _get_hash(self, text: str) -> str:
        return hashlib.md5(text.strip().lower().encode('utf-8')).hexdigest()

    def get_cached_embedding(self, text: str) -> np.ndarray:
        """Retrieves embedding from SQLite cache if exists."""
        text_hash = self._get_hash(text)
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        cursor.execute("SELECT embedding FROM embeddings WHERE text_hash = ?", (text_hash,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return pickle.loads(row[0])
        return None

    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Saves embedding to SQLite cache."""
        text_hash = self._get_hash(text)
        blob = pickle.dumps(embedding)
        conn = sqlite3.connect(self.cache_path)
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT OR REPLACE INTO embeddings (text_hash, embedding) VALUES (?, ?)", (text_hash, blob))
            conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
        finally:
            conn.close()

    def embed_text(self, text: str) -> np.ndarray:
        """Generates embedding for a single text string."""
        if not text or not text.strip():
            return np.zeros(384) # Default size of MiniLM-L6-v2 is 384
            
        cached = self.get_cached_embedding(text)
        if cached is not None:
            return cached
            
        if self.model is None:
            emb = self._generate_fallback_embedding(text)
        else:
            emb = self.model.encode(text)
        self.cache_embedding(text, emb)
        return emb

    def embed_batch(self, texts: list) -> np.ndarray:
        """Generates embeddings for a batch of text strings, utilizing cache when available."""
        if not texts:
            return np.array([])
            
        embeddings = []
        uncached_indices = []
        uncached_texts = []
        
        # Check cache
        for idx, t in enumerate(texts):
            cached = self.get_cached_embedding(t)
            if cached is not None:
                embeddings.append((idx, cached))
            else:
                uncached_indices.append(idx)
                uncached_texts.append(t)
                
        # Generate for uncached
        if uncached_texts:
            logger.info(f"Generating embeddings for {len(uncached_texts)} uncached texts...")
            if self.model is None:
                new_embs = np.array([self._generate_fallback_embedding(t) for t in uncached_texts])
            else:
                new_embs = self.model.encode(uncached_texts)
            for text, emb, idx in zip(uncached_texts, new_embs, uncached_indices):
                self.cache_embedding(text, emb)
                embeddings.append((idx, emb))
                
        # Reassemble in original order
        embeddings = sorted(embeddings, key=lambda x: x[0])
        return np.array([emb for idx, emb in embeddings])
