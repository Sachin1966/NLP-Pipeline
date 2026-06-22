import re
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class KeywordExtractor:
    def __init__(self):
        self.stop_words = {"the", "a", "an", "and", "but", "if", "or", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "than", "such", "both", "through", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "i", "you", "he", "she", "it", "we", "they", "my", "your", "his", "her", "its", "our", "their"}

    def extract_tfidf(self, texts: list, top_n: int = 5) -> list:
        """
        Extracts top keywords from a list of texts using TF-IDF.
        """
        if not texts:
            return []
        try:
            tfidf = TfidfVectorizer(stop_words='english', max_features=100)
            tfidf_matrix = tfidf.fit_transform(texts)
            feature_names = tfidf.get_feature_names_out()
            
            # Sum tf-idf scores for each word across all docs
            scores = tfidf_matrix.sum(axis=0).A1
            word_scores = list(zip(feature_names, scores))
            word_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            return [w for w, s in word_scores[:top_n]]
        except Exception:
            # Fallback to word counts
            words = []
            for t in texts:
                words.extend(re.findall(r'\b\w{4,}\b', t.lower()))
            words = [w for w in words if w not in self.stop_words]
            counts = collections.Counter(words)
            return [w for w, c in counts.most_common(top_n)]

    def extract_rake(self, text: str, top_n: int = 5) -> list:
        """
        Pure Python lightweight RAKE implementation.
        """
        if not text or not text.strip():
            return []
            
        # 1. Split text into candidate phrases by punctuation and stop words
        # Create stopword regex pattern
        stop_pattern = r'\b(' + '|'.join(map(re.escape, self.stop_words)) + r')\b'
        
        # Split by punctuation
        clauses = re.split(r'[,.?!;:\-\"\(\)\n]', text.lower())
        
        candidates = []
        for clause in clauses:
            # Split clause by stop words
            phrases = re.split(stop_pattern, clause)
            for p in phrases:
                p_clean = re.sub(r'\s+', ' ', p).strip()
                if p_clean and len(p_clean) > 2:
                    candidates.append(p_clean)
                    
        if not candidates:
            return []
            
        # 2. Compute word frequencies and word degrees
        word_freq = collections.defaultdict(int)
        word_degree = collections.defaultdict(int)
        
        for candidate in candidates:
            words = candidate.split()
            degree = len(words) - 1
            for word in words:
                word_freq[word] += 1
                word_degree[word] += degree
                
        # Calculate word scores = degree / frequency
        word_score = {}
        for word in word_freq:
            # degree = total co-occurring words count + frequency
            word_score[word] = (word_degree[word] + word_freq[word]) / float(word_freq[word])
            
        # 3. Calculate candidate phrase scores
        candidate_scores = {}
        for candidate in set(candidates):
            words = candidate.split()
            candidate_scores[candidate] = sum(word_score[w] for w in words)
            
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, score in sorted_candidates[:top_n]]

    def extract_keybert(self, text: str, embedder, top_n: int = 5) -> list:
        """
        KeyBERT-style keyword extraction.
        1. Extract candidate words (unigrams/bigrams).
        2. Embed document and candidate words.
        3. Rank candidate words based on cosine similarity to document.
        """
        if not text or not text.strip():
            return []
            
        # Extract candidates
        candidates = re.findall(r'\b\w{4,}\b', text.lower())
        candidates = list(set([w for w in candidates if w not in self.stop_words]))
        
        if not candidates:
            return self.extract_rake(text, top_n)
            
        try:
            # Embed doc
            doc_emb = embedder.embed_text(text)
            # Embed candidates
            cand_embs = embedder.embed_batch(candidates)
            
            # Cosine similarities
            # doc_emb shape: (D,), cand_embs shape: (C, D)
            doc_norm = doc_emb / np.linalg.norm(doc_emb)
            cand_norms = cand_embs / np.linalg.norm(cand_embs, axis=1, keepdims=True)
            
            similarities = np.dot(cand_norms, doc_norm)
            
            cand_sims = list(zip(candidates, similarities))
            cand_sims = sorted(cand_sims, key=lambda x: x[1], reverse=True)
            return [cand for cand, sim in cand_sims[:top_n]]
        except Exception:
            return self.extract_rake(text, top_n)
