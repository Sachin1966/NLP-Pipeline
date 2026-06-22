import re
import collections
import logging
from transformers import pipeline

logger = logging.getLogger(__name__)

class TextSummarizer:
    def __init__(self, model_name="sshleifer/distilbart-cnn-12-6", device=-1):
        self.device = device
        self.model_name = model_name
        self.summarizer_pipe = None
        
        import os
        low_mem_mode = os.getenv("LOW_MEMORY_MODE", "false").lower() == "true"
        
        from src.embeddings.embedder import check_huggingface_connectivity
        if low_mem_mode:
            logger.info("LOW_MEMORY_MODE is active. Skipping Abstractive Summarization model and falling back to Extractive TextRank summaries.")
            self.summarizer_pipe = None
        elif not check_huggingface_connectivity():
            logger.warning("Hugging Face is unreachable. TextSummarizer immediately falling back to Extractive TextRank summaries.")
            self.summarizer_pipe = None
        else:
            try:
                logger.info(f"Loading Abstractive Summarization pipeline: {model_name}")
                self.summarizer_pipe = pipeline("summarization", model=model_name, device=device)
            except Exception as e:
                logger.warning(f"Failed to load HF summarizer ({e}). Abstractive requests will fall back to Extractive TextRank summaries.")

    def summarize_extractive(self, text: str, num_sentences: int = 3) -> str:
        """
        Summarizes text by scoring and extracting the most important sentences.
        Uses sentence-level frequency overlap.
        """
        if not text or not text.strip():
            return ""
            
        # Split text into sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if len(sentences) <= num_sentences:
            return text
            
        # Compute word frequencies across the whole text
        words = re.findall(r'\b\w+\b', text.lower())
        word_freq = collections.Counter(words)
        
        # Max frequency for normalization
        max_freq = max(word_freq.values()) if word_freq else 1
        word_scores = {w: count / max_freq for w, count in word_freq.items()}
        
        # Score sentences based on word scores
        sentence_scores = {}
        for idx, sent in enumerate(sentences):
            sent_words = re.findall(r'\b\w+\b', sent.lower())
            score = sum(word_scores.get(w, 0) for w in sent_words)
            # Normalize by sentence length to avoid bias towards long sentences
            sentence_scores[idx] = score / (len(sent_words) + 1)
            
        # Get top sentences sorting by score
        top_indices = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:num_sentences]
        # Sort indices chronologically to preserve flow
        top_indices = sorted(top_indices)
        
        summary = " ".join([sentences[i] for i in top_indices])
        return summary

    def summarize_abstractive(self, text: str, min_length: int = 10, max_length: int = 80) -> str:
        """
        Abstractive summarization using Hugging Face pipelines with extractive fallback.
        """
        if not text or len(text.strip()) < 50:
            return text
            
        if self.summarizer_pipe:
            try:
                # Truncate input text to 1024 tokens (rough safety limit)
                truncated_text = text[:4096]
                res = self.summarizer_pipe(
                    truncated_text,
                    min_length=min_length,
                    max_length=max_length,
                    do_sample=False
                )
                return res[0]["summary_text"].strip()
            except Exception as e:
                logger.debug(f"HF Summarizer inference error ({e}). Using extractive fallback.")
                
        # Fallback to extractive
        return self.summarize_extractive(text, num_sentences=2)

    def summarize_cluster(self, reviews: list, num_sentences: int = 3) -> dict:
        """
        Summarizes a cluster of reviews.
        Returns extractive summary, abstractive summary, and count statistics.
        """
        if not reviews:
            return {"extractive": "", "abstractive": "", "count": 0}
            
        combined_text = " ".join([r["text"] if isinstance(r, dict) else str(r) for r in reviews])
        
        extractive = self.summarize_extractive(combined_text, num_sentences=num_sentences)
        
        # Keep abstractive input length bounded
        abstractive_input = combined_text[:3000]
        abstractive = self.summarize_abstractive(abstractive_input)
        
        return {
            "extractive": extractive,
            "abstractive": abstractive,
            "count": len(reviews)
        }
