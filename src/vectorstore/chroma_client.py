import os
import yaml
import logging
import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# Load configuration
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "pipeline_config.yaml")
chroma_path = "data/chroma_db"
collection_name = "customer_voice"

if os.path.exists(config_path):
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'vectorstore' in cfg:
                chroma_path = cfg['vectorstore'].get('chroma_path', chroma_path)
                collection_name = cfg['vectorstore'].get('collection_name', collection_name)
    except Exception as e:
        logger.warning(f"Failed to load vectorstore configuration: {e}")

class VectorStoreManager:
    def __init__(self, persist_dir=chroma_path, coll_name=collection_name):
        self.persist_dir = persist_dir
        self.collection_name = coll_name
        
        os.makedirs(persist_dir, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB persistent client at: {persist_dir}...")
        self.client = chromadb.PersistentClient(path=persist_dir)
        # We handle embeddings ourselves via SentenceEmbedder to avoid duplicates and have exact control
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        logger.info("ChromaDB collection initialized.")

    def add_reviews(self, ids: list, documents: list, metadatas: list, embeddings: np.ndarray):
        """Adds reviews documents and their embeddings to ChromaDB."""
        if not ids:
            return
            
        # Convert embeddings from numpy array to list for Chroma compatibility
        embeddings_list = embeddings.tolist() if hasattr(embeddings, 'tolist') else list(embeddings)
        
        # Batch addition in sizes of 2000 to prevent payload errors
        batch_size = 2000
        total = len(ids)
        for i in range(0, total, batch_size):
            end = min(i + batch_size, total)
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings_list[i:end]
            )
        logger.info(f"Successfully added {total} reviews to ChromaDB.")

    def semantic_search(self, query_embedding: np.ndarray, limit: int = 5, where_filter: dict = None) -> list:
        """Retrieves raw matches from ChromaDB vector search."""
        query_list = query_embedding.tolist() if hasattr(query_embedding, 'tolist') else list(query_embedding)
        
        results = self.collection.query(
            query_embeddings=[query_list],
            n_results=limit,
            where=where_filter
        )
        
        output = []
        if results and results["ids"] and results["ids"][0]:
            r_ids = results["ids"][0]
            r_docs = results["documents"][0]
            r_metas = results["metadatas"][0]
            r_distances = results["distances"][0] if "distances" in results else [0.0] * len(r_ids)
            
            for doc_id, doc, meta, dist in zip(r_ids, r_docs, r_metas, r_distances):
                output.append({
                    "id": int(doc_id),
                    "text": doc,
                    "metadata": meta,
                    "score": float(1.0 - dist) # Approximate cosine similarity from cosine distance
                })
        return output

    def hybrid_search(self, query: str, query_embedding: np.ndarray, db_reviews: list, limit: int = 5, where_filter: dict = None) -> list:
        """
        Runs Hybrid Search combining Vector Search & BM25 keyword matching using Reciprocal Rank Fusion (RRF).
        RRF scoring: Score(d) = sum(1 / (60 + rank(d)))
        """
        # If database is empty, return empty list
        if not db_reviews:
            return []
            
        # 1. Vector Retrieval (get top 50 candidates)
        vector_candidates = self.semantic_search(query_embedding, limit=max(50, limit * 3), where_filter=where_filter)
        
        # Apply where_filter manually to db_reviews if metadata filtering is requested
        filtered_reviews = db_reviews
        if where_filter:
            filtered_reviews = []
            for r in db_reviews:
                match = True
                for k, v in where_filter.items():
                    # Check if key is in metadata or row object
                    r_val = getattr(r, k, None)
                    if r_val != v:
                        match = False
                        break
                if match:
                    filtered_reviews.append(r)
                    
        if not filtered_reviews:
            return []

        # 2. BM25 Retrieval
        # Tokenize documents
        corpus_texts = [r.text for r in filtered_reviews]
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
        
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = query.lower().split(" ")
        bm25_scores = bm25.get_scores(tokenized_query)
        
        # Get top BM25 candidates
        bm25_indices = np.argsort(bm25_scores)[::-1][:max(50, limit * 3)]
        bm25_candidates = []
        for rank, idx in enumerate(bm25_indices):
            # Only include if there is some score overlap
            if bm25_scores[idx] > 0:
                rev = filtered_reviews[idx]
                bm25_candidates.append({
                    "id": rev.id,
                    "text": rev.text,
                    "metadata": {
                        "source": rev.source,
                        "user": rev.user,
                        "timestamp": rev.timestamp.isoformat() if hasattr(rev.timestamp, 'isoformat') else str(rev.timestamp),
                        "global_sentiment": rev.global_sentiment,
                        "category": rev.category
                    },
                    "score": float(bm25_scores[idx])
                })

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        doc_details = {}
        
        # Helper to record rank
        def process_candidate_list(candidates):
            for rank, cand in enumerate(candidates):
                doc_id = cand["id"]
                if doc_id not in rrf_scores:
                    rrf_scores[doc_id] = 0.0
                    doc_details[doc_id] = cand
                # RRF equation
                rrf_scores[doc_id] += 1.0 / (60.0 + rank + 1)
                
        process_candidate_list(vector_candidates)
        process_candidate_list(bm25_candidates)
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:limit]
        
        final_results = []
        for d_id in sorted_ids:
            cand = doc_details[d_id]
            cand["rrf_score"] = float(rrf_scores[d_id])
            final_results.append(cand)
            
        return final_results
