import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class TopicModeler:
    def __init__(self):
        pass

    def fit_lda(self, texts: list, n_topics: int = 5, n_words: int = 8) -> list:
        """
        Discovers topics using Latent Dirichlet Allocation (LDA).
        Returns a list of topics with keywords.
        """
        if not texts or len(texts) < n_topics:
            return [{"id": 0, "name": "General Feedback", "keywords": ["general", "feedback"]}]
            
        vectorizer = CountVectorizer(stop_words='english', max_features=1000)
        dtm = vectorizer.fit_transform(texts)
        
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        
        words = vectorizer.get_feature_names_out()
        topics = []
        
        for topic_idx, topic in enumerate(lda.components_):
            top_words_idx = topic.argsort()[:-n_words - 1:-1]
            top_words = [words[i] for i in top_words_idx]
            topics.append({
                "id": topic_idx,
                "name": f"LDA Topic {topic_idx + 1}: " + ", ".join(top_words[:3]),
                "keywords": top_words
            })
            
        return topics

    def fit_semantic_topics(self, texts: list, embeddings: np.ndarray, n_topics: int = 5, n_words: int = 8) -> dict:
        """
        Runs a lightweight c-TF-IDF (BERTopic-style) topic model:
        1. Clusters reviews using KMeans on SentenceTransformer embeddings.
        2. Merges documents in each cluster.
        3. Extracts keywords for each cluster using TF-IDF (Class-based TF-IDF).
        """
        if not texts or len(texts) < n_topics or embeddings is None or len(embeddings) != len(texts):
            return {
                "topics": [{"id": 0, "name": "General", "keywords": ["general"]}],
                "assignments": [0] * len(texts)
            }
            
        # 1. Clustering
        n_clusters = min(n_topics, len(texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 2. c-TF-IDF calculation
        # Group text by cluster
        df = pd.DataFrame({"text": texts, "cluster": cluster_labels})
        cluster_docs = df.groupby("cluster")["text"].apply(lambda x: " ".join(x)).reset_index()
        
        tfidf = TfidfVectorizer(stop_words='english', max_features=2000)
        try:
            tfidf_matrix = tfidf.fit_transform(cluster_docs["text"])
            words = tfidf.get_feature_names_out()
            
            topics = []
            for idx, row in cluster_docs.iterrows():
                cluster_id = int(row["cluster"])
                tf_row = tfidf_matrix[idx].toarray()[0]
                top_word_indices = tf_row.argsort()[:-n_words-1:-1]
                top_words = [words[i] for i in top_word_indices if tf_row[i] > 0]
                
                # If cluster has no distinct words, fill with placeholder
                if not top_words:
                    top_words = ["feedback", "customer", "product"]
                    
                topics.append({
                    "id": cluster_id,
                    "name": f"Topic {cluster_id + 1}: " + ", ".join(top_words[:3]),
                    "keywords": top_words
                })
        except Exception as e:
            logger.error(f"Error computing c-TF-IDF: {e}")
            topics = [{
                "id": int(i),
                "name": f"Cluster Topic {i+1}",
                "keywords": ["general"]
            } for i in range(n_clusters)]
            
        return {
            "topics": topics,
            "assignments": [int(label) for label in cluster_labels]
        }

    def track_topic_trends(self, df_reviews: pd.DataFrame, time_col: str = "timestamp", topic_col: str = "topic_id") -> pd.DataFrame:
        """
        Calculates topic frequencies over time to track trends.
        """
        if df_reviews.empty or time_col not in df_reviews.columns or topic_col not in df_reviews.columns:
            return pd.DataFrame()
            
        df = df_reviews.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        # Resample by month or week
        df["period"] = df[time_col].dt.to_period("M").astype(str)
        
        trend = df.groupby(["period", topic_col]).size().unstack(fill_value=0).reset_index()
        return trend
