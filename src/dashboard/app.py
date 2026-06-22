import os
import sys

# Add project root to sys.path to enable src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tempfile
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px

# Core imports
from src.database.connection import SessionLocal, get_db, init_db
from src.database.models import Review, Entity, AspectSentiment, QualityMetric, DriftMetric
from src.dashboard.auth_ui import render_auth_page
from src.ingestion.loader import DocumentIngester
from src.ingestion.validator import DataValidator
from src.preprocessing.cleaner import TextCleaner
from src.models.nlp_engine import NLPPipelineEngine
from src.embeddings.embedder import SentenceEmbedder
from src.vectorstore.chroma_client import VectorStoreManager
from src.rag.engine import RAGManager
from src.ingestion.synthetic import generate_synthetic_dataset

# Initialize Streamlit Page Config
st.set_page_config(
    page_title="Customer Voice Intelligence Engine",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme & premium styling overrides
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #E5E7EB;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-top: 5px;
    }
    .metric-title {
        font-size: 0.95rem;
        color: #4B5563;
        text-transform: uppercase;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE PIPELINE COMPONENT SINGLETONS ---
@st.cache_resource
def get_shared_resources():
    # Automatically initialize SQL database tables
    init_db()
    embedder = SentenceEmbedder()
    vectorstore = VectorStoreManager()
    pipeline = NLPPipelineEngine()
    rag = RAGManager()
    ingester = DocumentIngester()
    validator = DataValidator()
    return embedder, vectorstore, pipeline, rag, ingester, validator

try:
    db_embedder, db_vectorstore, nlp_engine, rag_manager, ingester, validator = get_shared_resources()
except Exception as e:
    st.error(f"Failed to initialize AI NLP models: {e}")
    st.stop()

# Authentication Guard
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    render_auth_page()
    st.stop()

# --- SIDEBAR CONTROLS & AUTH ---
st.sidebar.title(f"👤 {st.session_state['username']}")
st.sidebar.markdown(f"**Role:** `{st.session_state['role'].upper()}`")

import os
low_mem = os.getenv("LOW_MEMORY_MODE", "false").lower() == "true"
if low_mem:
    st.sidebar.info("🚀 **Low Memory Mode** is enabled. Using lightweight heuristic NLP fallback engines to conserve RAM.")

if st.sidebar.button("Logout", type="secondary", use_container_width=True):
    st.session_state["authenticated"] = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("📥 Ingestion Control")

# Document File Upload
uploaded_file = st.sidebar.file_uploader("Upload feedback doc", type=["csv", "xlsx", "xls", "json", "txt", "pdf", "docx", "png", "jpg"])
if uploaded_file is not None:
    if st.sidebar.button("Ingest Uploaded File", type="primary", use_container_width=True):
        with st.spinner("Processing file..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_f:
                temp_f.write(uploaded_file.getbuffer())
                temp_path = temp_f.name
                
            db = SessionLocal()
            try:
                records = ingester.load_file(temp_path)
                # Ingest records
                from src.api.endpoints import process_and_store_reviews
                inserted_count = process_and_store_reviews(records, db)
                st.sidebar.success(f"Ingested {inserted_count} valid records from {uploaded_file.name}!")
            except Exception as e:
                st.sidebar.error(f"Ingestion failed: {e}")
            finally:
                db.close()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
st.sidebar.markdown("---")
st.sidebar.subheader("🧪 Synthetic Data Tool")
synth_count = st.sidebar.slider("Select dataset volume", min_value=100, max_value=50000, value=1000, step=100)
if st.sidebar.button("Generate Synthetic Reviews", use_container_width=True):
    with st.spinner(f"Generating & Ingesting {synth_count} reviews in database..."):
        db = SessionLocal()
        try:
            output_csv = "data/raw/synthetic_dataset.csv"
            generate_synthetic_dataset(output_path=output_csv, count=synth_count)
            df = pd.read_csv(output_csv)
            records = df.to_dict(orient="records")
            
            from src.api.endpoints import process_and_store_reviews
            inserted = process_and_store_reviews(records, db)
            st.sidebar.success(f"Successfully generated and loaded {inserted} reviews!")
        except Exception as e:
            st.sidebar.error(f"Generation error: {e}")
        finally:
            db.close()

# --- LOAD DATA FOR RENDERING ---
db = SessionLocal()
try:
    reviews = db.query(Review).all()
    aspects = db.query(AspectSentiment).all()
    entities = db.query(Entity).all()
    metrics = db.query(QualityMetric).all()
finally:
    db.close()

df_reviews = pd.DataFrame([{
    "id": r.id, "text": r.text, "source": r.source, "user": r.user,
    "timestamp": r.timestamp, "global_sentiment": r.global_sentiment,
    "global_sentiment_score": r.global_sentiment_score, "rating": r.rating,
    "category": r.category, "emotion": r.emotion, "toxicity_score": r.toxicity_score,
    "is_toxic": r.is_toxic
} for r in reviews])

df_aspects = pd.DataFrame([{
    "id": a.id, "review_id": a.review_id, "aspect": a.aspect,
    "sentiment": a.sentiment, "score": a.score, "clause": a.clause
} for a in aspects])

df_entities = pd.DataFrame([{
    "id": e.id, "review_id": e.review_id, "text": e.text, "label": e.label
} for e in entities])

# Handle empty states
if df_reviews.empty:
    st.info("👋 Welcome! The Customer Voice database is currently empty. Use the sidebar to load synthetic reviews or upload feedback files.")
    st.stop()

# --- MASTER TABS ROUTING ---
tab_overview, tab_sentiment, tab_aspect, tab_topic, tab_entity, tab_search, tab_rag, tab_monitoring = st.tabs([
    "📊 Overview", "📈 Sentiment", "🏷️ Aspects", "💬 Topics", "🔍 NER Entities", "🔎 Semantic Search", "🤖 RAG Assistant", "🛡️ Monitoring"
])
# 1. OVERVIEW DASHBOARD
with tab_overview:
    st.subheader("Customer Voice Overview Dashboard")
    
    # Calculate KPIs
    total_revs = len(df_reviews)
    avg_sent = df_reviews["global_sentiment_score"].mean()
    pos_ratio = (df_reviews["global_sentiment"] == "POSITIVE").mean()
    toxic_ratio = df_reviews["is_toxic"].mean()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Total Reviews Ingested</div><div class="metric-value">{total_revs:,}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Average Sentiment Score</div><div class="metric-value">{avg_sent:.1%}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Positive Rating Ratio</div><div class="metric-value">{pos_ratio:.1%}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-title">Toxicity Flag Rate</div><div class="metric-value">{toxic_ratio:.1%}</div></div>', unsafe_allow_html=True)
        
    st.markdown("### Recent Customer Ingested Voice Feed")
    st.dataframe(df_reviews[["timestamp", "user", "source", "category", "global_sentiment", "text"]].tail(100), use_container_width=True)

# 2. SENTIMENT & EMOTION
with tab_sentiment:
    st.subheader("Brand Sentiment & Emotion Analysis")
    
    col_sent_l, col_sent_r = st.columns(2)
    with col_sent_l:
        sent_counts = df_reviews["global_sentiment"].value_counts().reset_index()
        fig_pie = px.pie(sent_counts, values="count", names="global_sentiment", color="global_sentiment",
                         color_discrete_map={"POSITIVE": "#10B981", "NEGATIVE": "#EF4444", "NEUTRAL": "#F59E0B"},
                         title="Overall Sentiment Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_sent_r:
        emotion_counts = df_reviews["emotion"].value_counts().reset_index()
        fig_bar = px.bar(emotion_counts, x="emotion", y="count", color="emotion",
                         title="Emotion Detection Breakdown")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.markdown("### Sentiment Over Time (Trend Index)")
    df_timeline = df_reviews.copy()
    df_timeline["date"] = pd.to_datetime(df_timeline["timestamp"]).dt.date
    sent_timeline = df_timeline.groupby("date")["global_sentiment_score"].mean().reset_index()
    fig_line = px.line(sent_timeline, x="date", y="global_sentiment_score", title="Sentiment Index Trend Curve")
    st.plotly_chart(fig_line, use_container_width=True)

# 3. ASPECT-BASED SENTIMENT
with tab_aspect:
    st.subheader("Aspect-Based Sentiment Dashboards")
    
    if not df_aspects.empty:
        aspect_grp = df_aspects.groupby(["aspect", "sentiment"]).size().unstack(fill_value=0).reset_index()
        # Stacked bar chart
        fig_aspect = px.bar(aspect_grp, x="aspect", y=["POSITIVE", "NEGATIVE", "NEUTRAL"],
                            color_discrete_map={"POSITIVE": "#10B981", "NEGATIVE": "#EF4444", "NEUTRAL": "#F59E0B"},
                            title="Sentiment Associations by Feature Aspects", barmode="stack")
        st.plotly_chart(fig_aspect, use_container_width=True)
    else:
        st.info("No detailed aspect records found in database yet. Try loading richer synthetic reviews.")

# 4. TOPIC MODELING
with tab_topic:
    st.subheader("c-TF-IDF Topic Modeling Discovery")
    
    num_topics = st.slider("Number of topics to discover", min_value=2, max_value=10, value=5)
    
    if st.button("Re-run Semantic Topic Modeling"):
        with st.spinner("Clustering sentence embeddings..."):
            # Fetch all review embeddings
            emb_db = SessionLocal()
            try:
                rev_list = emb_db.query(Review).all()
                texts = [r.text for r in rev_list]
                
                # Fetch embeddings via cache
                embeddings = db_embedder.embed_batch(texts)
                
                from src.models.topics import TopicModeler
                modeler = TopicModeler()
                topic_results = modeler.fit_semantic_topics(texts, embeddings, n_topics=num_topics)
                
                st.write("### Discovered Semantic Topic Clusters (c-TF-IDF Ranks)")
                for t in topic_results["topics"]:
                    st.markdown(f"🔹 **{t['name']}**  \n*Keywords:* {', '.join(t['keywords'])}")
            finally:
                emb_db.close()
    else:
        st.info("Click 'Re-run Semantic Topic Modeling' to run c-TF-IDF clustering.")

# 5. NER ENTITIES
with tab_entity:
    st.subheader("Named Entity Extraction (NER)")
    
    if not df_entities.empty:
        ent_grp = df_entities["label"].value_counts().reset_index()
        fig_ent = px.bar(ent_grp, x="label", y="count", color="label", title="Named Entity Category Mentions")
        st.plotly_chart(fig_ent, use_container_width=True)
        
        st.markdown("### Top Identified Key Entities")
        top_ents = df_entities["text"].value_counts().reset_index().head(20)
        st.dataframe(top_ents, use_container_width=True)
    else:
        st.info("No NER entities detected. Ingest feedback to run NER parsing.")

# 6. SEMANTIC & HYBRID SEARCH
with tab_search:
    st.subheader("Hybrid Semantic Search Engine")
    search_q = st.text_input("Enter search term", placeholder="I need help with login errors and SSO setting issues")
    limit_val = st.slider("Search results limit", 1, 10, 5)
    
    if search_q:
        with st.spinner("Retrieving matches using hybrid RRF query..."):
            # Embed query
            query_emb = db_embedder.embed_text(search_q)
            db_conn = SessionLocal()
            try:
                db_reviews_all = db_conn.query(Review).all()
                results = db_vectorstore.hybrid_search(
                    query=search_q,
                    query_embedding=query_emb,
                    db_reviews=db_reviews_all,
                    limit=limit_val
                )
                
                for idx, r in enumerate(results):
                    meta = r.get("metadata", {})
                    st.markdown(f"""
                    <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid #1E3A8A;">
                        <strong>User: {meta.get('user', 'Anonymous')}</strong> on <em>{meta.get('source', 'Channel')}</em> ({meta.get('timestamp', '')}) <br>
                        <strong>Sentiment:</strong> {meta.get('global_sentiment', 'Neutral')} | <strong>RRF Rank Score:</strong> {r.get('rrf_score', 0.0):.4f} <br>
                        <p style="margin-top: 8px; font-style: italic;">"{r['text']}"</p>
                    </div>
                    """, unsafe_allow_html=True)
            finally:
                db_conn.close()

# 7. CONVERSATIONAL RAG ASSISTANT
with tab_rag:
    st.subheader("AI RAG Chat Assistant")
    
    openai_key = st.text_input("OpenAI Key (optional)", type="password", help="sk-... keys")
    use_ollama_local = st.checkbox("Use Ollama Server", value=False)
    
    ollama_host_val = "http://localhost:11434"
    if use_ollama_local:
        ollama_host_val = st.text_input("Ollama Endpoint Host", value=os.getenv("OLLAMA_HOST", "http://localhost:11434"), help="Endpoint URL, e.g. http://localhost:11434 or a public ngrok tunnel URL.")
        
    ollama_model_choice = st.selectbox("Select Ollama Model", ["llama3.2", "llama3", "gemma:2b"])
    
    if st.button("Reset Conversation Memory"):
        rag_manager.clear_memory("default")
        st.success("Conversation history cleared!")
        
    chat_q = st.text_input("Ask about customer issues", placeholder="What are customers complaining about screen and batteries?")
    
    if chat_q:
        with st.spinner("Searching records & composing answer..."):
            query_emb = db_embedder.embed_text(chat_q)
            db_conn = SessionLocal()
            try:
                db_reviews_all = db_conn.query(Review).all()
                # Get relevant context
                retrieved_docs = db_vectorstore.hybrid_search(
                    query=chat_q,
                    query_embedding=query_emb,
                    db_reviews=db_reviews_all,
                    limit=5
                )
                
                # Answer
                rag_result = rag_manager.answer_question(
                    question=chat_q,
                    retrieved_docs=retrieved_docs,
                    openai_api_key=openai_key,
                    use_ollama=use_ollama_local,
                    ollama_model=ollama_model_choice,
                    session_id="default",
                    ollama_host=ollama_host_val
                )
                
                st.markdown("### 📝 AI Generated Response")
                st.write(rag_result["answer"])
                st.info(f"**Inference Engine:** {rag_result['engine_used']}")
                
                # Citations
                with st.expander("📚 Source Citations"):
                    for src in rag_result["sources"]:
                        st.markdown(f"- **[Source #{src['source_id']}]** User `{src['user']}` on {src['source']} ({src['timestamp'][:10]}): *\"{src['text']}\"*")
            finally:
                db_conn.close()

# 8. MONITORING
with tab_monitoring:
    st.subheader("EvidentlyAI Monitoring & Data Quality Dashboard")
    
    # Display Ingestion validation reports
    if metrics:
        st.write("### Data Quality Metrics Over Time")
        quality_df = pd.DataFrame([{
            "timestamp": m.timestamp,
            "missing": m.missing_count,
            "duplicates": m.duplicate_count,
            "corrupted": m.corrupted_count,
            "invalid_lang": m.invalid_lang_count
        } for m in metrics])
        
        st.line_chart(quality_df.set_index("timestamp"))
        st.dataframe(quality_df, use_container_width=True)
    else:
        st.info("No quality metric logs compiled yet. Run dataset ingestion to log reports.")
