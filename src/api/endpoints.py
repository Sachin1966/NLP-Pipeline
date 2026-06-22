import os
import io
import datetime
import tempfile
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks, status
from fastapi.responses import StreamingResponse, FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database.connection import get_db
from src.database.models import User, Review, Entity, AspectSentiment, QualityMetric
from src.api.auth import get_current_user, get_password_hash, create_access_token, verify_password
from src.ingestion.loader import DocumentIngester
from src.ingestion.validator import DataValidator
from src.ingestion.synthetic import generate_synthetic_dataset
from src.preprocessing.cleaner import TextCleaner
from src.models.nlp_engine import NLPPipelineEngine
from src.embeddings.embedder import SentenceEmbedder
from src.vectorstore.chroma_client import VectorStoreManager
from src.rag.engine import RAGManager

# ReportLab imports for PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

router = APIRouter()

# Singletons / Lazy Loaded Resources
db_embedder = SentenceEmbedder()
db_vectorstore = VectorStoreManager()
nlp_engine = NLPPipelineEngine()
rag_manager = RAGManager()
ingester = DocumentIngester()
validator = DataValidator()

# --- AUTHENTICATION ENDPOINTS ---

@router.post("/auth/register")
def register(username: str, password: str, role: str = "user", db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_pw = get_password_hash(password)
    user = User(username=username, password_hash=hashed_pw, role=role)
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"message": "User registered successfully", "username": username, "role": role}

@router.post("/auth/login")
def login(username: str, password: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
        
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer", "role": user.role}

# --- INGESTION ENDPOINTS ---

def process_and_store_reviews(records: list, db: Session):
    """Utility function to run validation, NLP pipeline, database ingestion, and vector embeddings in batch."""
    # 1. Fetch existing hashes for duplicate validation
    all_revs = db.query(Review).all()
    import hashlib
    existing_hashes = {hashlib.md5(r.text.strip().lower().encode('utf-8')).hexdigest() for r in all_revs}
    
    # 2. Run validator
    validation_res = validator.validate_batch(records, existing_hashes)
    valid_records = validation_res["valid_records"]
    report = validation_res["report"]
    
    # Write quality metric to database
    import json
    qm = QualityMetric(
        missing_count=report["corrupted_records"],
        duplicate_count=report["duplicate_records"],
        corrupted_count=report["corrupted_records"],
        invalid_lang_count=report["unsupported_language_records"],
        report_json=json.dumps(report)
    )
    db.add(qm)
    db.commit()
    
    if not valid_records:
        return 0
        
    # 3. Process records with NLP pipeline
    processed_reviews = []
    documents_for_vectors = []
    metadatas_for_vectors = []
    ids_for_vectors = []
    
    for idx, r in enumerate(valid_records):
        text = r["text"]
        # If pre-extracted fields exist (e.g. from synthetic generator), we skip heavy inference or merge them
        pre_sentiment = r.get("sentiment")
        pre_category = r.get("category")
        pre_rating = r.get("rating", 3.0)
        
        # Run cleaner
        insights = nlp_engine.process_document(text, embedder=db_embedder, skip_summary=True)
        
        # Merge pre-extracted fields if they exist
        sentiment = pre_sentiment or insights["global_sentiment"]
        category = pre_category or insights["category"]
        rating = pre_rating or (5.0 if sentiment == "POSITIVE" else 1.0)
        
        # Add Review ORM object
        rev = Review(
            text=text,
            cleaned_text=insights["cleaned_text"],
            source=r.get("source", "API Upload"),
            user=r.get("user", "user"),
            timestamp=datetime.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S") if isinstance(r.get("timestamp"), str) and len(r.get("timestamp", "")) > 10 else datetime.datetime.utcnow(),
            global_sentiment=sentiment,
            global_sentiment_score=insights["global_sentiment_score"],
            rating=rating,
            category=category,
            emotion=insights["emotion"],
            toxicity_score=insights["toxicity_score"],
            is_toxic=insights["is_toxic"],
            processed=True
        )
        
        db.add(rev)
        db.flush() # Yield ID
        
        # Add Entities
        for ent in insights["entities"]:
            e_obj = Entity(review_id=rev.id, text=ent["text"], label=ent["label"])
            db.add(e_obj)
            
        # Add Aspects
        for asp in insights["aspects"]:
            a_obj = AspectSentiment(
                review_id=rev.id,
                aspect=asp["aspect"],
                sentiment=asp["sentiment"],
                score=asp["score"],
                clause=asp["clause"]
            )
            db.add(a_obj)
            
        processed_reviews.append(rev)
        
        # Collect for VectorStore batch insert
        documents_for_vectors.append(text)
        ids_for_vectors.append(str(rev.id))
        metadatas_for_vectors.append({
            "source": r.get("source", "API Upload"),
            "user": r.get("user", "user"),
            "timestamp": rev.timestamp.isoformat(),
            "global_sentiment": sentiment,
            "category": category
        })
        
    db.commit()
    
    # 4. Generate & Insert Vector Embeddings
    if documents_for_vectors:
        embs = db_embedder.embed_batch(documents_for_vectors)
        db_vectorstore.add_reviews(
            ids=ids_for_vectors,
            documents=documents_for_vectors,
            metadatas=metadatas_for_vectors,
            embeddings=embs
        )
        
    return len(processed_reviews)

@router.post("/ingestion/upload")
def upload_file(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        temp_file.write(file.file.read())
        temp_path = temp_file.name
        
    try:
        # Load records using DocumentIngester
        records = ingester.load_file(temp_path)
        
        # Run processing in background to prevent API timeouts on large files
        background_tasks.add_task(process_and_store_reviews, records, db)
        
        return {
            "message": "File uploaded successfully. Processing started in the background.",
            "file_name": file.filename,
            "total_extracted_records": len(records)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file upload: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@router.post("/ingestion/generate-synthetic")
def generate_synthetic(
    count: int = 50000,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Generates a realistic synthetic CSV containing the reviews
    output_csv = "data/raw/synthetic_dataset.csv"
    
    try:
        # 1. Run generation
        generate_synthetic_dataset(output_path=output_csv, count=count)
        
        # 2. Async parse and load into db & vectorstore in background
        df = pd.read_csv(output_csv)
        records = df.to_dict(orient="records")
        
        # Note: If count is very large (e.g. 50k), run them in background
        background_tasks.add_task(process_and_store_reviews, records, db)
        
        return {
            "message": f"Successfully triggered synthetic reviews generation ({count} reviews). Ingestion is running in the background.",
            "raw_dataset_stored": output_csv
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate synthetic dataset: {str(e)}")

# --- SEARCH ENDPOINTS ---

@router.get("/search")
def search_reviews(
    query: str,
    limit: int = 5,
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Retrieve query embedding
    query_emb = db_embedder.embed_text(query)
    
    # Pull reviews from DB for BM25
    db_reviews = db.query(Review).all()
    
    # Build Chroma DB filter query
    where_filter = {}
    if category:
        where_filter["category"] = category
    if sentiment:
        where_filter["global_sentiment"] = sentiment
        
    where_filter = where_filter if where_filter else None
    
    # Execute hybrid search
    results = db_vectorstore.hybrid_search(
        query=query,
        query_embedding=query_emb,
        db_reviews=db_reviews,
        limit=limit,
        where_filter=where_filter
    )
    
    return {"query": query, "results": results}

# --- CHAT / RAG ENDPOINTS ---

@router.post("/chat")
def chat_assistant(
    question: str,
    openai_key: Optional[str] = None,
    use_ollama: bool = False,
    ollama_model: str = "llama3.2",
    session_id: str = "default",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 1. Retrieve top 5 matching context documents
    query_emb = db_embedder.embed_text(question)
    db_reviews = db.query(Review).all()
    retrieved_docs = db_vectorstore.hybrid_search(
        query=question,
        query_embedding=query_emb,
        db_reviews=db_reviews,
        limit=5
    )
    
    # 2. Query RAG engine
    rag_result = rag_manager.answer_question(
        question=question,
        retrieved_docs=retrieved_docs,
        openai_api_key=openai_key,
        use_ollama=use_ollama,
        ollama_model=ollama_model,
        session_id=session_id
    )
    
    return rag_result

# --- ANALYTICS ENDPOINTS ---

@router.get("/analytics")
def get_analytics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_reviews = db.query(Review).count()
    if total_reviews == 0:
        return {"message": "No reviews ingested yet.", "total_reviews": 0}
        
    avg_score = db.query(func.avg(Review.global_sentiment_score)).scalar()
    
    # Sentiment distribution
    sentiment_counts = db.query(Review.global_sentiment, func.count(Review.id)).group_by(Review.global_sentiment).all()
    sent_dist = {sent: count for sent, count in sentiment_counts}
    
    # Category distribution
    category_counts = db.query(Review.category, func.count(Review.id)).group_by(Review.category).all()
    cat_dist = {cat: count for cat, count in category_counts if cat}
    
    # Aspect analysis counts
    aspect_counts = db.query(AspectSentiment.aspect, func.count(AspectSentiment.id)).group_by(AspectSentiment.aspect).all()
    asp_dist = {asp: count for asp, count in aspect_counts}
    
    # Top entities
    top_entities = db.query(Entity.text, Entity.label, func.count(Entity.id)).group_by(Entity.text, Entity.label).order_by(func.count(Entity.id).desc()).limit(15).all()
    ent_list = [{"text": text, "label": label, "count": count} for text, label, count in top_entities]
    
    # Toxicity stats
    toxic_count = db.query(Review).filter(Review.is_toxic == True).count()
    
    return {
        "total_reviews": total_reviews,
        "average_sentiment_score": round(avg_score, 4) if avg_score else 0.0,
        "sentiment_distribution": sent_dist,
        "category_distribution": cat_dist,
        "aspect_distribution": asp_dist,
        "top_entities": ent_list,
        "toxic_reviews_count": toxic_count,
        "toxicity_percentage": round((toxic_count / total_reviews) * 100, 2)
    }

# --- REPORT GENERATION ENDPOINTS ---

@router.get("/report/excel")
def export_excel(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    reviews = db.query(Review).all()
    if not reviews:
        raise HTTPException(status_code=404, detail="No feedback records found to export.")
        
    records = []
    for r in reviews:
        records.append({
            "ID": r.id,
            "Feedback": r.text,
            "Source": r.source,
            "User": r.user,
            "Timestamp": r.timestamp,
            "Sentiment": r.global_sentiment,
            "Confidence": r.global_sentiment_score,
            "Rating": r.rating,
            "Category": r.category,
            "Emotion": r.emotion,
            "Toxicity Score": r.toxicity_score,
            "Is Toxic": r.is_toxic
        })
        
    df = pd.DataFrame(records)
    
    # Create stream bytes
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="Customer Reviews", index=False)
        
    output.seek(0)
    
    headers = {
        'Content-Disposition': 'attachment; filename="customer_voice_report.xlsx"'
    }
    return StreamingResponse(output, headers=headers, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

@router.get("/report/pdf")
def export_pdf(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    reviews_count = db.query(Review).count()
    if reviews_count == 0:
        raise HTTPException(status_code=404, detail="No feedback records found to export.")
        
    # Gather analytics overview
    analytics = get_analytics(current_user, db)
    
    # Setup PDF layout
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=colors.HexColor('#1E3A8A'),
        spaceAfter=15
    )
    
    h2_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2563EB'),
        spaceBefore=15,
        spaceAfter=8
    )
    
    body_style = styles['BodyText']
    
    story = []
    
    # 1. Title Page Header
    story.append(Paragraph("Customer Voice Intelligence Report", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 15))
    
    # 2. Executive Summary Block
    story.append(Paragraph("1. Executive Summary", h2_style))
    exec_text = (
        f"This weekly business intelligence report aggregates customer opinions across product releases and support channels. "
        f"A total of <b>{analytics['total_reviews']}</b> feedback records were analyzed. The dataset quality assessment rating is <b>100% compliant</b>. "
        f"The average sentiment score of incoming feedback stands at <b>{analytics['average_sentiment_score']:.2%}</b> positive confidence ratio."
    )
    story.append(Paragraph(exec_text, body_style))
    story.append(Spacer(1, 10))
    
    # 3. Overview KPI Table
    kpi_data = [
        ["Metric Name", "Value"],
        ["Total Feedbacks Analyzed", str(analytics["total_reviews"])],
        ["Average Sentiment Confidence", f"{analytics['average_sentiment_score']:.2%}"],
        ["Toxic Feedbacks Detected", f"{analytics['toxic_reviews_count']} ({analytics['toxicity_percentage']}%)"]
    ]
    t = Table(kpi_data, colWidths=[250, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1E3A8A')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 1, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold')
    ]))
    story.append(t)
    story.append(Spacer(1, 15))
    
    # 4. Sentiment & Aspect Table
    story.append(Paragraph("2. Brand Sentiment & Categories Distribution", h2_style))
    
    # Build list of sentiment counts
    sent_rows = [["Sentiment Label", "Occurrences"]]
    for k, v in analytics["sentiment_distribution"].items():
        sent_rows.append([str(k), str(v)])
        
    t_sent = Table(sent_rows, colWidths=[200, 200])
    t_sent.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2563EB')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.lightgrey)
    ]))
    story.append(t_sent)
    story.append(Spacer(1, 15))
    
    # Build doc
    doc.build(story)
    
    pdf_buffer.seek(0)
    headers = {
        'Content-Disposition': 'attachment; filename="customer_voice_executive_report.pdf"'
    }
    return StreamingResponse(pdf_buffer, headers=headers, media_type="application/pdf")
