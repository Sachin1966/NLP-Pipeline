import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Boolean, Table
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# Many-to-many relationship for reviews and topics
review_topics = Table(
    'review_topics',
    Base.metadata,
    Column('review_id', Integer, ForeignKey('reviews.id', ondelete='CASCADE'), primary_key=True),
    Column('topic_id', Integer, ForeignKey('topics.id', ondelete='CASCADE'), primary_key=True),
    Column('confidence', Float, default=1.0)
)

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(50), default="user")  # admin, manager, user
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class Review(Base):
    __tablename__ = 'reviews'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    text = Column(String, nullable=False)
    cleaned_text = Column(String, nullable=True)
    source = Column(String(100), nullable=True)
    user = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    global_sentiment = Column(String(20), nullable=True)
    global_sentiment_score = Column(Float, nullable=True)
    rating = Column(Float, nullable=True)
    category = Column(String(50), nullable=True)  # Billing Issue, Feature Request, Bug Report, etc.
    emotion = Column(String(50), nullable=True)
    toxicity_score = Column(Float, nullable=True)
    is_toxic = Column(Boolean, default=False)
    processed = Column(Boolean, default=False)
    
    # Relationships
    entities = relationship("Entity", back_populates="review", cascade="all, delete-orphan")
    aspects = relationship("AspectSentiment", back_populates="review", cascade="all, delete-orphan")
    topics = relationship("Topic", secondary=review_topics, back_populates="reviews")

class Entity(Base):
    __tablename__ = 'entities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey('reviews.id', ondelete='CASCADE'), nullable=False)
    text = Column(String(255), nullable=False, index=True)
    label = Column(String(100), nullable=False)  # ORG, PRODUCT, GPE, etc.
    
    review = relationship("Review", back_populates="entities")

class AspectSentiment(Base):
    __tablename__ = 'aspect_sentiments'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey('reviews.id', ondelete='CASCADE'), nullable=False)
    aspect = Column(String(100), nullable=False, index=True)  # battery, UI, features, etc.
    sentiment = Column(String(20), nullable=False)  # POSITIVE, NEGATIVE, NEUTRAL
    score = Column(Float, default=1.0)
    clause = Column(String, nullable=True)
    
    review = relationship("Review", back_populates="aspects")

class Topic(Base):
    __tablename__ = 'topics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    keywords = Column(String, nullable=True)  # Comma separated keywords
    
    reviews = relationship("Review", secondary=review_topics, back_populates="topics")

class QualityMetric(Base):
    __tablename__ = 'quality_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    missing_count = Column(Integer, default=0)
    duplicate_count = Column(Integer, default=0)
    corrupted_count = Column(Integer, default=0)
    invalid_lang_count = Column(Integer, default=0)
    report_json = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class DriftMetric(Base):
    __tablename__ = 'drift_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    metric_name = Column(String(255), nullable=False)
    value = Column(Float, nullable=False)
    report_json = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
