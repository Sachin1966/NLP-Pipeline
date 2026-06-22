import os
import yaml
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base

logger = logging.getLogger(__name__)

# Load config
config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs", "pipeline_config.yaml")
db_url = None
sqlite_path = "data/voice_intelligence.db"

try:
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            if cfg and 'database' in cfg:
                db_url = cfg['database'].get('postgres_url')
                sqlite_path = cfg['database'].get('sqlite_path', sqlite_path)
except Exception as e:
    logger.warning(f"Failed to load yaml config for database: {e}")

# Environment variables take precedence
if os.getenv("POSTGRES_URL"):
    db_url = os.getenv("POSTGRES_URL")

engine = None
SessionLocal = None

# Establish DB connection with fallback
try:
    if db_url:
        logger.info(f"Attempting connection to PostgreSQL database...")
        # Add quick timeout parameters to avoid hanging
        engine = create_engine(db_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        # Test connection
        conn = engine.connect()
        conn.close()
        logger.info("Successfully connected to PostgreSQL database.")
    else:
        raise ValueError("No PostgreSQL URL configured.")
except Exception as e:
    logger.warning(f"PostgreSQL connection failed or not configured ({e}). Falling back to SQLite...")
    # SQLite Fallback
    os.makedirs(os.path.dirname(os.path.abspath(sqlite_path)), exist_ok=True)
    sqlite_url = f"sqlite:///{os.path.abspath(sqlite_path)}"
    engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})
    logger.info(f"SQLite database initialized at: {sqlite_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    """Initializes tables in database."""
    Base.metadata.create_all(bind=engine)

def get_db():
    """Dependency generator for FastAPI context management."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
