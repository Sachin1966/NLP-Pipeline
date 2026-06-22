import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.database.connection import init_db
from src.api.endpoints import router as api_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Voice Intelligence API Platform",
    description="Production-grade FastAPI platform for customer reviews NLP extraction, Semantic Search, and RAG capabilities.",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    logger.info("Starting Customer Voice Platform API...")
    # Initialize SQL database schema
    init_db()
    logger.info("SQL Databases initialized successfully.")

# Include router
app.include_router(api_router, tags=["Customer Voice API"])

@app.get("/")
def read_root():
    return {
        "status": "online",
        "api_documentation": "/docs",
        "system": "Customer Voice Intelligence Engine"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
