"""
FastAPI Backend — Data Ingestion & Q&A Endpoints
=================================================
POST /upload           — Upload files (PDF, DOCX, TXT, PPT, Code, WAV)
POST /url              — Submit a web URL or Wikipedia link
POST /ask              — Ask a question (uses the LCEL orchestrator)
GET  /knowledge/status — Check knowledge base stats
DELETE /knowledge/clear — Clear the entire knowledge base
"""

import os
import sys
import shutil
import logging

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backend.core.ingestion import (
    load_document,
    load_url,
    store_documents,
    get_knowledge_stats,
    clear_knowledge_base,
)
from src.backend.orchestrator.orchestrator import EvaluatorGeneratorOrchestrator
from src.backend.orchestrator.logging_config import setup_logging
from src.config.settings import DATA_DIR

# ---- Setup ----
setup_logging()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Evaluator-Generator NLP Platform",
    description="Data Ingestion, Retrieval, and LLM-powered Q&A with feedback loop.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DATA_DIR, exist_ok=True)

# Single orchestrator instance
orchestrator = EvaluatorGeneratorOrchestrator()


# ============================================================
# Request / Response Models
# ============================================================

class URLRequest(BaseModel):
    url: str
    is_wikipedia: bool = False

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    question: str
    answer: str
    iterations: int
    accepted: bool
    feedback: str
    history: list = []


# ============================================================
# Endpoints
# ============================================================

@app.post("/upload", tags=["Knowledge Ingestion"])
async def upload_file(file: UploadFile = File(...)):
    """Upload a file, extract text, chunk, embed, and store in ChromaDB."""
    try:
        temp_path = os.path.join(DATA_DIR, file.filename)
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        documents = load_document(temp_path, file.filename)
        store_documents(documents)
        os.remove(temp_path)

        return {
            "message": f"Successfully processed '{file.filename}'.",
            "documents_loaded": len(documents),
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error processing '{file.filename}': {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/url", tags=["Knowledge Ingestion"])
async def process_url(request: URLRequest):
    """Fetch content from a URL or Wikipedia, chunk, embed, and store."""
    try:
        documents = load_url(request.url, request.is_wikipedia)
        store_documents(documents)
        return {
            "message": f"Successfully processed '{request.url}'.",
            "documents_loaded": len(documents),
        }
    except Exception as e:
        logger.error(f"Error processing URL '{request.url}': {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/ask", response_model=AskResponse, tags=["Q&A"])
async def ask_question(request: AskRequest):
    """
    Ask a question. Runs the full LCEL pipeline:
    Retrieve -> Generate -> Evaluate -> (feedback loop, max 4 iterations).
    """
    question = request.question
    logger.info(f"Question received: {question}")

    try:
        result = orchestrator.run(question)

        last_feedback = result.history[-1]["feedback"] if result.history else ""

        return AskResponse(
            question=question,
            answer=result.final_answer,
            iterations=result.iterations_used,
            accepted=result.accepted,
            feedback=last_feedback,
            history=result.history,
        )
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/knowledge/status", tags=["Knowledge Management"])
async def knowledge_status():
    """Check the current state of the knowledge base."""
    return get_knowledge_stats()


@app.delete("/knowledge/clear", tags=["Knowledge Management"])
async def knowledge_clear():
    """Clear the entire knowledge base."""
    return clear_knowledge_base()
