"""
Knowledge Extraction, Embedding & Vector Store Logic
=====================================================
Handles: Document loading, audio transcription, image OCR, text chunking,
         embedding generation, and ChromaDB storage/retrieval.
"""

import os
import logging
from typing import List

import whisper
# import speech_recognition as sr   # Google SpeechRecognition (commented out)
# from pydub import AudioSegment    # For audio conversion (commented out)

import pytesseract
from PIL import Image

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    WebBaseLoader,
    WikipediaLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# ---- Active Embeddings: HuggingFace (Free, Local) ----
from langchain_huggingface import HuggingFaceEmbeddings

# ---- Commented Out Embeddings: OpenAI (Requires API Key) ----
# from langchain_openai import OpenAIEmbeddings

from src.config.settings import (
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME,
    TOP_K_RESULTS,
)

logger = logging.getLogger(__name__)

# ============================================================
# Embedding Model
# ============================================================

def get_embeddings():
    """Return the embedding model to use for vectorization."""
    # ---- Active: HuggingFace Embeddings (Free) ----
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # ---- Commented Out: OpenAI Embeddings (Requires OPENAI_API_KEY) ----
    # return OpenAIEmbeddings()


# ============================================================
# Audio Transcription
# ============================================================

def transcribe_audio_whisper(file_path: str) -> str:
    """
    Transcribe audio file using OpenAI Whisper (Active).
    Supports WAV, MP3, M4A formats.
    """
    logger.info(f"Transcribing audio with Whisper: {file_path}")
    model = whisper.load_model("base")
    result = model.transcribe(file_path)
    return result["text"]


def transcribe_audio_google(file_path: str) -> str:
    """
    Transcribe audio using Google SpeechRecognition (Commented Out).
    To activate: uncomment the imports at the top and uncomment the body below.
    """
    # recognizer = sr.Recognizer()
    #
    # # Convert to WAV if needed
    # audio = AudioSegment.from_file(file_path)
    # wav_path = file_path + ".wav"
    # audio.export(wav_path, format="wav")
    #
    # with sr.AudioFile(wav_path) as source:
    #     audio_data = recognizer.record(source)
    #     try:
    #         text = recognizer.recognize_google(audio_data)
    #         return text
    #     except sr.UnknownValueError:
    #         return "Google Speech Recognition could not understand audio."
    #     except sr.RequestError as e:
    #         return f"Could not request results from Google; {e}"
    pass


# ============================================================
# Image OCR (Text Extraction from Images)
# ============================================================

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Supports PNG, JPG, JPEG, BMP, TIFF, WEBP.
    """
    logger.info(f"Extracting text from image with Tesseract: {file_path}")
    try:
        img = Image.open(file_path)
        text = pytesseract.image_to_string(img)
        text = text.strip()
        if not text:
            return "No text could be extracted from this image."
        return text
    except Exception as e:
        logger.error(f"OCR failed for {file_path}: {e}")
        return f"Error extracting text from image: {e}"


# ============================================================
# Document Loading
# ============================================================

CODE_EXTENSIONS = {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rb", ".rs", ".html", ".css"}


def load_document(file_path: str, filename: str) -> List[Document]:
    """
    Load a document based on its file extension.
    Returns a list of LangChain Document objects.
    """
    ext = os.path.splitext(filename)[1].lower()
    logger.info(f"Loading document: {filename} (type: {ext})")

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        return loader.load()

    elif ext in (".docx", ".doc"):
        loader = Docx2txtLoader(file_path)
        return loader.load()

    elif ext == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        return loader.load()

    elif ext == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
        return loader.load()

    elif ext in CODE_EXTENSIONS:
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "code"
            doc.metadata["language"] = ext.lstrip(".")
        return docs

    elif ext in (".ppt", ".pptx"):
        loader = UnstructuredPowerPointLoader(file_path)
        return loader.load()

    elif ext in (".wav", ".mp3", ".m4a"):
        # ---- Active: Whisper ----
        text = transcribe_audio_whisper(file_path)

        # ---- Commented Out: Google SpeechRecognition ----
        # text = transcribe_audio_google(file_path)

        return [Document(page_content=text, metadata={"source": filename, "file_type": "audio"})]

    elif ext in IMAGE_EXTENSIONS:
        text = extract_text_from_image(file_path)
        return [Document(page_content=text, metadata={"source": filename, "file_type": "image"})]

    else:
        raise ValueError(f"Unsupported file extension: '{ext}'.")


def load_url(url: str, is_wikipedia: bool = False) -> List[Document]:
    """Load content from a web URL or Wikipedia article."""
    logger.info(f"Loading URL: {url} (wikipedia={is_wikipedia})")

    if is_wikipedia:
        query = url.split("/")[-1].replace("_", " ") if "/" in url else url
        loader = WikipediaLoader(query=query, load_max_docs=1)
        return loader.load()
    else:
        loader = WebBaseLoader(url)
        return loader.load()


# ============================================================
# Text Splitting (Chunking)
# ============================================================

def chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s).")
    return chunks


# ============================================================
# Vector Store (ChromaDB)
# ============================================================

def get_vectorstore():
    """Load the existing ChromaDB vector store from disk."""
    if not os.path.exists(CHROMA_PERSIST_DIR):
        return None
    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def store_documents(documents: List[Document]):
    """Chunk, embed, and store documents in ChromaDB."""
    if not documents:
        raise ValueError("No documents to store.")

    chunks = chunk_documents(documents)
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")
    return vectorstore


def retrieve_context(question: str, top_k: int = TOP_K_RESULTS) -> str:
    """
    Retrieve relevant context from ChromaDB for a given question.
    Returns a single formatted string (matching Member 2's expected context format).
    """
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "No knowledge base found. Please upload documents first."

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "No relevant information found in the knowledge base."

    context_parts = []
    for i, doc in enumerate(relevant_docs, 1):
        source = doc.metadata.get("source", "Unknown")
        context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


def get_knowledge_stats() -> dict:
    """Get statistics about the current knowledge base."""
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return {"status": "empty", "total_chunks": 0}

    collection = vectorstore._collection
    count = collection.count()
    return {"status": "ready", "total_chunks": count}


def clear_knowledge_base():
    """Delete the entire ChromaDB vector store."""
    import shutil
    if os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)
        logger.info("Knowledge base cleared.")
    return {"status": "cleared"}
