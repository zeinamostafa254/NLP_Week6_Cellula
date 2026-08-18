# Evaluator-Generator AI Platform

## Project Overview

This project is an intelligent AI-powered platform based on an **Evaluator–Generator** workflow, developed for the Cellula Technologies NLP Internship (Project 3).

The platform allows users to provide various types of external knowledge (documents, source code, web pages, Wikipedia pages, and audio files), which are ingested into a Vector Database. It then employs a two-agent LLM architecture:
1. **Generator LLM**: Formulates answers based *strictly* on the retrieved external knowledge (with an isolated memory).
2. **Evaluator LLM**: Evaluates the Generator's answers for accuracy, relevance, and completeness (with an isolated memory).

A robust **Feedback Loop** ensures the Generator iteratively improves its answers up to a maximum of 4 iterations before delivering the final result to the user. The architecture utilizes **LangChain (LCEL)** for workflow orchestration and **Redis** for efficient caching.

## Project Structure

```text
week6/
├── src/
│   ├── frontend/            # Streamlit UI
│   │   └── app.py
│   ├── backend/             # FastAPI Backend
│   │   ├── api/
│   │   │   └── main.py      # FastAPI Endpoints
│   │   ├── core/
│   │   │   └── ingestion.py # Knowledge extraction, embedding, & vector store logic
│   │   └── llm/             # (For Member 2) Generator and Evaluator logic
│   └── config/              # Central configuration settings
│       └── settings.py
├── docs/                    # Documentation and architecture references
│   └── architecture.md
├── notebooks/               # Jupyter notebooks (.ipynb) for experiments and testing
├── data/                    # Temporary storage for uploaded files before processing
├── chroma_db/               # Persistent local Vector Database (Generated automatically)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd week6
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install System Dependencies (Optional but recommended):**
   - For audio processing (`Whisper` / `pydub`), ensure `ffmpeg` is installed on your system.
   - Install `redis-server` locally for caching capabilities.

## Running the Platform

Since the platform is built with a separated backend and frontend architecture, you need to run both concurrently.

### 1. Start the FastAPI Backend
```bash
uvicorn src.backend.api.main:app --reload --host 0.0.0.0 --port 8000
```
The backend API will be available at `http://localhost:8000`. You can access the automatic interactive API documentation at `http://localhost:8000/docs`.

### 2. Start the Streamlit Frontend
In a separate terminal, run:
```bash
streamlit run src/frontend/app.py
```
The User Interface will open in your default browser, usually at `http://localhost:8501`.

## Features
- **Multi-modal Data Ingestion**: Supports `.pdf`, `.docx`, `.txt`, `.ppt`, source-code, Wikipedia links, web URLs, and `.wav` audio files (via Whisper Speech-to-Text).
- **Dual LLM Architecture**: Generator and Evaluator models working in tandem.
- **Isolated Memory**: Strictly decoupled memory modules for both agents.
- **Feedback Loop Engine**: Max 4 iterations to refine answers and prevent hallucination.
- **Caching Layer**: Redis integration to cache embeddings and responses, improving latency and reducing API costs.

## Contribution
- **Part 1 (Data Ingestion & UI)**: Implemented in `src/backend/core/ingestion.py`, `src/backend/api/main.py`, and `src/frontend/app.py`.
- **Part 2 (Core LLM Engineering)**: Located in `src/backend/llm/`.
- **Part 3 (Orchestration & Feedback Loop)**: Handled within the LangChain chains and API endpoints.
