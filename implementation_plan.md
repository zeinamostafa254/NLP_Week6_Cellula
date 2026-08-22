# Project 3: Evaluator-Generator Workflow

This plan outlines the team division for the entire project and the specific implementation steps for Part 1 (Data Ingestion & UI) using Streamlit.

## Team Division (3 Members)

To efficiently build this system, the workload is divided into 3 distinct modules:

1. **Member 1: Data Ingestion & User Interface (The Frontend & Loader)**
   - Build the Streamlit interface.
   - Handle file uploads (PDF, DOCX, TXT, PPT, Code, WAV) and URLs.
   - Implement Speech-to-Text for WAV files.
   - Process data: Chunking, generating embeddings, and storing them in a Vector DB.

2. **Member 2: Core LLM Engineering (Generator & Evaluator Prompts)**
   - Design the Generator LLM to strictly answer from the retrieved context without hallucinating.
   - Design the Evaluator LLM to grade the answer based on accuracy, relevance, and completeness.
   - Manage the isolated independent memories for both models.

3. **Member 3: Orchestration, Caching & Feedback Loop (The System Architecture)**
   - Use LangChain Expression Language (LCEL) to connect the Generator and Evaluator.
   - Build the 4-iteration Feedback Loop logic.
   - Integrate Redis Caching to save embeddings and previous answers.
   - Handle logging, error management, and overall system integration.

---

## Part 1 Implementation Plan (Your Part)

You are responsible for **Member 1's tasks**. We will build the Data Ingestion pipeline and the Streamlit UI.

### Open Questions
> [!IMPORTANT]
> Please answer these questions so we can choose the right tools before writing the code:
> 1. **Embeddings Model:** Use the free local HuggingFace `BAAI/bge-m3` embedding model.
> 2. **Speech-to-Text:** For WAV files, should we use the `SpeechRecognition` library (which uses Google's free API and is very fast) or a local `Whisper` model? *(I recommend SpeechRecognition for simplicity at this stage).*
> 3. **Vector Store:** Are you okay with using `ChromaDB` as our local vector database?

### Proposed Changes

#### [NEW] [requirements.txt](file:///c:/Users/DELL/Desktop/NLP/week6/requirements.txt)
We will add the necessary dependencies:
- `streamlit`
- `langchain` and `langchain-community`
- `chromadb`
- `sentence-transformers` (if using HuggingFace)
- `pypdf`, `python-docx`, `unstructured` (for documents)
- `speechrecognition`, `pydub` (for audio)
- `wikipedia`

#### [NEW] [app.py](file:///c:/Users/DELL/Desktop/NLP/week6/app.py)
This is the main Streamlit application file. It will contain:
- A sidebar for users to upload multiple files and input URLs.
- A "Process Knowledge" button.
- The main chat interface layout (to be connected to the LLM later).

#### [NEW] [ingestion.py](file:///c:/Users/DELL/Desktop/NLP/week6/ingestion.py)
This file will handle the backend logic for data ingestion:
- Routing each file type to its correct LangChain Document Loader.
- Converting WAV audio to text.
- Splitting the text using `RecursiveCharacterTextSplitter`.
- Generating embeddings and storing them locally in a `chroma_db/` folder.

### Verification Plan
- Run the app using `streamlit run app.py`.
- Upload a test PDF and a URL.
- Verify that the app successfully processes them and creates the `chroma_db` folder without errors.
