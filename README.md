# IITB Insti-Assist 🎓

A Retrieval-Augmented Generation (RAG) assistant that answers questions about IIT Bombay academics — grounded strictly in real institute documents. It refuses to answer anything outside its knowledge base, rather than guessing or hallucinating.

## Live Demo
🔗 [Try it here](https://iitb-assistant-ccjwcxd6gtnndobjhdhkvf.streamlit.app/)

## Chosen Scope: Academic Assistant

This assistant focuses on **academics** — the academic calendar, curriculum structure, examination rules, and disciplinary procedures. This scope was chosen because this would allow me to use the UG rulebook which is a very dense and rule-heavy document which I could use to provide more context to the RAG agent.

## Data Sources

Five official IIT Bombay documents were used:
- Academic Calendar (2026–27)
- UG Curriculum — Mechanical Engineering
- Procedures document (July 2015)
- Punishments/disciplinary document (July 2015)
- UG Rulebook

## How It Works

1. **Ingestion** — PDFs are parsed with PyPDF2 and split into overlapping chunks (500 characters, 50-character overlap) using a custom sentence-aware chunking function.
2. **Embedding** — Each chunk is embedded using `all-mpnet-base-v2` (via `sentence-transformers`) and stored in a persistent **ChromaDB** collection.
3. **Retrieval** — At query time, the top-k most relevant chunks are retrieved via semantic search.
4. **Generation** — Retrieved chunks are injected into a prompt sent to **Groq's `openai/gpt-oss-120b`**, which is explicitly instructed to answer only from the provided context — and to say "I cannot answer this based on the provided context" otherwise.
5. **Source display** — Every answer is shown alongside the exact document and chunk it was pulled from, so answers are auditable.

## Tech Stack

| Component | Tool |
|---|---|
| Embeddings | `sentence-transformers` (`all-mpnet-base-v2`) |
| Vector DB | ChromaDB (persistent, local) |
| LLM | Groq API (`openai/gpt-oss-120b`) |
| UI | Streamlit |
| PDF parsing | PyPDF2 |

## Setup Instructions

### Option 1: Use the live deployed app (no setup needed)
Just open the link above — it's already running with the knowledge base pre-loaded.

### Option 2: Run it locally

1. **Clone the repo**

   git clone https://github.com/your-username/iitb-assistant.git
   cd iitb-assistant

2. **Install dependencies**

   pip install -r requirements.txt

3. **Set up your Groq API key**

   Copy the example env file and add your own key:

   cp .env.example .env

   Then open `.env` and replace the placeholder with your actual Groq API key:

   GROQ_API_KEY=your-key-here

   Get a free key at console.groq.com/keys.

4. **Run the app**

   streamlit run app.py

   This opens the app in your browser at http://localhost:8501.

Note: the `chroma_db` folder (containing pre-embedded documents) is already included in this repo, so the app will start up instantly with the knowledge base ready. If it's ever missing or deleted, the app will automatically rebuild it from the PDFs in the `documents/` folder on first run.

## Known Limitations

- PDF text extraction quality depends on how cleanly the source PDF is formatted — tables and multi-column layouts can extract with garbled spacing.
- Chunking is sentence-based with a fixed size/overlap; it doesn't account for document structure (headings, sections), so a chunk can occasionally split a rule across two pieces.
- Retrieval uses a fixed top-k (2) with no re-ranking step — a more sophisticated pipeline might re-rank candidates or dynamically adjust k based on query complexity.
- Only one embedding model was tried; results weren't benchmarked against alternatives.
- The scope is limited to five documents — broader coverage (e.g. hostel life, clubs) is out of scope for this version.

## What I'd Improve With More Time
- I would make changes to the current project to allow the RAG agent to have conversational memory.
- Add a re-ranking step after initial retrieval to improve precision on ambiguous queries.
- Chunk by document structure (headings/sections) instead of a fixed character count.
- Add a feedback mechanism so users can flag incorrect or unhelpful answers.
- Expand the document set to cover more of the "General Insti Assistant" scope.
