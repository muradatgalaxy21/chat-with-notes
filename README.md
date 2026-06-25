# Chat with Notes

An ingestion and retrieval pipeline for "Chat With Your Notes" using a local vector store (ChromaDB), local embeddings (SentenceTransformers), and LLM integration.

## Setup Instructions

### 1. Prerequisites
- Python 3.10+
- Git

### 2. Clone the Repository
```bash
git clone <repository-url>
cd chat-with-notes
```

### 3. Create and Activate Virtual Environment
We use a virtual environment named `chat-with-notes` matching the project name:
```powershell
# Windows PowerShell
python -m venv chat-with-notes
.\chat-with-notes\Scripts\activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Environment Variables
Copy `.env.example` to `.env` and fill in your Groq API key:
```bash
cp .env.example .env
```

## Usage

### Ingesting PDF Notes
Place your PDF notes into the `notes/` directory, then run:
```bash
python ingest.py --notes-dir notes
```

### Querying Notes
To run a query:
```bash
python ingest.py --notes-dir notes --query "your question here"
```
