# CLAUDE.md

Guidance for Claude Code working in this repo.

## Project

**Chat With Notes** — RAG app. Ask questions about study PDFs, get answers grounded only in the notes, with cited source pages.

Stack: ChromaDB (local persistent vector store), SentenceTransformers `all-MiniLM-L6-v2` (local embeddings), Groq `llama-3.1-8b-instant` (LLM).

## Files

- `ingest.py` — PDF load → 500-word chunks (50 overlap) → embed → ChromaDB `notes` collection with `{source_file, page_number}` metadata. Upsert avoids dupes. CLI: `--notes-dir --reset --query --query-only`.
- `rag.py` — retrieve top-4 chunks, answer from context only (else "I couldn't find that in your notes."), dedup citations. CLI: `python rag.py "question"`.
- `app.py` — Streamlit UI. Upload PDFs → Process → ask. Path-traversal guard on uploads.
- `notes/` — source PDFs.
- `chroma_db/` — persisted vector store (git-ignored).

## Conventions

- Python 3.10+, stdlib argparse CLIs, module-level `# --- Section ---` comment banners.
- Answers must stay grounded in retrieved context — never add outside knowledge to prompts.
- Default to latest/most capable Claude models when building AI features.

## Task tracking

- `TASKS.md` is the private, git-ignored task tracker.
- **Review `TASKS.md` before writing code or discussing project work.**
- **Update `TASKS.md` whenever a task is planned, completed, or recommended.**

## Environment

- Requires `GROQ_API_KEY` in `.env`.
- Windows / PowerShell. venv named `chat-with-notes`.
