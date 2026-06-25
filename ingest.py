"""
Ingestion pipeline for "Chat With Your Notes".

Loads PDFs from a folder, splits each page's text into overlapping word
chunks, embeds them with a local sentence-transformers model, and stores
them in a persistent ChromaDB collection with {source_file, page_number}
metadata so answers can cite their sources.

Usage:
    # Ingest all PDFs in ./notes into the vector store
    python ingest.py --notes-dir notes

    # Ingest then run a test query (Session 1 "done" check)
    python ingest.py --notes-dir notes --query "what is gradient descent?"
"""

import argparse
import os
import re
import sys
import glob

from pypdf import PdfReader
import chromadb
from chromadb.utils import embedding_functions

# --- Config -----------------------------------------------------------------

CHUNK_SIZE = 500       # words per chunk
CHUNK_OVERLAP = 50     # words shared between consecutive chunks
DB_DIR = "chroma_db"   # persistent vector store location
COLLECTION_NAME = "notes"
EMBED_MODEL = "all-MiniLM-L6-v2"  # local, free, fast


# --- Dedup helper -----------------------------------------------------------

def normalize_chunk(text):
    """Key for detecting near-identical chunks across pages.

    Slide decks repeat the same outline/agenda slide on many pages; the only
    difference is the leading page number. Strip a leading number and collapse
    whitespace so those repeats map to one key and can be dropped — otherwise
    6+ identical outline chunks crowd real content out of the top results.
    """
    stripped = re.sub(r"^\s*\d+\s*", "", text)
    return re.sub(r"\s+", " ", stripped).strip().lower()


# --- PDF loading ------------------------------------------------------------

def load_pdf_pages(pdf_path):
    """Yield (page_number, text) for each page with extractable text.

    page_number is 1-based to match what a human sees in a PDF viewer.
    """
    reader = PdfReader(pdf_path)
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            yield i + 1, text


# --- Chunking ---------------------------------------------------------------

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of ~chunk_size words.

    Overlap keeps context from spilling across chunk boundaries so a
    sentence split in two is still retrievable from at least one chunk.
    """
    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    if step <= 0:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]
        if chunk:
            chunks.append(" ".join(chunk))
        if start + chunk_size >= len(words):
            break
    return chunks


# --- Vector store -----------------------------------------------------------

def get_collection(reset=False):
    """Return the ChromaDB collection, creating it if needed."""
    client = chromadb.PersistentClient(path=DB_DIR)
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )


def ingest(notes_dir, reset=False):
    """Load every PDF in notes_dir into the vector store."""
    pdf_paths = sorted(glob.glob(os.path.join(notes_dir, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {notes_dir!r}. Add some .pdf files and retry.")
        return

    collection = get_collection(reset=reset)

    ids, documents, metadatas = [], [], []
    seen_chunks = set()  # normalized text -> drop repeated boilerplate slides
    skipped = 0
    for pdf_path in pdf_paths:
        source_file = os.path.basename(pdf_path)
        print(f"Reading {source_file} ...")
        for page_number, page_text in load_pdf_pages(pdf_path):
            for ci, chunk in enumerate(chunk_text(page_text)):
                key = normalize_chunk(chunk)
                if not key or key in seen_chunks:
                    skipped += 1
                    continue
                seen_chunks.add(key)
                ids.append(f"{source_file}-p{page_number}-c{ci}")
                documents.append(chunk)
                metadatas.append(
                    {"source_file": source_file, "page_number": page_number}
                )

    if not documents:
        print("No extractable text found. Are these scanned/image PDFs?")
        return

    # Upsert so re-running on the same files updates instead of duplicating.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    print(f"Ingested {len(documents)} chunks from {len(pdf_paths)} PDF(s) "
          f"({skipped} duplicate/empty chunks skipped).")


# --- Test query (Session 1 done check) --------------------------------------

def query(question, n_results=8):
    """Print the top matching chunks with their source file + page."""
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    if not docs:
        print("No results. Did you ingest any PDFs yet?")
        return

    print(f"\nTop {len(docs)} chunks for: {question!r}\n")
    for rank, (doc, meta) in enumerate(zip(docs, metas), start=1):
        snippet = doc[:300].replace("\n", " ")
        print(f"[{rank}] {meta['source_file']} (page {meta['page_number']})")
        print(f"    {snippet}...\n")


# --- CLI --------------------------------------------------------------------

def main():
    # PDFs often carry symbol-font glyphs that crash the default Windows
    # cp1252 console. Force UTF-8 and replace anything unencodable.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB.")
    parser.add_argument("--notes-dir", default="notes",
                        help="Folder containing PDF files (default: notes)")
    parser.add_argument("--reset", action="store_true",
                        help="Delete and rebuild the collection from scratch")
    parser.add_argument("--query",
                        help="Run a test query after ingestion")
    parser.add_argument("--query-only", action="store_true",
                        help="Skip ingestion, just run --query")
    args = parser.parse_args()

    if not args.query_only:
        ingest(args.notes_dir, reset=args.reset)

    if args.query:
        query(args.query)


if __name__ == "__main__":
    main()
