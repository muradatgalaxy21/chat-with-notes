"""
Answer generation for "Chat With Your Notes".

Takes a question, retrieves the most relevant chunks from the ChromaDB
collection built by ingest.py, and asks a Groq-hosted LLM to answer using
ONLY that context. Every answer is followed by its source pages so the
result is verifiable.

Usage:
    python rag.py "what is gradient descent?"
"""

import os
import sys

from dotenv import load_dotenv
from groq import Groq

from ingest import get_collection

# --- Config -----------------------------------------------------------------

load_dotenv()

# Current, fast, free-tier Groq model. Swap if Groq retires it.
LLM_MODEL = "llama-3.3-70b-versatile"
N_RESULTS = 4

SYSTEM_PROMPT = (
    "You are a study assistant. Answer the question using ONLY the context "
    "provided below. If the answer is not in the context, say exactly: "
    "\"I couldn't find that in your notes.\" Do not use outside knowledge. "
    "Be concise."
)


# --- Retrieval --------------------------------------------------------------

def retrieve(question, n_results=N_RESULTS):
    """Return (documents, metadatas) for the top matching chunks."""
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=n_results)
    return results["documents"][0], results["metadatas"][0]


def build_context(docs, metas):
    """Format retrieved chunks into a numbered, citable context block."""
    blocks = []
    for doc, meta in zip(docs, metas):
        tag = f"[{meta['source_file']} p.{meta['page_number']}]"
        blocks.append(f"{tag}\n{doc}")
    return "\n\n".join(blocks)


# --- Answer generation ------------------------------------------------------

def answer(question, n_results=N_RESULTS):
    """Return (answer_text, sources) where sources is a list of (file, page)."""
    docs, metas = retrieve(question, n_results=n_results)
    if not docs:
        return "I couldn't find that in your notes.", []

    context = build_context(docs, metas)
    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content.strip()

    # No sources when the model couldn't answer from the notes.
    if text.startswith("I couldn't find that in your notes"):
        return text, []

    # Dedupe sources, keep order.
    seen, sources = set(), []
    for meta in metas:
        key = (meta["source_file"], meta["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append(key)

    return text, sources


# --- CLI --------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print('Usage: python rag.py "your question"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    text, sources = answer(question)

    print(f"\n{text}\n")
    if sources:
        cites = ", ".join(f"{f} (p.{p})" for f, p in sources)
        print(f"Sources: {cites}")


if __name__ == "__main__":
    main()
