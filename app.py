"""
Streamlit UI for "Chat With Your Notes".

Upload one or more PDFs, click Process to build the vector store, then ask
questions and get answers grounded in your notes with cited source pages.

Run:
    streamlit run app.py
"""

import os

import streamlit as st

from ingest import ingest
from rag import answer

NOTES_DIR = "notes"

st.set_page_config(page_title="Chat With Your Notes", page_icon="📚")

st.title("📚 Chat With Your Notes")
st.caption(
    "Upload your study PDFs, then ask questions. Answers come only from your "
    "notes, with the source page cited."
)


# --- Step 1: upload + process ----------------------------------------------

st.subheader("1. Add your notes")

uploaded = st.file_uploader(
    "Upload PDF(s)", type="pdf", accept_multiple_files=True
)

if st.button("Process", disabled=not uploaded):
    os.makedirs(NOTES_DIR, exist_ok=True)
    notes_root = os.path.realpath(NOTES_DIR)
    for f in uploaded:
        safe_name = os.path.basename(f.name)
        dest = os.path.realpath(os.path.join(NOTES_DIR, safe_name))
        if (not safe_name or safe_name.startswith(".")
                or not dest.startswith(notes_root + os.sep)):
            st.error(f"Rejected unsafe filename: {f.name}")
            continue
        with open(dest, "wb") as out:
            out.write(f.getbuffer())

    with st.spinner("Reading, chunking and embedding your notes..."):
        ingest(NOTES_DIR, reset=True)

    st.session_state["ready"] = True
    st.success(f"Processed {len(uploaded)} file(s). Ask away below.")


# --- Step 2: ask -----------------------------------------------------------

st.subheader("2. Ask a question")

question = st.text_input("Your question", placeholder="e.g. What is overfitting?")

if st.button("Ask", disabled=not question):
    with st.spinner("Searching your notes..."):
        text, sources = answer(question)

    st.markdown("### Answer")
    st.write(text)

    if sources:
        cites = ", ".join(f"{f} (p.{p})" for f, p in sources)
        st.markdown(f"**Sources:** {cites}")
