import streamlit as st
from modules.extract import extract_text_from_pdf
from modules.chunk import split_text
from modules.embed_store import embed_and_store
from modules.rag_answer import summarize_report
import tempfile
import os

st.set_page_config(page_title="Medical Report Summarizer", layout="centered")

st.title("🧠 RAG-based Medical Report Summarizer (Gemini)")
st.markdown("Upload a medical report PDF to get a summary in simple, understandable language.")

uploaded_file = st.file_uploader("📄 Upload your medical report (PDF)", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info("⏳ Extracting and processing the report...")

    raw_text = extract_text_from_pdf(tmp_path)
    chunks = split_text(raw_text)
    index, model_embed, stored_chunks = embed_and_store(chunks)

    if st.button("🧾 Generate Summary"):
        with st.spinner("Generating response using Gemini..."):
            summary = summarize_report(stored_chunks)
            st.success("✅ Summary generated!")
            st.markdown("### 📋 Summary:")
            st.write(summary)
