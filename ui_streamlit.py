import streamlit as st
from app.document_loader import load_pdf, split_text
from app.embedder import generate_embeddings
from app.vector_store import VectorStore
from app.rag_pipeline import answer_question

st.title("📄🔍 RAG Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
question = st.text_input("Ask a question about the document")

if uploaded_file and question:
    text = load_pdf(uploaded_file)
    chunks = split_text(text)
    embeddings = generate_embeddings(chunks)

    vs = VectorStore()
    vs.build_index(embeddings, chunks)
    answer = answer_question(question, vs)

    st.subheader("Answer")
    st.write(answer)
