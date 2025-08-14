import os
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_pdf(file_path):
    with open(file_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def split_text(text, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)
