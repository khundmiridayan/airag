from langchain.embeddings import OpenAIEmbeddings

def generate_embeddings(text_chunks):
    embedder = OpenAIEmbeddings()
    return embedder.embed_documents(text_chunks)
