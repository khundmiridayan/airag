from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings

from app.vector_store import VectorStore

def answer_question(query, vector_store):
    embedder = OpenAIEmbeddings()
    query_embedding = embedder.embed_query(query)
    context_chunks = vector_store.query(query_embedding)
    context = "\n".join(context_chunks)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    prompt = f"Answer the question using the context below:\n\nContext:\n{context}\n\nQuestion: {query}"
    return llm.predict(prompt)
