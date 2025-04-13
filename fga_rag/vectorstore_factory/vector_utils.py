from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama.embeddings import OllamaEmbeddings


def create_embeddings(name) -> Embeddings:
    if name == "bge-m3":
        return OllamaEmbeddings(model=name)
    elif name == "BAAI/bge-small-en-v1.5":
        return HuggingFaceEmbeddings(model_name=name)
    raise ValueError(f"Embedding model {name} not supported.")


def create_vectorstore(docs, embedding: Embeddings):
    return FAISS.from_documents(docs, embedding=embedding)
