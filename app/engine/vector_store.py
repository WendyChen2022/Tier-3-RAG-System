"""ChromaDB vector store — local sentence-transformers embedding (no API key)."""
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from app.core.config import get_settings

settings = get_settings()


def build_vector_store(documents=None) -> Chroma:
    # Local embedding model — already cached by HybridChunker, zero extra cost
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.reranker_device},
        encode_kwargs={"normalize_embeddings": True},
    )

    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)

    store = Chroma(
        client=client,
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
    )

    if documents:
        logger.info(f"Indexing {len(documents)} document chunks")
        store.add_documents(documents)

    return store
