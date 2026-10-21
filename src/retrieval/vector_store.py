import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from src.retrieval.reranker import CANDIDATES, rerank


DB_PATH = "chroma_db"
COLLECTION_NAME = "gpo_annual_report"


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    return client, collection


def index_chunks(chunks: list[dict], reset: bool = True) -> int:
    client, collection = get_collection()

    if reset:
        client.delete_collection(COLLECTION_NAME)
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            ),
            metadata={"hnsw:space": "cosine"},
        )

    batch_size = 100

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        collection.add(
            ids=[
                f"gpo-chunk-{start + i}"
                for i in range(len(batch))
            ],
            documents=[item["text"] for item in batch],
            metadatas=[item["metadata"] for item in batch],
        )

    return collection.count()


def search_chunks(question: str, top_k: int = 5) -> list[dict]:
    _, collection = get_collection()

    if collection.count() == 0:
        raise ValueError("Vector database empty hai. Pehle indexing run karo.")

    # Vector search proposes candidates; a cross-encoder picks the top_k.
    results = collection.query(
        query_texts=[question],
        n_results=min(max(CANDIDATES, top_k), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for i, document in enumerate(results["documents"][0]):
        matches.append({
            "text": document,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return rerank(question, matches)[:top_k]