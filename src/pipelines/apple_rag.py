import json

import chromadb
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from src.generation.answer_generator import generate_answer
from src.retrieval.reranker import CANDIDATES, rerank


DB_PATH = "chroma_db"
COLLECTION_NAME = "apple_document_records"
JSON_PATH = "data/apple_imac/extracted_records.json"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def ingest_apple():
    with open(JSON_PATH, encoding="utf-8") as file:
        records = json.load(file)

    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        if "does not exist" not in str(e).lower():
            raise

    collection = client.create_collection(
        COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    for start in range(0, len(records), 100):
        batch = records[start:start + 100]

        collection.add(
            ids=[r["id"] for r in batch],
            # Embed the record itself; return it with its linked endnotes.
            embeddings=embedding_function(
                [r.get("content", r["text"]) for r in batch]
            ),
            documents=[r["text"] for r in batch],
            metadatas=[
                {"type": r["type"], "page": r["page"]}
                for r in batch
            ],
        )

    print(f"Indexed {collection.count()} records.")


def retrieve_apple(question: str, top_k: int = 5) -> list[dict]:
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(
        COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    if collection.count() == 0:
        raise ValueError("Apple index empty hai. Pehle ingest karo.")

    # Rows of one table share the same document (the whole table), so
    # over-fetch and keep each document once; then a cross-encoder picks
    # the top_k.
    result = collection.query(
        query_texts=[question],
        n_results=min(max(CANDIDATES, top_k * 3), collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    retrieved = []
    seen = set()

    for i, text in enumerate(result["documents"][0]):
        metadata = result["metadatas"][0][i]

        if text in seen:
            continue

        seen.add(text)

        retrieved.append({
            "text": text,
            "metadata": {
                "source": "apple_imac_report.pdf",
                "page": metadata["page"],
                "type": metadata["type"],
            },
            "distance": result["distances"][0][i],
        })

    return rerank(question, retrieved)[:top_k]


def ask_apple(question: str, top_k: int = 5) -> dict:
    retrieved = retrieve_apple(question, top_k=top_k)
    answer = generate_answer(question, retrieved)

    return {
        "answer": answer,
        "sources": [
            {
                "page": r["metadata"]["page"],
                "type": r["metadata"]["type"],
            }
            for r in retrieved
        ],
        "retrieved_records": retrieved,
    }


if __name__ == "__main__":
    ingest_apple()

    while True:
        question = input("\nQuestion (quit to exit): ").strip()

        if question.lower() in {"quit", "exit"}:
            break

        if not question:
            continue

        result = ask_apple(question)

        print("\nAnswer:\n", result["answer"])
        print("\nSources:")

        for source in result["sources"]:
            print(f"- Page {source['page']} ({source['type']})")