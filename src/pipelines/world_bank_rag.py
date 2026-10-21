from src.ingestion.world_bank_loader import (
    load_world_bank_sections,
    load_world_bank_tables,
)
from src.retrieval.hierarchical_retriever import (
    index_sections,
    retrieve_hierarchical,
)
from src.generation.answer_generator import generate_answer


PDF_PATH = "/home/bacancy/Imran/advanced-rag-challenge/data/world_bank/GEP-Jun-2026.pdf"


def ingest_world_bank():
    sections = load_world_bank_sections(PDF_PATH)
    tables = load_world_bank_tables(PDF_PATH)
    child_count = index_sections(sections, tables)

    print(f"Sections detected: {len(sections)}")
    print(f"Table rows parsed: {len(tables)}")
    print(f"Child chunks indexed: {child_count}")

    return child_count


def ask_world_bank(question: str, top_k: int = 5) -> dict:
    matches = retrieve_hierarchical(question, top_k=top_k)
    answer = generate_answer(question, matches)

    sources = [
        {
            "source": item["metadata"]["source"],
            "chapter": item["metadata"]["chapter"],
            "section": item["metadata"]["title"],
            "page": item["metadata"]["page"],
            "kind": item["metadata"]["kind"],
        }
        for item in matches
    ]

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_sections": matches,
    }


if __name__ == "__main__":
    ingest_world_bank()

    while True:
        question = input("\nQuestion (quit to exit): ").strip()

        if question.lower() in {"quit", "exit"}:
            break

        if not question:
            continue

        result = ask_world_bank(question)

        print("\nAnswer:\n", result["answer"])
        print("\nRetrieved sections:")

        for source in result["sources"]:
            print(
                f"- {source['chapter']} > {source['section']} "
                f"(page {source['page']}, {source['kind']})"
            )