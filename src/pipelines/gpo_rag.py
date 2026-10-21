from src.ingestion.text_loader import load_pdf_pages, chunk_pages
from src.retrieval.vector_store import index_chunks, search_chunks
from src.generation.answer_generator import generate_answer


PDF_PATH = "data/gpo/GPO-Anual-report.pdf"


def ingest_gpo() -> int:
    pages = load_pdf_pages(PDF_PATH)
    chunks = chunk_pages(pages)

    count = index_chunks(chunks, reset=True)

    print(f"Extracted pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Indexed chunks: {count}")

    return count


def ask_gpo(question: str, top_k: int = 5) -> dict:
    retrieved = search_chunks(question, top_k=top_k)
    answer = generate_answer(question, retrieved)

    sources = []
    seen = set()

    for item in retrieved:
        metadata = item["metadata"]
        key = (metadata["source"], metadata["page"])

        if key not in seen:
            seen.add(key)
            sources.append({
                "source": metadata["source"],
                "page": metadata["page"],
            })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": retrieved,
    }


if __name__ == "__main__":
    ingest_gpo()

    while True:
        question = input("\nQuestion (quit to exit): ").strip()

        if question.lower() in {"quit", "exit"}:
            break

        if not question:
            continue

        result = ask_gpo(question)

        print("\nAnswer:\n", result["answer"])
        print("\nSources:")

        for source in result["sources"]:
            print(f"- {source['source']}, page {source['page']}")