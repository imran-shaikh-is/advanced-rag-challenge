import chromadb

from langchain_text_splitters import RecursiveCharacterTextSplitter
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)

from src.retrieval.reranker import CANDIDATES, rerank


DB_PATH = "chroma_db"
PARENT_COLLECTION = "world_bank_parents"
CHILD_COLLECTION = "world_bank_children"
SOURCE = "world_bank_report.pdf"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)


def get_collections():
    client = chromadb.PersistentClient(path=DB_PATH)

    parents = client.get_or_create_collection(
        PARENT_COLLECTION,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    children = client.get_or_create_collection(
        CHILD_COLLECTION,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    return client, parents, children


def build_children(
    sections: list[dict],
    table_records: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Split sections into page-level child chunks; one child per table row.

    Each child is prefixed with "chapter > section" so the embedding knows
    where the passage sits, and keeps the exact page it came from.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parents = []
    children = []

    for section_index, section in enumerate(sections):
        parent_id = f"wb-parent-{section_index}"
        heading = f"{section['chapter']} > {section['title']}"
        parents.append({
            "id": parent_id,
            "text": section["text"],
            "metadata": {
                "chapter": section["chapter"][:500],
                "title": section["title"][:500],
                "start_page": section["start_page"],
                "end_page": section["end_page"],
            },
        })

        seq = 0

        for page in section["pages"]:
            for text in splitter.split_text(page["text"]):
                children.append({
                    "id": f"{parent_id}-child-{seq}",
                    "text": f"[{heading}]\n{text}",
                    "metadata": {
                        "parent_id": parent_id,
                        "seq": seq,
                        "kind": "text",
                        "chapter": section["chapter"][:500],
                        "title": section["title"][:500],
                        "page": page["page"],
                    },
                })
                seq += 1

    tables = {}

    for record in table_records:
        key = (record["title"], record["page"])
        tables.setdefault(key, []).append(record)

    for table_index, ((title, page), rows) in enumerate(tables.items()):
        parent_id = f"wb-table-{table_index}"
        parents.append({
            "id": parent_id,
            "text": "\n".join(r["text"] for r in rows),
            "metadata": {
                "chapter": rows[0]["chapter"][:500],
                "title": title[:500],
                "start_page": page,
                "end_page": page,
            },
        })

        for seq, row in enumerate(rows):
            children.append({
                "id": f"{parent_id}-row-{seq}",
                "text": row["text"],
                "metadata": {
                    "parent_id": parent_id,
                    "seq": seq,
                    "kind": "table_row",
                    "chapter": row["chapter"][:500],
                    "title": title[:500],
                    "page": page,
                },
            })

    return parents, children


def index_sections(
    sections: list[dict],
    table_records: list[dict] | None = None,
) -> int:
    client, _, _ = get_collections()

    # Rebuild both collections so stale parent-child links are removed.
    client.delete_collection(PARENT_COLLECTION)
    client.delete_collection(CHILD_COLLECTION)
    _, parents, children = get_collections()

    parent_items, child_items = build_children(
        sections,
        table_records or [],
    )

    batch_size = 100

    for collection, items in (
        (parents, parent_items),
        (children, child_items),
    ):
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            collection.add(
                ids=[item["id"] for item in batch],
                documents=[item["text"] for item in batch],
                metadatas=[item["metadata"] for item in batch],
            )

    return children.count()


def expand_window(children, metadata: dict, child_text: str, window: int):
    """Return the matched child with its neighbours from the same parent."""
    if metadata["kind"] == "table_row" or window == 0:
        return child_text

    parent_id = metadata["parent_id"]
    seq = metadata["seq"]
    ids = [
        f"{parent_id}-child-{i}"
        for i in range(seq - window, seq + window + 1)
        if i >= 0
    ]
    found = children.get(ids=ids, include=["documents", "metadatas"])
    by_seq = sorted(
        zip(found["metadatas"], found["documents"]),
        key=lambda pair: pair[0]["seq"],
    )

    # Neighbours repeat the "[chapter > section]" prefix; keep it once.
    parts = [by_seq[0][1]] + [
        text.split("\n", 1)[-1] for _, text in by_seq[1:]
    ]
    return "\n".join(parts)


def retrieve_hierarchical(
    question: str,
    top_k: int = 5,
    context: str = "window",
    window: int = 1,
    max_per_parent: int = 2,
) -> list[dict]:
    """Search child chunks, then expand each hit for the LLM.

    context="window": matched child plus `window` neighbours on each side
    (compact, keeps the exact page). context="parent": the whole section.
    At most `max_per_parent` hits come from one section or table, so a
    single forecast table cannot fill every slot.

    Vector search proposes candidates; a cross-encoder re-ranks them.
    """
    _, parents, children = get_collections()

    if children.count() == 0:
        raise ValueError(
            "World Bank index empty hai. Pehle indexing run karo."
        )

    results = children.query(
        query_texts=[question],
        n_results=min(max(CANDIDATES, top_k * 5), children.count()),
        include=["documents", "metadatas", "distances"],
    )
    candidates = rerank(question, [
        {"text": text, "metadata": metadata, "distance": distance}
        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
    ])

    matches = []
    per_parent = {}

    for candidate in candidates:
        child_text = candidate["text"]
        metadata = candidate["metadata"]
        parent_id = metadata["parent_id"]

        if per_parent.get(parent_id, 0) >= max_per_parent:
            continue

        per_parent[parent_id] = per_parent.get(parent_id, 0) + 1

        if context == "parent":
            parent = parents.get(
                ids=[metadata["parent_id"]],
                include=["documents", "metadatas"],
            )
            text = parent["documents"][0]
        else:
            text = expand_window(children, metadata, child_text, window)

        matches.append({
            "text": text,
            "matched_child": child_text,
            "metadata": {
                "source": SOURCE,
                "page": metadata["page"],
                "chapter": metadata["chapter"],
                "title": metadata["title"],
                "kind": metadata["kind"],
            },
            "distance": candidate["distance"],
            "rerank_score": candidate["rerank_score"],
        })

        if len(matches) == top_k:
            break

    return matches
