from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_pages(pdf_path: str) -> list[dict]:
    """Extract text page by page and retain page numbers."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = []

    with fitz.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "text": text,
                    "page": page_index + 1,
                    "source": path.name,
                })

    if not pages:
        raise ValueError(
            "PDF mein extractable text nahi mila. "
            "Document scanned ho sakta hai; OCR ki zarurat hogi."
        )

    return pages


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> list[dict]:
    """Split each page separately to preserve accurate page citations."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []

    for page_data in pages:
        text_chunks = splitter.split_text(page_data["text"])

        for chunk_index, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": page_data["source"],
                    "page": page_data["page"],
                    "chunk_index": chunk_index,
                },
            })

    return chunks