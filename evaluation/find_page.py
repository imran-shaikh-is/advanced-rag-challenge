"""Find which PDF pages contain a phrase, to fill expected_pages.

Page numbers match the pipelines (PDF page index, starting at 1),
not the page numbers printed on the report itself.

Usage:
    python -m evaluation.find_page apple "recycled aluminum"
    python -m evaluation.find_page world_bank "global growth"
"""

import sys

import fitz


PDFS = {
    "gpo": "data/gpo/GPO-Anual-report.pdf",
    "world_bank": "data/world_bank/GEP-Jun-2026.pdf",
    "apple": "data/apple_imac/apple_imac_report.pdf",
}


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in PDFS:
        print(__doc__)
        sys.exit(1)

    doc, phrase = sys.argv[1], " ".join(sys.argv[2:])
    phrase_lower = phrase.lower()
    hits = 0

    with fitz.open(PDFS[doc]) as pdf:
        for page_index, page in enumerate(pdf):
            text = " ".join(page.get_text("text").split())
            position = text.lower().find(phrase_lower)

            if position == -1:
                continue

            hits += 1
            start = max(0, position - 100)
            snippet = text[start:position + len(phrase) + 100]
            print(f"Page {page_index + 1}: ...{snippet}...\n")

    print(f"'{phrase}' found on {hits} page(s).")


if __name__ == "__main__":
    main()
