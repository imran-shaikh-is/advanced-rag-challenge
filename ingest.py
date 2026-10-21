"""Build all three vector indexes in chroma_db/.

Usage:
    python ingest.py                 # all documents
    python ingest.py apple gpo       # selected documents
"""

import json
import sys

from src.ingestion.apple_loader import extract_apple_content
from src.pipelines.apple_rag import JSON_PATH, ingest_apple
from src.pipelines.gpo_rag import ingest_gpo
from src.pipelines.world_bank_rag import ingest_world_bank


def build_apple():
    # Regenerate the extracted records before indexing them.
    records = extract_apple_content("data/apple_imac/apple_imac_report.pdf")

    with open(JSON_PATH, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)

    print(f"Extracted records: {len(records)}")
    ingest_apple()


STEPS = {
    "gpo": ingest_gpo,
    "world_bank": ingest_world_bank,
    "apple": build_apple,
}


if __name__ == "__main__":
    selected = sys.argv[1:] or list(STEPS)

    for name in selected:
        if name not in STEPS:
            sys.exit(f"Unknown document '{name}'. Choose from: {', '.join(STEPS)}")

        print(f"\n== {name} ==")
        STEPS[name]()
