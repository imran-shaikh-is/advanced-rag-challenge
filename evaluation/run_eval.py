"""Automated evaluation runner for all three RAG pipelines.

Usage:
    python -m evaluation.run_eval                      # all documents, full RAG
    python -m evaluation.run_eval --doc apple          # one document
    python -m evaluation.run_eval --retrieval-only     # skip Groq, retrieval metrics only
    python -m evaluation.run_eval --top-k 8
    python -m evaluation.run_eval --questions-dir evaluation/drafts
    python -m evaluation.run_eval --rescore evaluation/results/<run>.json   # no Groq calls

Question file schema (evaluation/<doc>_questions.json):
    {
      "question": "...",
      "expected_answer": "reference answer (for human review)",
      "expected_keywords": ["facts", "that must appear", "in the answer"],
      "expected_pages": [12, 13],
      "category": "text | table | chart | footnote"
    }

Metrics are only computed for questions that have the matching ground truth:
    page_hit        - any retrieved result covers an expected page
    first_hit_rank  - rank (1-based) of the first result covering an expected page
    keyword_recall  - fraction of expected_keywords found in the answer
    not_found       - the model said the answer was not in the context
"""

import argparse
import csv
import json
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from src.generation.answer_generator import generate_answer
from src.pipelines.apple_rag import retrieve_apple
from src.retrieval.hierarchical_retriever import retrieve_hierarchical
from src.retrieval.vector_store import search_chunks


EVAL_DIR = Path("evaluation")
RESULTS_DIR = EVAL_DIR / "results"

DOCUMENTS = {
    "gpo": {
        "questions": "gpo_questions.json",
        "retrieve": search_chunks,
    },
    "world_bank": {
        "questions": "world_bank_questions.json",
        "retrieve": retrieve_hierarchical,
    },
    "apple": {
        "questions": "apple_questions.json",
        "retrieve": retrieve_apple,
    },
}

NOT_FOUND_MARKERS = (
    "not found",
    "not contain",
    "does not mention",
    "no information",
    "not provided",
    "not available",
    "does not include",
    "does not specify",
    "does not state",
)


def load_questions(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def page_range(metadata: dict) -> tuple[int, int]:
    # World Bank results are parent sections spanning several pages.
    start = metadata.get("page", metadata.get("start_page"))
    end = metadata.get("end_page", start)
    return int(start), int(end)


def page_metrics(retrieved: list[dict], expected_pages: list[int]) -> dict:
    if not expected_pages:
        return {"page_hit": None, "first_hit_rank": None}

    expected = set(expected_pages)

    for rank, item in enumerate(retrieved, start=1):
        start, end = page_range(item["metadata"])

        if expected & set(range(start, end + 1)):
            return {"page_hit": True, "first_hit_rank": rank}

    return {"page_hit": False, "first_hit_rank": None}


def keyword_recall(answer: str | None, keywords: list[str]) -> float | None:
    if answer is None or not keywords:
        return None

    answer_norm = normalize(answer)
    found = sum(normalize(k) in answer_norm for k in keywords)
    return found / len(keywords)


def normalize(text: str) -> str:
    # LLM answers use narrow no-break spaces, Unicode hyphens and "%"
    # where the report says "percent"; compare on a canonical form.
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[\u2010-\u2015\u2212]", "-", text)
    text = text.replace("percent", "%")
    # "thermal-module" vs "thermal module": ignore hyphens between words.
    text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
    return re.sub(r"\s+", "", text)


def is_not_found(answer: str | None) -> bool | None:
    if answer is None:
        return None

    return any(m in answer.lower() for m in NOT_FOUND_MARKERS)


def evaluate_question(
    doc: str,
    item: dict,
    top_k: int,
    retrieval_only: bool,
) -> dict:
    question = item["question"]
    retrieve = DOCUMENTS[doc]["retrieve"]

    started = time.perf_counter()
    error = None
    retrieved = []
    answer = None

    try:
        retrieved = retrieve(question, top_k=top_k)

        if not retrieval_only:
            answer = generate_answer(question, retrieved)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started

    retrieved_summary = []

    for item_retrieved in retrieved:
        metadata = item_retrieved["metadata"]
        start, end = page_range(metadata)

        retrieved_summary.append({
            "start_page": start,
            "end_page": end,
            "title": metadata.get("title"),
            "type": metadata.get("type"),
            "distance": (
                None if item_retrieved["distance"] is None
                else round(item_retrieved["distance"], 4)
            ),
            "preview": item_retrieved.get(
                "matched_child",
                item_retrieved["text"],
            )[:200],
        })

    expected_keywords = item.get("expected_keywords", [])

    return {
        "document": doc,
        "question": question,
        "category": item.get("category"),
        "expected_answer": item.get("expected_answer"),
        "expected_pages": item.get("expected_pages", []),
        "expected_keywords": expected_keywords,
        "answer": answer,
        "not_found": is_not_found(answer),
        "keyword_recall": keyword_recall(answer, expected_keywords),
        **page_metrics(retrieved, item.get("expected_pages", [])),
        "retrieved": retrieved_summary,
        "latency_s": round(latency, 2),
        "error": error,
    }


def summarize(results: list[dict]) -> dict:
    def mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    page_scored = [r for r in results if r["page_hit"] is not None]
    keyword_scored = [r for r in results if r["keyword_recall"] is not None]

    return {
        "questions": len(results),
        "errors": sum(r["error"] is not None for r in results),
        "with_expected_pages": len(page_scored),
        "page_hit_rate": mean([float(r["page_hit"]) for r in page_scored]),
        "mrr": mean([
            1 / r["first_hit_rank"] if r["first_hit_rank"] else 0.0
            for r in page_scored
        ]),
        "with_expected_keywords": len(keyword_scored),
        "avg_keyword_recall": mean([r["keyword_recall"] for r in keyword_scored]),
        "not_found_answers": sum(bool(r["not_found"]) for r in results),
        "avg_latency_s": mean([r["latency_s"] for r in results]),
    }


def save_csv(results: list[dict], path: Path):
    fields = [
        "document", "question", "category", "expected_pages",
        "retrieved_pages", "page_hit", "first_hit_rank",
        "keyword_recall", "not_found", "answer", "latency_s", "error",
    ]

    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for r in results:
            row = {k: r.get(k) for k in fields}
            row["expected_pages"] = ";".join(map(str, r["expected_pages"]))
            row["retrieved_pages"] = ";".join(
                str(x["start_page"]) if x["start_page"] == x["end_page"]
                else f"{x['start_page']}-{x['end_page']}"
                for x in r["retrieved"]
            )
            writer.writerow(row)


def print_summary(summaries: dict):
    print("\n" + "=" * 72)
    print(
        f"{'document':<12}{'qs':>4}{'err':>5}{'page_hit':>10}"
        f"{'mrr':>7}{'kw_recall':>11}{'not_found':>11}{'lat(s)':>9}"
    )
    print("-" * 72)

    def fmt(value):
        return "-" if value is None else f"{value:.2f}"

    for doc, s in summaries.items():
        print(
            f"{doc:<12}{s['questions']:>4}{s['errors']:>5}"
            f"{fmt(s['page_hit_rate']):>10}{fmt(s['mrr']):>7}"
            f"{fmt(s['avg_keyword_recall']):>11}"
            f"{s['not_found_answers']:>11}{fmt(s['avg_latency_s']):>9}"
        )

    print("=" * 72)
    print("'-' = no ground truth yet (fill expected_pages / expected_keywords).")


def rescore(path: Path, questions_dir: Path):
    """Re-score saved answers against the current question files."""
    with open(path, encoding="utf-8") as file:
        run = json.load(file)

    expected = {}

    for doc in run["run"]["documents"]:
        for item in load_questions(questions_dir / DOCUMENTS[doc]["questions"]):
            expected[(doc, item["question"])] = item

    summaries = {}

    for r in run["results"]:
        item = expected.get((r["document"], r["question"]))

        if item:
            r["expected_answer"] = item.get("expected_answer")
            r["expected_keywords"] = item.get("expected_keywords", [])
            r["expected_pages"] = item.get("expected_pages", [])

        retrieved = [
            {"metadata": {"page": x["start_page"], "end_page": x["end_page"]}}
            for x in r["retrieved"]
        ]
        r.update(page_metrics(retrieved, r["expected_pages"]))
        r["keyword_recall"] = keyword_recall(
            r["answer"],
            r["expected_keywords"],
        )
        r["not_found"] = is_not_found(r["answer"])

    for doc in run["run"]["documents"]:
        summaries[doc] = summarize(
            [r for r in run["results"] if r["document"] == doc]
        )

    run["summary"] = summaries
    base = path.with_name(f"{path.stem}_rescored")

    with open(f"{base}.json", "w", encoding="utf-8") as file:
        json.dump(run, file, indent=2, ensure_ascii=False)

    save_csv(run["results"], Path(f"{base}.csv"))

    print_summary(summaries)
    print(f"\nSaved: {base}.json")
    print(f"Saved: {base}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--doc",
        choices=[*DOCUMENTS, "all"],
        default="all",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip answer generation (no Groq calls).",
    )
    parser.add_argument(
        "--questions-dir",
        type=Path,
        default=EVAL_DIR,
        help="Folder with <doc>_questions.json files (e.g. evaluation/drafts).",
    )
    parser.add_argument(
        "--rescore",
        type=Path,
        help="Recompute answer metrics for a saved results JSON (no API calls).",
    )
    args = parser.parse_args()

    if args.rescore:
        rescore(args.rescore, args.questions_dir)
        return

    docs = list(DOCUMENTS) if args.doc == "all" else [args.doc]
    all_results = []
    summaries = {}

    for doc in docs:
        questions = load_questions(
            args.questions_dir / DOCUMENTS[doc]["questions"]
        )
        print(f"\n[{doc}] {len(questions)} questions")

        doc_results = []

        for index, item in enumerate(questions, start=1):
            result = evaluate_question(
                doc,
                item,
                args.top_k,
                args.retrieval_only,
            )
            doc_results.append(result)

            status = "ERROR" if result["error"] else (
                {True: "HIT", False: "MISS", None: "n/a"}[result["page_hit"]]
            )
            print(f"  {index:>2}. [{status:<5}] {item['question'][:70]}")

            if result["error"]:
                print(f"      {result['error']}")

        summaries[doc] = summarize(doc_results)
        all_results.extend(doc_results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode = "retrieval" if args.retrieval_only else "full"
    base = RESULTS_DIR / f"eval_{args.doc}_{mode}_{stamp}"

    with open(f"{base}.json", "w", encoding="utf-8") as file:
        json.dump(
            {
                "run": {
                    "timestamp": stamp,
                    "documents": docs,
                    "top_k": args.top_k,
                    "retrieval_only": args.retrieval_only,
                },
                "summary": summaries,
                "results": all_results,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    save_csv(all_results, Path(f"{base}.csv"))

    print_summary(summaries)
    print(f"\nSaved: {base}.json")
    print(f"Saved: {base}.csv")


if __name__ == "__main__":
    main()
