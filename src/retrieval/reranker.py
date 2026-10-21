"""Cross-encoder re-ranking of vector search candidates.

The bi-encoder (all-MiniLM-L6-v2) embeds question and passage separately,
which misses short, precise questions: "What is the global growth forecast
for 2026?" ranked regional outlooks above the page that answers it. A
cross-encoder reads question and passage together and orders a few dozen
candidates far more accurately, at roughly a second per query on CPU.
"""

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATES = 30

_model = None


def get_model() -> CrossEncoder:
    global _model

    if _model is None:
        _model = CrossEncoder(MODEL_NAME)

    return _model


def rerank(question: str, items: list[dict], key: str = "text") -> list[dict]:
    """Return items sorted by cross-encoder relevance, with "rerank_score"."""
    if not items:
        return items

    scores = get_model().predict([(question, item[key]) for item in items])

    for item, score in zip(items, scores):
        item["rerank_score"] = float(score)

    return sorted(items, key=lambda item: item["rerank_score"], reverse=True)
