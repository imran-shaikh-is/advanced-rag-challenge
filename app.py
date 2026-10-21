import streamlit as st

from src.pipelines.gpo_rag import ask_gpo
from src.pipelines.world_bank_rag import ask_world_bank
from src.pipelines.apple_rag import ask_apple


st.set_page_config(
    page_title="Advanced RAG Challenge",
    page_icon="📚",
)

st.title("📚 Advanced RAG Challenge")

pipeline_map = {
    "GPO Annual Report": ask_gpo,
    "World Bank Economic Report": ask_world_bank,
    "Apple iMac Environmental Report": ask_apple,
}

document = st.selectbox("Select a document", list(pipeline_map))
top_k = st.slider("Retrieved results", 1, 10, 5)

# One conversation per document, kept across reruns.
history = st.session_state.setdefault("history", {}).setdefault(document, [])


def format_source(source: dict) -> str:
    """One readable line per source, e.g. "Page 87 · table_row · Europe ..."."""
    page = source.get("page")
    parts = [f"**Page {page}**"] if page is not None else []

    if source.get("chapter"):
        parts.append(f"{source['chapter']} › {source['section']}")
    elif source.get("section"):
        parts.append(source["section"])

    kind = source.get("kind") or source.get("type")

    if kind:
        parts.append(f"`{kind}`")

    return " · ".join(parts) or str(source)


def evidence_items(result: dict) -> list[dict]:
    for key in ("retrieved_chunks", "retrieved_sections", "retrieved_records"):
        if key in result:
            return result[key]

    return []


def show_result(result: dict):
    st.markdown(result["answer"])

    with st.expander(f"Sources ({len(result['sources'])})"):
        for source in result["sources"]:
            st.markdown(f"- {format_source(source)}")

    with st.expander("Inspect retrieved evidence"):
        for rank, item in enumerate(evidence_items(result), start=1):
            metadata = item["metadata"]
            score = item.get("rerank_score")
            score_text = f" · relevance {score:.2f}" if score is not None else ""
            st.markdown(f"**{rank}. Page {metadata.get('page')}**{score_text}")
            st.text(item["text"])


for turn in history:
    with st.chat_message("user"):
        st.markdown(turn["question"])

    with st.chat_message("assistant"):
        show_result(turn["result"])

question = st.chat_input(f"Ask a question about the {document}...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating answer..."):
            try:
                result = pipeline_map[document](question, top_k=top_k)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                result = None

        if result:
            show_result(result)
            history.append({"question": question, "result": result})
