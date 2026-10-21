import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    )

    if not api_key:
        raise ValueError("GROQ_API_KEY .env file mein configure karo.")

    if not retrieved_chunks:
        return "Relevant context nahi mila. Document mein verify karein."

    context_parts = []

    for item in retrieved_chunks:
        metadata = item["metadata"]

        context_parts.append(
            f"[Source: {metadata['source']}, "
            f"Page: {metadata['page']}]\n{item['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document question-answering assistant. "
                    "Answer using only the supplied context. "
                    "If the context does not contain the answer, "
                    "clearly say that the information was not found. "
                    "Do not invent facts, figures, or citations. "
                    "Answer clearly and concisely."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Retrieved context:\n{context}"
                ),
            },
        ],
    )

    answer = response.choices[0].message.content
    return answer or "Model se answer generate nahi ho paya."