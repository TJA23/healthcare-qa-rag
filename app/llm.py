from typing import Iterable

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL
from app.vectorstore import Hit


SYSTEM_PROMPT = (
    "You are a careful medical information assistant. Answer the user's question "
    "using ONLY the context provided. If the context does not contain the answer, "
    "say so plainly. Cite sources by their [Source: title] tag. Keep answers concise "
    "and clear. Always end with: 'This is general information, not medical advice. "
    "Consult a qualified healthcare professional.'"
)


def _format_context(hits: Iterable[Hit]) -> str:
    blocks = []
    for hit in hits:
        blocks.append(f"[Source: {hit.title}]\n{hit.text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, hits: list[Hit]) -> str:
    context = _format_context(hits)
    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above. Cite sources inline."
    )

    if not GROQ_API_KEY:
        snippet = "\n\n".join(f"• {h.title}: {h.text[:280]}..." for h in hits)
        return (
            "[Groq API key not set — returning retrieved context without LLM synthesis.]\n\n"
            "Top retrieved passages:\n\n"
            f"{snippet}\n\n"
            "This is general information, not medical advice. "
            "Consult a qualified healthcare professional."
        )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=600,
    )
    return response.choices[0].message.content.strip()
