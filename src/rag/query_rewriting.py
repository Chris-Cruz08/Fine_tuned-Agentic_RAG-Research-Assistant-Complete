import os
from src.models.finetuned_model import generate

REWRITE_SYSTEM_PROMPT = (
    "You rewrite user questions into concise, keyword-rich search queries optimized for "
    "vector similarity search. Strip filler words. Keep technical terms. Output ONLY the "
    "rewritten query on a single line, nothing else.\n\n"
    "Example:\n"
    "Question: How does attention work in transformer models?\n"
    "Rewritten: attention mechanism transformer architecture\n\n"
    "Example:\n"
    "Question: What are the main differences between RAG and fine-tuning approaches?\n"
    "Rewritten: RAG vs fine-tuning comparison differences"
)


def rewrite_query(question: str) -> str:
    """Rewrite a user question into an optimal search query."""
    model_id = os.environ["HF_MODEL_ID"]
    rewritten = generate(
        model_id=model_id,
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=question,
        max_new_tokens=64,
    )
    return rewritten.strip().strip('"')