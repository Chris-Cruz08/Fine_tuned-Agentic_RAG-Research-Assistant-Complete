import os
from src.models.finetuned_model import generate

EXPANSION_SYSTEM_PROMPT = (
    "You generate exactly 3 alternative search queries covering different aspects of a "
    "topic. Output ONLY the 3 queries, one per line, no numbering, no dashes, no extra text.\n\n"
    "Example:\n"
    "Query: attention mechanism transformer architecture\n"
    "Output:\n"
    "self-attention computation steps\n"
    "multi-head attention explained\n"
    "attention vs recurrent neural networks"
)


def expand_query(query: str) -> list[str]:
    """Generate 3 additional query variants covering different aspects of the topic."""
    model_id = os.environ["HF_MODEL_ID"]
    raw = generate(
        model_id=model_id,
        system_prompt=EXPANSION_SYSTEM_PROMPT,
        user_prompt=f"Query: {query}\nOutput:",
        max_new_tokens=128,
    )
    lines = [line.strip("-• ").strip() for line in raw.split("\n") if line.strip()]
    variants = [v for v in lines if v.lower() != query.lower()]
    return variants[:3] if variants else [query]