import os
from src.models.finetuned_model import generate

ANGLES = [
    "a more technical/mechanistic phrasing of this query",
    "a comparison or contrast angle related to this query",
    "a broader conceptual or applied angle related to this query",
]

EXPANSION_SYSTEM_PROMPT = (
    "You rewrite a search query into ONE alternative version, based on the angle given. "
    "Output ONLY the rewritten query on a single line. No labels, no quotes, no explanation."
)


def expand_query(query: str) -> list[str]:
    """Generate 3 additional query variants, one per angle, via separate calls for reliability."""
    model_id = os.environ["HF_MODEL_ID"]
    variants = []

    for angle in ANGLES:
        raw = generate(
            model_id=model_id,
            system_prompt=EXPANSION_SYSTEM_PROMPT,
            user_prompt=f"Original query: {query}\nAngle: {angle}\nRewritten query:",
            max_new_tokens=32,
        )
        cleaned = raw.strip().strip('"').split("\n")[0].strip()
        if cleaned and cleaned.lower() != query.lower():
            variants.append(cleaned)

    return variants if variants else [query]