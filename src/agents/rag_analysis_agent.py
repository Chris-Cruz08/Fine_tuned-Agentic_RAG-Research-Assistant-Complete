import os
from src.graph.state import ResearchState
from src.rag.query_rewriting import rewrite_query
from src.rag.query_expansion import expand_query
from src.rag.retrieval import retrieve_children, get_parent_chunks
from src.rag.rrf import reciprocal_rank_fusion
from src.rag.reranking import rerank
from src.models.finetuned_model import generate
from langsmith import traceable

ANSWER_SYSTEM_PROMPT = (
    "You are an expert AI research assistant. Answer the question based only on the "
    "provided context. Be precise, technical, and cite specific details from the context."
)


@traceable(name="answer_question_full_pipeline", run_type="chain")
def answer_question(question: str, use_query_expansion: bool = True, use_reranking: bool = True) -> dict:
    """Run the advanced RAG pipeline for one question, with optional toggles for expansion/reranking."""
    rewritten = rewrite_query(question)

    if use_query_expansion:
        variants = expand_query(rewritten)
        all_queries = [rewritten] + variants
    else:
        variants = []
        all_queries = [rewritten]

    ranked_lists = [retrieve_children(q, top_k=20) for q in all_queries]
    fused = reciprocal_rank_fusion(ranked_lists) if len(ranked_lists) > 1 else ranked_lists[0]

    if use_reranking:
        top_chunks = rerank(rewritten, fused[:20], top_n=5)
    else:
        top_chunks = fused[:5]

    parent_chunks = get_parent_chunks(top_chunks)
    if not parent_chunks:
        # Flat chunking mode has no parents - fall back to the child chunks themselves as context
        parent_chunks = top_chunks

    context = "\n\n---\n\n".join(p["text"] for p in parent_chunks)

    model_id = os.environ["HF_MODEL_ID"]
    answer = generate(
        model_id=model_id,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=f"Context: {context}\nQuestion: {question}",
        max_new_tokens=512,
    )

    return {
        "question": question,
        "answer": answer,
        "sources": [c["source_url"] for c in parent_chunks],
        "rewritten_query": rewritten,
        "expanded_queries": variants,
        "top_chunks": top_chunks,
    }


async def rag_analysis_agent(state: ResearchState) -> ResearchState:
    """Agent 3: answer every research question using the full advanced RAG pipeline."""
    questions = state.get("research_questions", [])
    if not questions:
        return {**state, "error": "No research questions provided", "current_agent": "rag_analysis_agent"}

    use_expansion = state.get("use_query_expansion", True)
    use_rerank = state.get("use_reranking", True)

    answers = [answer_question(q, use_expansion, use_rerank) for q in questions]

    return {
        **state,
        "answers": answers,
        "current_agent": "rag_analysis_agent",
        "error": None,
    }