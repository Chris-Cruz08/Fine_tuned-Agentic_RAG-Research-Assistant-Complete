import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.metrics import AnswerRelevancy
from ragas import EvaluationDataset
from datasets import Dataset

from src.rag.embeddings import embedding_model

load_dotenv()

answer_relevancy = AnswerRelevancy(strictness=1)

def get_ragas_llm():
    """Use Groq's free tier (gpt-oss, open-weight) as the fast judge LLM for Ragas."""
    groq_llm = ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.environ["GROQ_API_KEY"],
        temperature=0,
    )
    return LangchainLLMWrapper(groq_llm)


def get_ragas_embeddings():
    """Local open-source embeddings — no need for a cloud call here."""
    return LangchainEmbeddingsWrapper(embedding_model)


def evaluate_generation(question: str, answer: str, contexts: list[str], reference: str) -> dict:
    """Run Ragas faithfulness, context precision, and answer relevancy on one Q/A pair."""
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    dataset = Dataset.from_dict({
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
        "reference": [reference],
    })
    eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

    results = ragas_evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=120, max_retries=2, max_workers=1),
    )

    return results.to_pandas().iloc[0].to_dict()