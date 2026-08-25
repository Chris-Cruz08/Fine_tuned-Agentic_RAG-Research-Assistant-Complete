import os
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness, context_precision, answer_relevancy
from ragas import EvaluationDataset
from datasets import Dataset

from src.models.finetuned_model import load_model
from src.rag.embeddings import embedding_model


def get_ragas_llm():
    """Wrap the already-loaded fine-tuned model as a Ragas-compatible judge LLM."""
    model_id = os.environ["HF_MODEL_ID"]
    model, tokenizer = load_model(model_id)

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
    )
    langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return LangchainLLMWrapper(langchain_llm)


def get_ragas_embeddings():
    """Wrap the same bge-small embedding model used for retrieval, for Ragas metrics that need embeddings."""
    return LangchainEmbeddingsWrapper(embedding_model)


def evaluate_generation(question: str, answer: str, contexts: list[str]) -> dict:
    """Run Ragas faithfulness, context precision, and answer relevancy on one Q/A pair."""
    ragas_llm = get_ragas_llm()
    ragas_embeddings = get_ragas_embeddings()

    dataset = Dataset.from_dict({
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts],
    })
    eval_dataset = EvaluationDataset.from_hf_dataset(dataset)

    results = ragas_evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return results.to_pandas().iloc[0].to_dict()