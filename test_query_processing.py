import os
from dotenv import load_dotenv

load_dotenv()

from src.rag.query_rewriting import rewrite_query
from src.rag.query_expansion import expand_query

question = "What are the tradeoffs of using LoRA versus full fine-tuning for large language models?"

rewritten = rewrite_query(question)
print("Rewritten query:", rewritten)

variants = expand_query(rewritten)
print("Expansion variants:")
for v in variants:
    print(" -", v)