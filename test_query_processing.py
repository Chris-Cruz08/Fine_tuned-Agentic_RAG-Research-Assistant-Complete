import os
from dotenv import load_dotenv

load_dotenv()

from src.rag.query_rewriting import rewrite_query
from src.rag.query_expansion import expand_query

question = "How does attention work in transformer models?"

rewritten = rewrite_query(question)
print("Rewritten query:", rewritten)

variants = expand_query(rewritten)
print("Expansion variants:")
for v in variants:
    print(" -", v)