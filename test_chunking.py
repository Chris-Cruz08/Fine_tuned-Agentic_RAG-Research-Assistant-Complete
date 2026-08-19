from src.rag.chunking import chunk_document

sample_text = "# Title\n\nSome text here. " * 100
chunks = chunk_document(sample_text, "markdown", "https://test.com")

parents = sum(1 for x in chunks if x["level"] == "parent")
children = sum(1 for x in chunks if x["level"] == "child")

print(f"Total chunks: {len(chunks)}")
print(f"Parents: {parents}, Children: {children}")
print("Sample child chunk:")
print(chunks[1])

from src.tools.qdrant_tools import store_chunks

result = store_chunks(chunks)
print("Qdrant storage result:", result)