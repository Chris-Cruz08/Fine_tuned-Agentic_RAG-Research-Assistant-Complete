from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
import hashlib


def stable_id(text: str) -> int:
    """Deterministic ID that stays the same across every Python process/run."""
    return int(hashlib.md5(text.encode()).hexdigest()[:16], 16) % (2**63)


def chunk_document(content: str, content_format: str, source_url: str, use_hierarchical: bool = True) -> list[dict]:
    """Create hierarchical parent/child chunks from a document.

    Parent chunks: 1500 chars, give context.
    Child chunks: 400 chars, used for precise retrieval, linked back to their parent.
    """
    if content_format == "markdown":
        parent_splitter = MarkdownTextSplitter(chunk_size=1500, chunk_overlap=150)
        child_splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50)
    else:
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

    if not use_hierarchical:
        # Flat chunking: treat everything as single-level "child" chunks (400 chars), no parent linking
        flat_chunks = child_splitter.split_text(content)
        return [
            {
                "id": f"{source_url}::flat::{idx}",
                "text": text,
                "level": "child",
                "parent_id": None,
                "chunk_index": idx,
                "source_url": source_url,
                "format": content_format,
            }
            for idx, text in enumerate(flat_chunks)
        ]

    parent_chunks = parent_splitter.split_text(content)

    all_chunks = []
    for parent_idx, parent_text in enumerate(parent_chunks):
        parent_id = f"{source_url}::parent::{parent_idx}"

        all_chunks.append({
            "id": parent_id,
            "text": parent_text,
            "level": "parent",
            "parent_id": None,
            "chunk_index": parent_idx,
            "source_url": source_url,
            "format": content_format,
        })

        child_texts = child_splitter.split_text(parent_text)
        for child_idx, child_text in enumerate(child_texts):
            all_chunks.append({
                "id": f"{parent_id}::child::{child_idx}",
                "text": child_text,
                "level": "child",
                "parent_id": parent_id,
                "chunk_index": child_idx,
                "source_url": source_url,
                "format": content_format,
            })

    return all_chunks