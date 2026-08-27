BENCHMARK_QUESTIONS = [
    {
        "question": "What is retrieval augmented generation?",
        "reference": "Retrieval augmented generation combines a retrieval system that fetches relevant external documents with a generative language model that uses that retrieved context to produce more accurate, grounded responses.",
        "relevance_keywords": ["retrieval", "generat", "external document", "knowledge base"],
    },
    {
        "question": "What are common challenges with RAG systems?",
        "reference": "Common RAG challenges include retrieval quality issues, RAG poisoning from malicious documents, scalability of vector indexes, and misalignment between retriever and generator representations.",
        "relevance_keywords": ["poison", "scalab", "retrieval quality", "hallucinat", "relevance"],
    },
    {
        "question": "How does chunking affect RAG performance?",
        "reference": "Chunking strategy affects how well retrieved passages match a query — chunk size, overlap, and hierarchical parent/child structures influence retrieval precision and the amount of context available to the generator.",
        "relevance_keywords": ["chunk", "hierarchical", "context window", "passage"],
    },
]