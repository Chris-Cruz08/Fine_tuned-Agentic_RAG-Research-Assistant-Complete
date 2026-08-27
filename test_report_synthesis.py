from src.agents.report_generation_agent import synthesize_report

fake_answers = [
    {
        "question": "What is self-attention?",
        "answer": "Self-attention is a mechanism that allows a model to assign different importance to different parts of the input.",
        "sources": ["https://fake2.com"],
    }
]

report = synthesize_report("transformers", fake_answers)
print(report)