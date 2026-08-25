from src.evaluation.generation_metrics import evaluate_generation

question = "What is self-attention?"
answer = "Self-attention is a mechanism that allows a model to assign different importance to different parts of the input."
contexts = ["Self-attention is a mechanism in transformers where each token attends to every other token in the sequence to compute contextual representations."]

scores = evaluate_generation(question, answer, contexts)
print("Generation metrics:", scores)