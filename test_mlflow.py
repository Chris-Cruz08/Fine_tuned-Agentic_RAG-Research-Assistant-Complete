from src.evaluation.benchmark_runner import run_full_benchmark_with_logging

retrieval, generation = run_full_benchmark_with_logging()
print("Logged to MLflow successfully.")
print("Retrieval:", retrieval)