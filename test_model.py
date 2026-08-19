import os
from dotenv import load_dotenv
from src.models.finetuned_model import generate

load_dotenv()

response = generate(
    model_id=os.environ["HF_MODEL_ID"],
    system_prompt="You are an expert AI research assistant. Answer the question based only on the provided context. Be precise, technical, and cite specific details from the context.",
    user_prompt="Context: Transformers use self-attention to weigh the importance of different words in a sequence.\nQuestion: What mechanism do transformers use?",
)
print("Model response:", response)