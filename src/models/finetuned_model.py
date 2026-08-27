import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()

BASE_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

_loaded_models = {}  # cache so we don't reload the same model twice


def load_model(adapter_id: str):
    """Load the base Qwen3-4B model in 4-bit, then apply the LoRA adapter on top."""
    if adapter_id in _loaded_models:
        return _loaded_models[adapter_id]

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(adapter_id, token=hf_token)

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=quant_config,
        device_map="auto",
        token=hf_token,
    )

    model = PeftModel.from_pretrained(base_model, adapter_id, token=hf_token)
    model.eval()

    _loaded_models[adapter_id] = (model, tokenizer)
    return model, tokenizer


def generate(model_id: str, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
    """Run a chat-style generation with thinking mode disabled."""
    model, tokenizer = load_model(model_id)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            #temperature=0.3,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()