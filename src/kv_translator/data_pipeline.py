import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class KVDatasetGenerator:
    """Automates extraction of physical KV cache activations from real model weights."""

    def __init__(self, src_model_id: str, tg_model_id: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        # Load tokenizers
        self.tokenizer = AutoTokenizer.from_pretrained(src_model_id)

        # Lazy initialization or direct device target loading depending on host resources
        print(f"Loading Source Activation Context: {src_model_id}")
        self.src_model = AutoModelForCausalLM.from_pretrained(
            src_model_id, torch_dtype=torch.float16, device_map=self.device
        )
        print(f"Loading Target Activation Context: {tg_model_id}")
        self.tg_model = AutoModelForCausalLM.from_pretrained(
            tg_model_id, torch_dtype=torch.float16, device_map=self.device
        )

    def extract_pair(self, text_prompt: str) -> tuple[tuple, tuple]:
        """Runs inference sequentially over shared inputs to gather target cache matrices."""
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            src_outputs = self.src_model(inputs.input_ids, use_cache=True)
            tg_outputs = self.tg_model(inputs.input_ids, use_cache=True)

        return src_outputs.past_key_values, tg_outputs.past_key_values
