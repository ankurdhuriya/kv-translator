# Cross-Model KV Cache Translator (🔀 kv-translator)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Astral uv](https://img.shields.io/badge/managed%20by-uv-purple.svg)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)

**kv-translator** is a model-agnostic infrastructure library designed to translate Key-Value (KV) caches between structurally disparate Large Language Models (LLMs) on the fly. 

By mapping latent spaces and resolving architectural mismatches (differing attention heads, layer counts, and positional encodings), this framework skips the computationally expensive **prefill phase** during model handoffs (e.g., routing from a small draft model like Llama-3.2-3B or Qwen-2.5-1.5B to a massive model like Llama-3-70B or Qwen-2.5-72B). This reduces **Time-to-First-Token (TTFT)** by up to **2-4x** in cascading inference, distributed speculative decoding, and routing pipelines.

---

## 🚀 Key Features

- **Truly Model-Agnostic:** Seamlessly translates across different model families (Llama, Qwen, Mistral, Gemma) by flattening configurations into raw latent feature vectors.
- **Dynamic GQA/MQA Resolving:** Automatically bridges dimension mismatches between models with different numbers of attention heads (e.g., translating from 2 KV heads to 8 KV heads).
- **Position-Aware Inversion:** Decouples sequence-specific Rotary Positional Embeddings (RoPE) using inverse rotation matrices before projection, preventing spatial data corruption.
- **Hugging Face Integrated:** Zero-overhead entry points designed to handle standard `past_key_values` formats in 3 lines of code.
- **Built with `uv`:** Blazing-fast setup, deterministic lockfiles, and integrated linting with Ruff.

---

## 🛠️ Installation

Manage your environment and install dependencies cleanly using [Astral `uv`](https://github.com/astral-sh/uv):

```sh
# Clone the repository
git clone [https://github.com/ankurdhuriya/kv-translator.git](https://github.com/ankurdhuriya/kv-translator.git)
cd kv-translator

# Sync and install the project in editable mode
uv sync
````

Alternatively, install directly via `pip` once published:

Bash

```sh
pip install kv-translator
```

## 💻 Quick Start (Hugging Face Integration)

Below is an end-to-end example showing how to extract a real cache from a small model, pass it through the registry wrapper, and instantly resume decoding inside a larger model—bypassing its prefill context execution entirely.

Python

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_translator import (
    LLAMA_3_2_3B, 
    LLAMA_3_70B, 
    FullKVCacheTranslator, 
    CrossModelKVRegistry
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Initialize Global Configs & Shared Tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
prompt = "Explain the mathematical constraints of PagedAttention layers in modern LLM serving engines:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# 2. Extract Cache from Source Model (Prefill Phase)
src_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)
with torch.no_grad():
    src_outputs = src_model(inputs.input_ids, use_cache=True)

# 3. Instantiate and Load the Translation Weights Bridge
translator = FullKVCacheTranslator(LLAMA_3_2_3B, LLAMA_3_70B).to(device).to(torch.float16)
# translator.load_state_dict(torch.load("llama_3b_to_70b_weights.pt"))

# 4. Seamlessly Translate the Cache Array via the Registry API
translated_past_key_values = CrossModelKVRegistry.translate(
    source_cache=src_outputs.past_key_values,
    src_cfg=LLAMA_3_2_3B,
    tg_cfg=LLAMA_3_70B,
    translator_model=translator
)

# 5. Inject Directly into Target Model and Resume Instant Generation
tg_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-70B-Instruct", 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# Trick target into continuing decoding using a dummy next-token container
next_token_trigger = torch.tensor([[tokenizer.eos_token_id]], device=device)
attention_mask = torch.ones((inputs.input_ids.shape[0], inputs.input_ids.shape[1] + 1), device=device)

with torch.no_grad():
    tg_outputs = tg_model.generate(
        input_ids=next_token_trigger,
        past_key_values=translated_past_key_values,
        attention_mask=attention_mask,
        max_new_tokens=50
    )

print(tokenizer.decode(tg_outputs[0], skip_special_tokens=True))
```

## 🏋️ Knowledge Distillation Pipeline

To train the mapping layers for a custom model configuration block (e.g., mapping a `QWEN_2_5_1_5B` source to a `QWEN_2_5_72B` target):

1. **Configure Model Properties:** Add your custom dimensions to `src/kv_translator/config.py`.
    
2. **Run the Distillation Harness:** Execute training over real text pairs using the optimization framework:
    
    Bash
    
    ```
    uv run python src/kv_translator/trainer.py
    ```
    

## 🧪 Testing

The repository maintains an automated unit-testing baseline with zero global network overhead. Run the validation checks locally:

Bash

```
uv run pytest tests/ -v
```

## 🤝 Contributing

Contributions are welcome! Please branch from `main`, maintain atomized commit schemas, and ensure that all incoming code additions pass the accompanying Ruff formatting and PyTest CI pipelines before request merge.

## 📄 License

Distributed under the Apache 2.0 License. See `LICENSE` for details.