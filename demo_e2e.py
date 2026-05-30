import torch
from kv_translator.config import LLAMA_3_2_3B, LLAMA_3_70B
from kv_translator.pipeline import FullKVCacheTranslator
from kv_translator.registry import CrossModelKVRegistry


def simulate_real_world_execution():
    print("=" * 60)
    print("🚀 RUNNING LOCAL END-TO-END KV-TRANSLATOR PARITY CHECK")
    print("=" * 60)

    # 1. Instantiate our core structural translation pipeline
    print("\n[1/4] Initializing Translator Weights...")
    translator = FullKVCacheTranslator(LLAMA_3_2_3B, LLAMA_3_70B)

    # 2. Simulate raw extraction (What PR 6 achieves via models)
    print("[2/4] Generating Simulated Source Model (3B) KV Cache Tensors...")
    batch_size = 1
    seq_len = 32

    # Mimicking Hugging Face output format: tuple of layers containing (Key, Value)
    # Inside Hugging Face, shape format is explicitly [B, H, T, D]
    mock_hf_source_past_key_values = tuple(
        (
            torch.randn(
                batch_size, LLAMA_3_2_3B.num_kv_heads, seq_len, LLAMA_3_2_3B.head_dim
            ),
            torch.randn(
                batch_size, LLAMA_3_2_3B.num_kv_heads, seq_len, LLAMA_3_2_3B.head_dim
            ),
        )
        for _ in range(LLAMA_3_2_3B.num_layers)
    )
    print(
        f"      -> Input Source Shape (per layer): {mock_hf_source_past_key_values[0][0].shape} [B, H, T, D]"
    )
    print(f"      -> Input Depth: {len(mock_hf_source_past_key_values)} Source Layers")

    # 3. Process data format adaptation and tensor mapping (What PR 5 achieves)
    print("\n[3/4] Triggering CrossModelKVRegistry Interface Layer...")
    translated_hf_cache = CrossModelKVRegistry.translate(
        source_cache=mock_hf_source_past_key_values,
        src_cfg=LLAMA_3_2_3B,
        tg_cfg=LLAMA_3_70B,
        translator_model=translator,
    )

    # 4. Assert structural conversion metrics
    print("\n[4/4] Evaluating Output Structural Parity...")
    print(f"      -> Translated Output Depth: {len(translated_hf_cache)} Target Layers")
    print(
        f"      -> Target Key Shape (Layer 0):  {translated_hf_cache[0][0].shape} [B, H, T, D]"
    )
    print(
        f"      -> Target Value Shape (Layer 0): {translated_hf_cache[0][1].shape} [B, H, T, D]"
    )

    # Run verification asserts
    assert len(translated_hf_cache) == LLAMA_3_70B.num_layers, "Depth mapping error!"
    assert translated_hf_cache[0][0].shape == (
        batch_size,
        LLAMA_3_70B.num_kv_heads,
        seq_len,
        LLAMA_3_70B.head_dim,
    ), "Shape expansion error!"

    print("\n" + "=" * 60)
    print(
        "✅ SUCCESS: E2E TENSOR HOOKS VALIDATED! CORE ARCHITECTURE SKELETON OPERATIONAL."
    )
    print("=" * 60)


if __name__ == "__main__":
    simulate_real_world_execution()
