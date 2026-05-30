import torch
from kv_translator.config import LLAMA_3_2_3B, LLAMA_3_70B
from kv_translator.pipeline import FullKVCacheTranslator
from kv_translator.registry import CrossModelKVRegistry


def test_registry_huggingface_adapter_bridge():
    # Setup mapping dependencies
    translator = FullKVCacheTranslator(LLAMA_3_2_3B, LLAMA_3_70B)

    # Replicate standard nested Hugging Face cache structure: [24 Layers][Key/Value Tuple][B, H, T, D]
    mock_hf_past_key_values = tuple(
        (torch.randn(1, LLAMA_3_2_3B.num_kv_heads, 32, LLAMA_3_2_3B.head_dim),
         torch.randn(1, LLAMA_3_2_3B.num_kv_heads, 32, LLAMA_3_2_3B.head_dim))
        for _ in range(LLAMA_3_2_3B.num_layers)
    )

    # Translate using the simplified wrapper layer API
    output_cache = CrossModelKVRegistry.translate(
        source_cache=mock_hf_past_key_values,
        src_cfg=LLAMA_3_2_3B,
        tg_cfg=LLAMA_3_70B,
        translator_model=translator,
    )

    # Verify target layers match exactly 80 blocks formatted for Llama-3-70B
    assert len(output_cache) == LLAMA_3_70B.num_layers
    # Check that individual layers match Hugging Face layout orientations: [B, H_tg, T, D]
    assert output_cache[0][0].shape == (1, LLAMA_3_70B.num_kv_heads, 32, LLAMA_3_70B.head_dim)
    assert output_cache[0][1].shape == (1, LLAMA_3_70B.num_kv_heads, 32, LLAMA_3_70B.head_dim)