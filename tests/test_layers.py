import torch
from kv_translator.config import LLAMA_3_2_3B, LLAMA_3_70B
from kv_translator.layers import KVLayerTranslator


def test_layer_translator_forward():
    translator = KVLayerTranslator(LLAMA_3_2_3B, LLAMA_3_70B)

    # Simulate Llama 3.2 3B cache tensor shapes
    mock_k = torch.randn(1, 32, LLAMA_3_2_3B.num_kv_heads, LLAMA_3_2_3B.head_dim)
    mock_v = torch.randn(1, 32, LLAMA_3_2_3B.num_kv_heads, LLAMA_3_2_3B.head_dim)

    k_out, v_out = translator(mock_k, mock_v)

    # Assert structural targets match expected Llama 70B sizes
    assert k_out.shape == (1, 32, LLAMA_3_70B.num_kv_heads, LLAMA_3_70B.head_dim)
    assert v_out.shape == (1, 32, LLAMA_3_70B.num_kv_heads, LLAMA_3_70B.head_dim)
