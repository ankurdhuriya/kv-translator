from kv_translator.config import LLAMA_3_2_3B, LLAMA_3_70B


def test_model_configurations():
    assert LLAMA_3_2_3B.num_layers == 24
    assert LLAMA_3_70B.num_layers == 80
    assert LLAMA_3_2_3B.head_dim == LLAMA_3_70B.head_dim == 128
