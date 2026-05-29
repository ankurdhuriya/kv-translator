from dataclasses import dataclass


@dataclass(frozen=True)
class LLAMAttentionConfig:
    num_layers: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float


# Real physical dimensions of Meta's Llama models
LLAMA_3_2_3B = LLAMAttentionConfig(
    num_layers=24,  # Llama 3.2 3B has 24 layers
    num_kv_heads=8,
    head_dim=128,
    rope_theta=500000.0,
)

LLAMA_3_70B = LLAMAttentionConfig(
    num_layers=80,  # Llama 3 70B has 80 layers
    num_kv_heads=8,
    head_dim=128,
    rope_theta=500000.0,
)
