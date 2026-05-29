import torch
from kv_translator.rope import RoPEHandler


def test_rope_inversion_parity():
    # Construct a random raw hidden token array
    batch, seq, heads, head_dim = 2, 64, 8, 128
    original_tensor = torch.randn(batch, seq, heads, head_dim)
    theta = 500000.0

    # Phase 1: Encode positions
    rotated = RoPEHandler.apply(original_tensor, theta)
    assert rotated.shape == original_tensor.shape
    assert not torch.allclose(original_tensor, rotated, atol=1e-4)

    # Phase 2: Invert/Strip positions
    unrotated = RoPEHandler.inverse(rotated, theta)

    # Matrix multiplication reconstruction validation
    assert torch.allclose(original_tensor, unrotated, atol=1e-4), (
        "RoPE Inversion identity failed math parity!"
    )
