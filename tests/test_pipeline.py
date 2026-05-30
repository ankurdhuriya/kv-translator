import torch
import pytest
from kv_translator import LLAMA_3_2_3B, LLAMA_3_70B, FullKVCacheTranslator


def test_full_pipeline_cache_translation():
    """Verifies that the multi-layer pipeline can expand 24 layers of Llama 3.2 3B
    cache up to exactly 80 layers matching Llama 3 70B layout dimensions.
    """
    # Instantiate the full network model translator mapping 3B -> 70B
    pipeline = FullKVCacheTranslator(LLAMA_3_2_3B, LLAMA_3_70B)

    batch_size = 2
    seq_len = 64

    # Generate a dummy complete model cache representation for Llama 3.2 3B (24 Layers)
    mock_src_k = [
        torch.randn(
            batch_size, seq_len, LLAMA_3_2_3B.num_kv_heads, LLAMA_3_2_3B.head_dim
        )
        for _ in range(LLAMA_3_2_3B.num_layers)
    ]
    mock_src_v = [
        torch.randn(
            batch_size, seq_len, LLAMA_3_2_3B.num_kv_heads, LLAMA_3_2_3B.head_dim
        )
        for _ in range(LLAMA_3_2_3B.num_layers)
    ]

    # Run the full sequence forward translation pass
    tg_k, tg_v = pipeline(mock_src_k, mock_src_v)

    # Assert depth mappings matched Target structural constraints (80 Layers)
    assert len(tg_k) == LLAMA_3_70B.num_layers
    assert len(tg_v) == LLAMA_3_70B.num_layers

    # Assert spatial dimension changes are maintained accurately [B, T, H_tg, D]
    expected_tg_shape = (
        batch_size,
        seq_len,
        LLAMA_3_70B.num_kv_heads,
        LLAMA_3_70B.head_dim,
    )

    assert tg_k[0].shape == expected_tg_shape, (
        f"Expected key shape {expected_tg_shape}, got {tg_k[0].shape}"
    )
    assert tg_v[0].shape == expected_tg_shape, (
        f"Expected value shape {expected_tg_shape}, got {tg_v[0].shape}"
    )


def test_pipeline_layer_mapping_logic():
    """Verifies the mathematical boundary boundaries of the linear layer-interpolation engine."""
    pipeline = FullKVCacheTranslator(LLAMA_3_2_3B, LLAMA_3_70B)

    # The first target layer (0) must map exactly to source layer 0
    assert pipeline.layer_mapping[0] == 0

    # The last target layer (79) must map to the highest index available in the source (23)
    assert (
        pipeline.layer_mapping[LLAMA_3_70B.num_layers - 1]
        == LLAMA_3_2_3B.num_layers - 1
    )

    # Verify strict monotonic layer scaling sequences
    for i in range(len(pipeline.layer_mapping) - 1):
        assert pipeline.layer_mapping[i] <= pipeline.layer_mapping[i + 1], (
            "Layer mapping sequence should be non-decreasing"
        )


def test_pipeline_invalid_layer_count_exception():
    """Ensures that passing an incorrect number of source layers immediately raises a ValueError."""
    pipeline = FullKVCacheTranslator(LLAMA_3_2_3B, LLAMA_3_70B)

    # Pass an incomplete cache list containing only 5 layers instead of the required 24
    bad_mock_k = [torch.randn(1, 16, 8, 128) for _ in range(5)]
    bad_mock_v = [torch.randn(1, 16, 8, 128) for _ in range(5)]

    with pytest.raises(ValueError, match="Expected exactly 24 source layers"):
        pipeline(bad_mock_k, bad_mock_v)
