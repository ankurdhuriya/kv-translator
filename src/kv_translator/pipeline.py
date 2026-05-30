import math
import torch
import torch.nn as nn
from kv_translator.config import LLAMAttentionConfig
from kv_translator.layers import KVLayerTranslator


class FullKVCacheTranslator(nn.Module):
    """Coordinates and translates an entire sequence of Key-Value (KV) caches
    across mismatched model architectures (e.g., Llama-3.2-3B to Llama-3-70B).
    """

    def __init__(self, src_cfg: LLAMAttentionConfig, tg_cfg: LLAMAttentionConfig):
        """
        Initializes the full network translator. It mathematically maps each target layer
        to an interpolated source layer and instantiates corresponding translation blocks.
        """
        super().__init__()
        self.src_cfg = src_cfg
        self.tg_cfg = tg_cfg

        # Map every Target layer index to an interpolated Source layer index
        self.layer_mapping = {}
        for tg_l in range(tg_cfg.num_layers):
            # Calculate the fractional scaling ratio across model depths
            fractional_idx = tg_l * (src_cfg.num_layers / tg_cfg.num_layers)
            corresponding_src_l = min(
                int(math.floor(fractional_idx)), src_cfg.num_layers - 1
            )
            self.layer_mapping[tg_l] = corresponding_src_l

        # Instantiate a dedicated parameter-driven layer block for every target layer
        self.layer_translators = nn.ModuleList(
            [KVLayerTranslator(src_cfg, tg_cfg) for _ in range(tg_cfg.num_layers)]
        )

    def forward(
        self, src_k_cache: list[torch.Tensor], src_v_cache: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Translates whole-model KV caches sequentially layer by layer.

        Args:
            src_k_cache: List of length `src_cfg.num_layers` containing tensors of shape [B, T, H_src, D]
            src_v_cache: List of length `src_cfg.num_layers` containing tensors of shape [B, T, H_src, D]

        Returns:
            tg_k_cache: List of length `tg_cfg.num_layers` containing tensors of shape [B, T, H_tg, D]
            tg_v_cache: List of length `tg_cfg.num_layers` containing tensors of shape [B, T, H_tg, D]
        """
        if (
            len(src_k_cache) != self.src_cfg.num_layers
            or len(src_v_cache) != self.src_cfg.num_layers
        ):
            raise ValueError(
                f"Expected exactly {self.src_cfg.num_layers} source layers, "
                f"got Keys length: {len(src_k_cache)}, Values length: {len(src_v_cache)}"
            )

        tg_k_cache = []
        tg_v_cache = []

        # Iterate over target depth space sequentially
        for tg_layer_idx in range(self.tg_cfg.num_layers):
            # Lookup the source layer responsible for servicing this target domain
            src_layer_idx = self.layer_mapping[tg_layer_idx]

            k_src_layer = src_k_cache[src_layer_idx]
            v_src_layer = src_v_cache[src_layer_idx]

            # Translate using the dedicated block weights
            k_tg, v_tg = self.layer_translators[tg_layer_idx](k_src_layer, v_src_layer)

            tg_k_cache.append(k_tg)
            tg_v_cache.append(v_tg)

        return tg_k_cache, tg_v_cache
