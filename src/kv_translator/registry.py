import torch
from kv_translator.config import LLAMAttentionConfig
from kv_translator.pipeline import FullKVCacheTranslator


class CrossModelKVRegistry:
    """The public unified entrypoint module interfacing directly with Hugging Face models."""

    @staticmethod
    def translate(
        source_cache: tuple[tuple[torch.Tensor, torch.Tensor], ...],
        src_cfg: LLAMAttentionConfig,
        tg_cfg: LLAMAttentionConfig,
        translator_model: FullKVCacheTranslator,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        """Converts and maps Hugging Face multi-layer caches between differing architectures.

        Args:
            source_cache: Core nested past_key_values block structure from standard HF runs.
            src_cfg: Architecture definition payload for the original model.
            tg_cfg: Architecture definition payload for the destination model.
            translator_model: Loaded instance of the weight mapping network layer.
        """
        # 1. Reshape Hugging Face format [B, H, T, D] -> Pipeline Internal format [B, T, H, D]
        src_k_list = []
        src_v_list = []

        for layer_idx in range(len(source_cache)):
            k_layer_hf, v_layer_hf = source_cache[layer_idx]
            # Transpose attention heads and token lengths securely
            src_k_list.append(k_layer_hf.transpose(1, 2))
            src_v_list.append(v_layer_hf.transpose(1, 2))

        # 2. Run core sequence translation mechanics
        tg_k_list, tg_v_list = translator_model(src_k_list, src_v_list)

        # 3. Format back to target HF structure [B, T, H, D] -> [B, H, T, D]
        hf_formatted_cache = []
        for layer_idx in range(len(tg_k_list)):
            k_tg_hf = tg_k_list[layer_idx].transpose(1, 2)
            v_tg_hf = tg_v_list[layer_idx].transpose(1, 2)
            hf_formatted_cache.append((k_tg_hf, v_tg_hf))

        return tuple(hf_formatted_cache)
