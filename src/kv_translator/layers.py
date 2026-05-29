import torch
import torch.nn as nn
from kv_translator.config import LLMAttentionConfig
from kv_translator.rope import RoPEHandler


class KVLayerTranslator(nn.Module):
    def __init__(self, src_cfg: LLMAttentionConfig, tg_cfg: LLMAttentionConfig):
        super().__init__()
        self.src_cfg = src_cfg
        self.tg_cfg = tg_cfg

        src_flat = src_cfg.num_kv_heads * src_cfg.head_dim
        tg_flat = tg_cfg.num_kv_heads * tg_cfg.head_dim

        # Projection multi-layer alignment channels
        self.k_projector = nn.Sequential(
            nn.Linear(src_flat, tg_flat, bias=False),
            nn.SiLU(),
            nn.Linear(tg_flat, tg_flat, bias=False),
        )
        self.v_projector = nn.Sequential(
            nn.Linear(src_flat, tg_flat, bias=False),
            nn.SiLU(),
            nn.Linear(tg_flat, tg_flat, bias=False),
        )

    def forward(
        self, k_src: torch.Tensor, v_src: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, _ = k_src.shape

        # 1. Strip positional encodings
        k_unrotated = RoPEHandler.inverse(k_src, self.src_cfg.rope_theta)

        # 2. Reshape for linear projection
        k_flat = k_unrotated.reshape(b, t, -1)
        v_flat = v_src.reshape(b, t, -1)

        # 3. Translate across model channels
        k_tg_raw = self.k_projector(k_flat).reshape(
            b, t, self.tg_cfg.num_kv_heads, self.tg_cfg.head_dim
        )
        v_tg = self.v_projector(v_flat).reshape(
            b, t, self.tg_cfg.num_kv_heads, self.tg_cfg.head_dim
        )

        # 4. Re-apply target positional parameters
        k_tg = RoPEHandler.apply(k_tg_raw, self.tg_cfg.rope_theta)

        return k_tg, v_tg
