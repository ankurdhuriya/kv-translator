import torch


class RoPEHandler:
    @staticmethod
    def compute_freqs(
        seq_len: int, head_dim: int, theta: float, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim)
        )
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    @classmethod
    def apply(cls, x: torch.Tensor, theta: float) -> torch.Tensor:
        # Expected shape: [Batch, Seq, Heads, Dim]
        _, seq_len, _, head_dim = x.shape
        cos, sin = cls.compute_freqs(seq_len, head_dim, theta, x.device)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)  # Broadcastable across heads
        return (x * cos) + (cls._rotate_half(x) * sin)

    @classmethod
    def inverse(cls, x: torch.Tensor, theta: float) -> torch.Tensor:
        _, seq_len, _, head_dim = x.shape
        cos, sin = cls.compute_freqs(seq_len, head_dim, theta, x.device)
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        # Transposed rotation matrix application: change sign of the sine component rotation
        return (x * cos) - (cls._rotate_half(x) * sin)
