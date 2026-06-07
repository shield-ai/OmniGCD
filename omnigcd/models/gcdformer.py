"""GCDformer model used by OmniGCD.

The paper describes GCDformer as a Transformer set processor operating on a
low-dimensional GCD latent space. Each input point is concatenated with either
(1) a sinusoidal label embedding for known/labeled points, or (2) a learned mask
embedding for unlabeled points. No positional encoding is used by default because
GCD inputs are sets, not sequences.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class GCDFormerConfig:
    """Configuration for :class:`GCDFormer`.

    Defaults mirror the main paper setting: 2D GCD latent-space input/output,
    224 data-token dimensions + 32 label-token dimensions = 256 model width,
    6 transformer blocks and 4 attention heads.
    """

    input_dim: int = 2
    output_dim: int = 2
    data_embed_dim: int = 224
    label_embed_dim: int = 32
    n_layer: int = 6
    n_head: int = 4
    max_label_id: int = 1000
    dropout: float = 0.0
    use_positional_embeddings: bool = False
    max_sequence_length: int = 3000

    @property
    def d_model(self) -> int:
        return self.data_embed_dim + self.label_embed_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GCDFormerConfig":
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in allowed})


class GCDFormer:  # real base class assigned below after importing torch
    pass


def _build_torch_model():
    import math
    import torch
    import torch.nn as nn

    class SinusoidalLabelEmbedding(nn.Module):
        """Sinusoidal label embedding used for known labels.

        This is analogous to Transformer sinusoidal position encoding, but the
        integer being encoded is a class label rather than a token position.
        """

        def __init__(self, dim: int, max_label_id: int) -> None:
            super().__init__()
            label_ids = torch.arange(0, max_label_id + 1).unsqueeze(1)
            frequencies = torch.pow(
                10000.0,
                -torch.arange(0, dim, 2, dtype=torch.float32) / dim,
            )
            table = torch.zeros(max_label_id + 1, dim, dtype=torch.float32)
            table[:, 0::2] = torch.sin(label_ids * frequencies)
            table[:, 1::2] = torch.cos(label_ids * frequencies)
            self.register_buffer("table", table, persistent=True)

        def forward(self, labels: torch.Tensor) -> torch.Tensor:
            labels = labels.clamp(min=0, max=self.table.shape[0] - 1).long()
            return self.table[labels]

    class TransformerBlock(nn.Module):
        def __init__(self, config: GCDFormerConfig) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(config.d_model)
            self.attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.n_head,
                dropout=config.dropout,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(config.d_model)
            self.mlp = nn.Sequential(
                nn.Linear(config.d_model, 4 * config.d_model),
                nn.GELU(),
                nn.Linear(4 * config.d_model, config.d_model),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
            x = x + y
            x = x + self.mlp(self.norm2(x))
            return x

    class TorchGCDFormer(nn.Module):
        def __init__(self, config: GCDFormerConfig) -> None:
            super().__init__()
            if config.d_model % config.n_head != 0:
                raise ValueError("config.d_model must be divisible by config.n_head")
            self.config = config
            self.data_projection = nn.Linear(config.input_dim, config.data_embed_dim)
            self.label_embedding = SinusoidalLabelEmbedding(
                dim=config.label_embed_dim,
                max_label_id=config.max_label_id,
            )
            self.mask_embedding = nn.Embedding(1, config.label_embed_dim)
            self.position_embedding = (
                nn.Embedding(config.max_sequence_length, config.d_model)
                if config.use_positional_embeddings
                else None
            )
            self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
            self.final_norm = nn.LayerNorm(config.d_model)
            self.output_projection = nn.Linear(config.d_model, config.output_dim)
            self.apply(self._init_weights)

        def forward(
            self,
            points: torch.Tensor,
            labels: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            """Transform a tokenized GCD latent space.

            Args:
                points: Tensor with shape ``[batch, n_points, input_dim]``.
                labels: Integer tensor with shape ``[batch, n_points]``. Known
                    points should use positive class IDs. Unlabeled/masked points
                    usually use ``0``.
                mask: Integer/bool tensor with shape ``[batch, n_points]`` where
                    ``1`` means unlabeled/masked and ``0`` means known/labeled.

            Returns:
                Tensor with shape ``[batch, n_points, output_dim]``.
            """
            data_tokens = self.data_projection(points)
            known_label_tokens = self.label_embedding(labels)
            masked_label_tokens = self.mask_embedding(torch.zeros_like(labels).long())
            label_tokens = torch.where(mask.bool().unsqueeze(-1), masked_label_tokens, known_label_tokens)
            x = torch.cat([data_tokens, label_tokens], dim=-1)

            if self.position_embedding is not None:
                positions = torch.arange(x.shape[1], device=x.device).long()
                x = x + self.position_embedding(positions).unsqueeze(0)

            for block in self.blocks:
                x = block(x)
            return self.output_projection(self.final_norm(x))

        @staticmethod
        def _init_weights(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

        def save_checkpoint(self, path: str) -> None:
            torch.save({"config": self.config.to_dict(), "state_dict": self.state_dict()}, path)

        @classmethod
        def load_checkpoint(cls, path: str, map_location: str | torch.device = "cpu") -> "TorchGCDFormer":
            checkpoint = torch.load(path, map_location=map_location)
            config = GCDFormerConfig.from_dict(checkpoint["config"])
            model = cls(config)
            model.load_state_dict(checkpoint["state_dict"])
            return model

    return TorchGCDFormer


try:
    GCDFormer = _build_torch_model()
except ModuleNotFoundError:
    # Allows importing omnigcd metadata and numpy-only evaluation utilities in
    # environments where PyTorch has not yet been installed. Instantiating the
    # model still requires installing the training extra, e.g. `pip install -e .[train]`.
    class GCDFormer:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ModuleNotFoundError(
                "GCDFormer requires PyTorch. Install it with `pip install -e .[train]` "
                "or follow the platform-specific instructions at https://pytorch.org/."
            )
