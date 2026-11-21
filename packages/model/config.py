from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_layers: int
    hidden_size: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn_dim: int
    max_position_embeddings: int = 2048
    rope_theta: float = 100000.0
    rms_norm_eps: float = 1e-6
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size != self.n_heads * self.head_dim:
            raise ValueError("hidden_size must equal n_heads * head_dim")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        return cls(**json.loads(Path(path).read_text()))
