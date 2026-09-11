"""Typed configuration for the independent monthly H3 GNN baseline."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class GNNTrainingConfig:
    """Inputs and conservative initial hyperparameters for Model B."""

    sales_csv: str = "data/sales_2020_25.csv"
    market_indicator_csv: str = "data/market_indicators/fred_indicators.csv"
    train_ratio: float = 0.7
    random_seed: int = 42
    graph_hidden_dim: int = 32
    head_hidden_dim: int = 128
    dropout_rate: float = 0.2
    graph_layer_norm: bool = False
    graph_residual: bool = False
    epochs: int = 30
    learning_rate: float = 3e-4
    patience: int = 8
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    min_learning_rate: float = 1e-6

    def __post_init__(self) -> None:
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between zero and one")
        if min(self.graph_hidden_dim, self.head_hidden_dim, self.epochs, self.patience) < 1:
            raise ValueError("GNN dimensions and epoch counts must be positive")
        if not 0 < self.lr_plateau_factor < 1:
            raise ValueError("lr_plateau_factor must be between zero and one")

    def as_dict(self) -> dict:
        """Return run metadata in a JSON-safe form."""
        return asdict(self)
