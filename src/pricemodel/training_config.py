"""Typed configuration for reproducible neural-model training runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Validated hyperparameters and input locations for one training run."""

    sales_csv: str = "data/sales_2020_25.csv"
    market_indicator_csv: str = "data/market_indicators/fred_indicators.csv"
    train_ratio: float = 0.7
    temporal_split: bool = True
    random_seed: int = 42
    embedding_dim: int = 128
    community_embedding_dim: int = 16
    hidden_dim: int = 256
    continuous_time_dim: int = 3
    market_dim: int = 2
    epochs: int = 30
    batch_size: int = 256
    learning_rate: float = 3e-4
    dropout_rate: float = 0.2
    estimate_uncertainty: bool = True
    pooling_strategy: str = "center_weighted"
    # Mean-stage early stopping is deliberately longer than the scheduler
    # patience, leaving several epochs to benefit from a reduced learning rate.
    patience: int = 8
    lr_plateau_factor: float = 0.5
    lr_plateau_patience: int = 2
    min_learning_rate: float = 1e-6
    global_aux_weight: float = 0.5
    residual_penalty: float = 1e-2
    uncertainty_calibration_epochs: int = 10
    uncertainty_patience: int = 3

    def __post_init__(self):
        if not 0 < self.train_ratio < 1:
            raise ValueError("train_ratio must be between zero and one")
        if not self.temporal_split:
            raise ValueError(
                "temporal_split must remain enabled for leakage-safe local features"
            )
        if self.random_seed < 0:
            raise ValueError("random_seed must be zero or positive")
        if self.community_embedding_dim < 1:
            raise ValueError("community_embedding_dim must be positive")
        if self.pooling_strategy not in {"mean", "center_weighted", "learnable"}:
            raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
        if min(
            self.epochs, self.batch_size, self.patience,
            self.lr_plateau_patience, self.uncertainty_calibration_epochs,
            self.uncertainty_patience,
        ) < 1:
            raise ValueError("epoch counts, batch size, and patience values must be positive")
        if not 0 < self.lr_plateau_factor < 1:
            raise ValueError("lr_plateau_factor must be between zero and one")
        if self.min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be positive")
        if self.global_aux_weight < 0 or self.residual_penalty < 0:
            raise ValueError("loss weights and penalties must be non-negative")

    def train_kwargs(self, property_dim: int) -> dict:
        """Return the arguments accepted by ``ModelManager.train_model``."""
        return {
            "embedding_dim": self.embedding_dim,
            "community_embedding_dim": self.community_embedding_dim,
            "hidden_dim": self.hidden_dim,
            "property_dim": property_dim,
            "continuous_time_dim": self.continuous_time_dim,
            "market_dim": self.market_dim,
            "epochs": self.epochs,
            "batch": self.batch_size,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "estimate_uncertainty": self.estimate_uncertainty,
            "pooling_strategy": self.pooling_strategy,
            "patience": self.patience,
            "lr_plateau_factor": self.lr_plateau_factor,
            "lr_plateau_patience": self.lr_plateau_patience,
            "min_learning_rate": self.min_learning_rate,
            "global_aux_weight": self.global_aux_weight,
            "residual_penalty": self.residual_penalty,
            "uncertainty_calibration_epochs": self.uncertainty_calibration_epochs,
            "uncertainty_patience": self.uncertainty_patience,
            "random_seed": self.random_seed,
        }

    def as_dict(self) -> dict:
        """Return JSON-serializable run metadata."""
        return asdict(self)
