"""Backward-compatible imports for the refactored neural-model modules.

New code should import DatasetBuilder from data_pipeline, EnhancedEmbeddingModel
from network, and PriceTrainer from trainer. Legacy lowercase aliases remain
available here while scripts migrate.
"""

from .data_pipeline import DatasetBuilder, create_vocab, dataset, vocab_replace_tensor
from .network import EnhancedEmbeddingModel
from .trainer import PriceTrainer, price_predictor

__all__ = [
    "DatasetBuilder",
    "EnhancedEmbeddingModel",
    "PriceTrainer",
    "create_vocab",
    "dataset",
    "price_predictor",
    "vocab_replace_tensor",
]
