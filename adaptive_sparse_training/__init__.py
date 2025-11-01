"""
Adaptive Sparse Training (AST) - Energy-Efficient Deep Learning

Developed by Oluwafemi Idiakhoa

Production-ready implementation of Adaptive Sparse Training with Sundew Adaptive Gating.
Achieves 60%+ energy savings with zero accuracy degradation.

Example usage:
    >>> from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
    >>>
    >>> config = ASTConfig(target_activation_rate=0.40)
    >>> trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
    >>> results = trainer.train(epochs=100)
"""

from .trainer import AdaptiveSparseTrainer
from .sundew import SundewAlgorithm
from .config import ASTConfig

__version__ = "1.0.0"
__author__ = "Oluwafemi Idiakhoa"
__all__ = ["AdaptiveSparseTrainer", "SundewAlgorithm", "ASTConfig"]
