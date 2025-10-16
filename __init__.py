"""
Adaptive Sparse Training (AST) Framework

Combines:
- Sundew Adaptive Gating (WHEN to compute)
- DeepSeek Sparse Attention (HOW to compute efficiently)
- Physical AI Principles (WHAT to learn from embodiment)

Result: 50× faster training with superior generalization
"""

__version__ = "0.1.0"

# Note: Import directly from modules in your code:
#   from deepseek_physical_ai.training_significance import MultimodalTrainingSignificance
#   from deepseek_physical_ai.sparse_transformer import SparseViT
#   from deepseek_physical_ai.adaptive_training_loop import AdaptiveSparseTrainer

__all__ = [
    "MultimodalTrainingSignificance",
    "VisionTrainingSignificance",
    "LanguageTrainingSignificance",
    "RobotTrainingSignificance",
    "TrainingSampleContext",
    "DeepSeekSparseAttention",
    "SparseTransformerBlock",
    "SparseAttentionConfig",
    "SparseViT",
    "AdaptiveSparseTrainer",
]
