# deepseek_physical_ai/training_significance.py
"""
Multimodal Training Significance Model

Computes sample significance for adaptive training across:
- Vision (ImageNet, COCO, etc.)
- Language (LLM pretraining, fine-tuning)
- Audio (speech, music)
- Robotics (manipulation, navigation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import Sundew interfaces
import sys
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from sundew.interfaces import ProcessingContext, SignificanceModel


@dataclass
class TrainingSampleContext(ProcessingContext):
    """Extended context for training sample significance."""

    # Core sample data
    sample_id: int = 0
    modality: str = "vision"
    batch_index: int = 0
    epoch: int = 0

    # Learning dynamics
    current_loss: float = 0.0
    loss_history: List[float] = field(default_factory=list)
    gradient_norm: Optional[float] = None
    prediction_entropy: Optional[float] = None

    # Curriculum signals
    sample_difficulty: float = 0.5
    seen_count: int = 0
    last_seen_epoch: int = -1

    # Physical feedback (for embodied tasks)
    physical_success: Optional[bool] = None
    sim2real_gap: Optional[float] = None
    human_feedback: Optional[float] = None

    # Modality-specific features
    modality_features: Optional[Dict[str, Any]] = None


class MultimodalTrainingSignificance(SignificanceModel):
    """
    Compute training sample significance across modalities.

    Significance = weighted combination of:
    1. Learning value: Predicted gradient magnitude
    2. Difficulty: Loss landscape curvature
    3. Novelty: Representation diversity
    4. Uncertainty: Prediction entropy
    5. Physical grounding: Real-world feedback
    """

    def __init__(
        self,
        modality: str,
        gradient_predictor: Optional[Any] = None,
        novelty_buffer_size: int = 1000,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.modality = modality
        self.gradient_predictor = gradient_predictor
        self.novelty_buffer_size = novelty_buffer_size

        # Adaptive weighting
        self.config = config or {}
        self.w_learning = self.config.get("w_learning", 0.35)
        self.w_difficulty = self.config.get("w_difficulty", 0.25)
        self.w_novelty = self.config.get("w_novelty", 0.20)
        self.w_uncertainty = self.config.get("w_uncertainty", 0.10)
        self.w_physical = self.config.get("w_physical", 0.10)

        # Modality-specific parameters
        self._init_modality_params()

        # Novelty tracking
        self.representation_buffer: List[np.ndarray] = []
        self.sample_stats: Dict[int, Dict[str, float]] = {}

        # Learning dynamics
        self.epoch_samples_seen = 0
        self.high_significance_count = 0

    def _init_modality_params(self) -> None:
        """Initialize modality-specific thresholds."""
        modality_configs = {
            "vision": {
                "base_difficulty": 0.5,
                "novelty_threshold": 0.3,
                "entropy_scale": 2.0,
            },
            "language": {
                "base_difficulty": 0.6,
                "novelty_threshold": 0.4,
                "entropy_scale": 1.5,
            },
            "audio": {
                "base_difficulty": 0.55,
                "novelty_threshold": 0.35,
                "entropy_scale": 1.8,
            },
            "robot": {
                "base_difficulty": 0.7,
                "novelty_threshold": 0.25,
                "entropy_scale": 2.5,
                "physical_weight_boost": 2.0,
            },
        }

        self.modality_config = modality_configs.get(self.modality, modality_configs["vision"])

        # Boost physical feedback for robotics
        if self.modality == "robot":
            self.w_physical *= self.modality_config["physical_weight_boost"]
            total = (
                self.w_learning
                + self.w_difficulty
                + self.w_novelty
                + self.w_uncertainty
                + self.w_physical
            )
            self.w_learning /= total
            self.w_difficulty /= total
            self.w_novelty /= total
            self.w_uncertainty /= total
            self.w_physical /= total

    def compute_significance(self, context: ProcessingContext) -> Tuple[float, Dict[str, Any]]:
        """Compute comprehensive training sample significance."""

        # Cast to TrainingSampleContext if needed
        if not isinstance(context, TrainingSampleContext):
            # Create minimal context
            context = TrainingSampleContext(
                timestamp=context.timestamp,
                sequence_id=context.sequence_id,
                features=context.features,
                history=context.history,
                metadata=context.metadata,
            )

        # 1. Learning Value
        learning_sig = self._compute_learning_value(context)

        # 2. Difficulty
        difficulty_sig = self._compute_difficulty(context)

        # 3. Novelty
        novelty_sig = self._compute_novelty(context)

        # 4. Uncertainty
        uncertainty_sig = self._compute_uncertainty(context)

        # 5. Physical Grounding
        physical_sig = self._compute_physical_feedback(context)

        # Weighted combination
        significance = (
            self.w_learning * learning_sig
            + self.w_difficulty * difficulty_sig
            + self.w_novelty * novelty_sig
            + self.w_uncertainty * uncertainty_sig
            + self.w_physical * physical_sig
        )

        # Curriculum adjustment
        if context.seen_count > 5:
            familiarity_penalty = 1.0 / (1.0 + 0.1 * context.seen_count)
            significance *= familiarity_penalty

        significance = float(np.clip(significance, 0.0, 1.0))

        explanation = {
            "learning_value": learning_sig,
            "difficulty": difficulty_sig,
            "novelty": novelty_sig,
            "uncertainty": uncertainty_sig,
            "physical_feedback": physical_sig,
            "final_significance": significance,
            "modality": self.modality,
            "seen_count": context.seen_count,
        }

        return significance, explanation

    def _compute_learning_value(self, context: TrainingSampleContext) -> float:
        """Predict gradient magnitude contribution."""
        if context.gradient_norm is not None:
            typical_grad = 1.0
            return min(context.gradient_norm / typical_grad, 1.0)

        if len(context.loss_history) >= 2:
            loss_delta = context.loss_history[-1] - context.loss_history[-2]
            if loss_delta < 0:
                learning_value = min(abs(loss_delta) / 0.5, 1.0)
            else:
                learning_value = 0.3 + min(abs(loss_delta) / 0.5, 0.7)
            return learning_value

        return 0.5

    def _compute_difficulty(self, context: TrainingSampleContext) -> float:
        """Estimate sample difficulty."""
        if context.current_loss > 0:
            typical_loss = {"vision": 1.0, "language": 2.0, "audio": 1.5, "robot": 2.5}
            norm_loss = context.current_loss / typical_loss.get(self.modality, 1.0)
            difficulty = min(norm_loss, 1.0)
        else:
            difficulty = context.sample_difficulty

        # Curriculum: easy early, hard late
        if context.epoch < 5:
            difficulty *= 0.5
        elif context.epoch > 20:
            difficulty = 0.3 + 0.7 * difficulty

        return float(np.clip(difficulty, 0.0, 1.0))

    def _compute_novelty(self, context: TrainingSampleContext) -> float:
        """Compute representation novelty."""
        if context.modality_features is None:
            return 0.5

        representation = self._extract_representation(context.modality_features)

        if len(self.representation_buffer) == 0:
            novelty = 1.0
        else:
            distances = [
                np.linalg.norm(representation - rep)
                for rep in self.representation_buffer[-self.novelty_buffer_size :]
            ]
            min_distance = min(distances)
            threshold = self.modality_config["novelty_threshold"]
            novelty = min(min_distance / threshold, 1.0)

        self.representation_buffer.append(representation)
        if len(self.representation_buffer) > self.novelty_buffer_size:
            self.representation_buffer.pop(0)

        return novelty

    def _compute_uncertainty(self, context: TrainingSampleContext) -> float:
        """Compute model uncertainty."""
        if context.prediction_entropy is not None:
            entropy_scale = self.modality_config["entropy_scale"]
            uncertainty = min(context.prediction_entropy / entropy_scale, 1.0)
            return uncertainty

        if context.current_loss > 0:
            return min(context.current_loss / 2.0, 1.0)

        return 0.5

    def _compute_physical_feedback(self, context: TrainingSampleContext) -> float:
        """Incorporate physical world feedback."""
        if self.modality != "robot":
            return 0.0

        physical_sig = 0.0

        if context.physical_success is not None:
            physical_sig += 0.8 if not context.physical_success else 0.2

        if context.sim2real_gap is not None:
            physical_sig += min(context.sim2real_gap / 0.5, 0.7)

        if context.human_feedback is not None:
            physical_sig += abs(context.human_feedback) * 0.5

        return float(np.clip(physical_sig, 0.0, 1.0))

    def _extract_representation(self, features: Dict[str, Any]) -> np.ndarray:
        """Extract fixed-size representation."""
        if "embedding" in features:
            return np.array(features["embedding"])

        feature_vector = []
        for key in sorted(features.keys()):
            val = features[key]
            if isinstance(val, (int, float)):
                feature_vector.append(float(val))

        # Pad or truncate to fixed size
        if len(feature_vector) < 64:
            feature_vector.extend([0.0] * (64 - len(feature_vector)))
        return np.array(feature_vector[:64])

    def update(self, context: ProcessingContext, outcome: Optional[Dict[str, Any]]) -> None:
        """Update significance model based on outcome."""
        if outcome is None or not isinstance(context, TrainingSampleContext):
            return

        sample_id = context.sample_id

        if sample_id not in self.sample_stats:
            self.sample_stats[sample_id] = {
                "seen_count": 0,
                "total_loss": 0.0,
                "total_gradient_norm": 0.0,
                "avg_significance": 0.0,
            }

        stats = self.sample_stats[sample_id]
        stats["seen_count"] += 1
        stats["total_loss"] += outcome.get("loss", 0.0)
        stats["total_gradient_norm"] += outcome.get("gradient_norm", 0.0)

    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "modality": self.modality,
            "weights": {
                "learning": self.w_learning,
                "difficulty": self.w_difficulty,
                "novelty": self.w_novelty,
                "uncertainty": self.w_uncertainty,
                "physical": self.w_physical,
            },
            "sample_stats": self.sample_stats,
        }

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set model parameters."""
        if "weights" in params:
            w = params["weights"]
            self.w_learning = w.get("learning", self.w_learning)
            self.w_difficulty = w.get("difficulty", self.w_difficulty)
            self.w_novelty = w.get("novelty", self.w_novelty)
            self.w_uncertainty = w.get("uncertainty", self.w_uncertainty)
            self.w_physical = w.get("physical", self.w_physical)

        if "sample_stats" in params:
            self.sample_stats = params["sample_stats"]


# Modality-specific classes

class VisionTrainingSignificance(MultimodalTrainingSignificance):
    """Vision-specific training significance."""

    def __init__(self, **kwargs):
        super().__init__(modality="vision", **kwargs)
        self.edge_threshold = 50.0
        self.texture_threshold = 0.3


class LanguageTrainingSignificance(MultimodalTrainingSignificance):
    """Language-specific training significance."""

    def __init__(self, **kwargs):
        super().__init__(modality="language", **kwargs)
        self.perplexity_threshold = 20.0
        self.rare_token_weight = 1.5


class RobotTrainingSignificance(MultimodalTrainingSignificance):
    """Robot-specific training significance."""

    def __init__(self, **kwargs):
        super().__init__(modality="robot", **kwargs)
        self.force_threshold = 5.0
        self.smoothness_threshold = 0.1
