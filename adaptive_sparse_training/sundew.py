"""
Sundew Algorithm - Adaptive Sample Selection with PI Control

Developed by Oluwafemi Idiakhoa
"""

import torch


class SundewAlgorithm:
    """
    Sundew Adaptive Gating Algorithm

    Uses PI-controlled adaptive sample selection to identify and process
    only the most important training samples, achieving massive energy savings.

    Key Features:
        - Multi-factor significance scoring (loss + entropy)
        - EMA-smoothed PI controller for stable threshold adaptation
        - Real-time energy tracking
        - Fallback mechanism to prevent zero-activation failures

    Args:
        config (ASTConfig): Configuration object

    Example:
        >>> from adaptive_sparse_training import SundewAlgorithm, ASTConfig
        >>> config = ASTConfig(target_activation_rate=0.40)
        >>> sundew = SundewAlgorithm(config)
        >>> active_mask, stats = sundew.select_samples(losses, outputs)
    """

    def __init__(self, config):
        self.target_activation_rate = config.target_activation_rate
        self.activation_threshold = config.initial_threshold
        self.kp = config.adapt_kp
        self.ki = config.adapt_ki
        self.integral_error = 0.0
        self.ema_alpha = config.ema_alpha
        self.activation_rate_ema = config.target_activation_rate

        # Energy tracking
        self.energy_per_activation = config.energy_per_activation
        self.energy_per_skip = config.energy_per_skip
        self.total_baseline_energy = 0.0
        self.total_actual_energy = 0.0

    def compute_significance(self, losses, outputs):
        """
        Compute sample importance scores using loss and prediction entropy

        Args:
            losses (torch.Tensor): Per-sample loss values [batch_size]
            outputs (torch.Tensor): Model logits [batch_size, num_classes]

        Returns:
            torch.Tensor: Significance scores [batch_size]
        """
        # Handle scalar loss (when reduction='mean' was used incorrectly)
        if losses.dim() == 0:
            raise ValueError(
                "Loss tensor has no dimensions (scalar). "
                "Please ensure your criterion uses reduction='none' for per-sample losses. "
                "Example: nn.CrossEntropyLoss(reduction='none')"
            )

        # RAW loss component (higher loss = more important)
        loss_component = losses

        # RAW entropy component (higher entropy = more uncertain)
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # Weighted combination (70% loss, 30% entropy)
        significance = 0.7 * loss_component + 0.3 * entropy
        return significance

    def select_samples(self, losses, outputs):
        """
        Select important samples based on significance scores

        Args:
            losses (torch.Tensor): Per-sample loss values [batch_size]
            outputs (torch.Tensor): Model logits [batch_size, num_classes]

        Returns:
            tuple: (active_mask, energy_info)
                - active_mask (torch.Tensor): Boolean mask [batch_size]
                - energy_info (dict): Statistics about activation and energy
        """
        # Handle scalar loss (when reduction='mean' was used incorrectly)
        if losses.dim() == 0:
            raise ValueError(
                "Loss tensor has no dimensions (scalar). "
                "Please ensure your criterion uses reduction='none' for per-sample losses. "
                "Example: nn.CrossEntropyLoss(reduction='none')"
            )

        batch_size = losses.size(0)

        # Compute significance scores
        significance = self.compute_significance(losses, outputs)

        # Select samples above threshold
        active_mask = significance > self.activation_threshold
        num_active = active_mask.sum().item()

        # Fallback: ensure minimum 10% activation
        min_active = max(2, int(batch_size * 0.10))
        if num_active < min_active:
            _, top_indices = torch.topk(significance, min_active)
            active_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            active_mask[top_indices] = True
            num_active = min_active

        # Update activation rate EMA
        current_activation_rate = num_active / batch_size
        self.activation_rate_ema = (
            self.ema_alpha * current_activation_rate +
            (1 - self.ema_alpha) * self.activation_rate_ema
        )

        # PI controller for threshold adaptation
        error = self.activation_rate_ema - self.target_activation_rate
        proportional = self.kp * error

        # Anti-windup: only accumulate integral within bounds
        if 0.5 < self.activation_threshold < 10.0:
            self.integral_error += error
            self.integral_error = max(-100, min(100, self.integral_error))
        else:
            self.integral_error *= 0.90

        # Update threshold
        new_threshold = self.activation_threshold + proportional + self.ki * self.integral_error
        self.activation_threshold = max(0.5, min(10.0, new_threshold))

        # Energy tracking
        baseline_energy = batch_size * self.energy_per_activation
        actual_energy = (num_active * self.energy_per_activation +
                        (batch_size - num_active) * self.energy_per_skip)

        self.total_baseline_energy += baseline_energy
        self.total_actual_energy += actual_energy

        energy_savings = 0.0
        if self.total_baseline_energy > 0:
            energy_savings = ((self.total_baseline_energy - self.total_actual_energy) /
                             self.total_baseline_energy * 100)

        energy_info = {
            'num_active': num_active,
            'activation_rate': current_activation_rate,
            'activation_rate_ema': self.activation_rate_ema,
            'threshold': self.activation_threshold,
            'energy_savings': energy_savings,
        }

        return active_mask, energy_info

    def reset(self):
        """Reset energy tracking (call at start of new epoch)"""
        self.total_baseline_energy = 0.0
        self.total_actual_energy = 0.0
