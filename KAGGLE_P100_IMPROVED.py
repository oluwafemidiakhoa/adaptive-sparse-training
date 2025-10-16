#!/usr/bin/env python3
"""
P100 GPU Test - Using Working KAGGLE_VIT_BATCHED_STANDALONE.py Template
========================================================================

This version includes the fixes that made T4 tests successful:
- Learning rate scheduler (CosineAnnealingLR)
- Gradient clipping
- Better configuration

Run this on Kaggle P100 for accurate comparison with T4 results.
"""

# Copy the entire KAGGLE_VIT_BATCHED_STANDALONE.py implementation
# Then run baseline vs AST comparison

import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# SUNDEW CONFIG & ALGORITHM
# ============================================================================

@dataclass
class SundewConfig:
    """Sundew adaptive gating configuration."""
    activation_threshold: float = 0.6
    target_activation_rate: float = 0.06
    gate_temperature: float = 0.15
    energy_pressure: float = 0.3
    max_energy: float = 100.0
    dormancy_regen: tuple = (1.0, 3.0)
    adapt_kp: float = 0.12
    adapt_ki: float = 0.008


class SundewAlgorithm:
    """Lightweight Sundew gating."""

    def __init__(self, config: SundewConfig):
        self.config = config
        self.threshold = config.activation_threshold
        self.energy = config.max_energy
        self.integral_error = 0.0
        self.total_processed = 0
        self.total_activated = 0

    def process(self, features: Dict[str, float]) -> Optional[Dict]:
        """Gate decision based on significance."""
        significance = features.get("significance", 0.5)

        # Adaptive threshold (PI control)
        self.total_processed += 1
        activation_rate = self.total_activated / max(self.total_processed, 1)
        error = self.config.target_activation_rate - activation_rate
        self.integral_error += error

        # Threshold adjustment
        self.threshold += self.config.adapt_kp * error + self.config.adapt_ki * self.integral_error
        self.threshold = max(0.1, min(0.9, self.threshold))

        # Energy regeneration
        regen = np.random.uniform(*self.config.dormancy_regen)
        self.energy = min(self.config.max_energy, self.energy + regen)

        # Gate decision
        if significance > self.threshold:
            gate_prob = 1.0 / (1.0 + np.exp(-(significance - self.threshold) / self.config.gate_temperature))

            if np.random.random() < gate_prob and self.energy > 10.0:
                self.energy -= 10.0
                self.total_activated += 1
                return {"activated": True}

        return None


# ============================================================================
# VISION TRANSFORMER
# ============================================================================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


# ============================================================================
# BATCHED ADAPTIVE TRAINER (FROM WORKING STANDALONE)
# ============================================================================

class ImprovedBatchedTrainer:
    """Improved trainer with scheduler and gradient clipping."""

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.01)
        )

        # Learning rate scheduler (KEY ADDITION!)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 10),
        )

        # Loss function (per-sample)
        self.criterion = nn.CrossEntropyLoss(reduction='none')

        # Sundew gating
        self.use_gating = config.get("use_gating", True)
        if self.use_gating:
            sundew_cfg = SundewConfig(
                target_activation_rate=config.get("target_activation_rate", 0.06),
                activation_threshold=config.get("activation_threshold", 0.6),
                gate_temperature=config.get("gate_temperature", 0.15),
                energy_pressure=config.get("energy_pressure", 0.3),
                adapt_kp=config.get("adapt_kp", 0.12),
                adapt_ki=config.get("adapt_ki", 0.008),
            )
            self.sundew_algo = SundewAlgorithm(sundew_cfg)
        else:
            self.sundew_algo = None

        # Metrics
        self.metrics = {
            "epoch_times": [],
            "train_losses": [],
            "val_accuracies": [],
            "activation_rates": [],
        }

    def _compute_batch_significance(self, inputs, targets):
        """Vectorized significance computation."""
        with torch.no_grad():
            outputs = self.model(inputs)
            losses = self.criterion(outputs, targets)

        mean_intensity = inputs.mean(dim=[1, 2, 3])
        std_intensity = inputs.std(dim=[1, 2, 3])

        significance = (
            0.5 * (losses / (losses.max() + 1e-6)) +
            0.3 * (std_intensity / (std_intensity.max() + 1e-6)) +
            0.2 * torch.ones_like(losses)
        )
        return significance, losses

    def train_epoch(self):
        """Train one epoch with optional gating."""
        self.model.train()
        epoch_loss = 0.0
        samples_processed = 0
        samples_skipped = 0
        num_batches = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size = inputs.size(0)

            if self.use_gating:
                # Compute significance
                significance, losses = self._compute_batch_significance(inputs, targets)

                # Gate decisions
                gate_decisions = []
                sig_np = significance.cpu().numpy()
                loss_np = losses.cpu().numpy()

                for i in range(batch_size):
                    features = {
                        "significance": float(sig_np[i]),
                        "loss": float(loss_np[i])
                    }
                    result = self.sundew_algo.process(features)
                    gate_decisions.append(result is not None)

                # Mask
                active_mask = torch.tensor(gate_decisions, device=self.device)
                num_active = active_mask.sum().item()

                if num_active == 0:
                    samples_skipped += batch_size
                    continue

                inputs = inputs[active_mask]
                targets = targets[active_mask]
                samples_processed += num_active
                samples_skipped += (batch_size - num_active)
            else:
                samples_processed += batch_size

            # Forward + backward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets).mean()
            loss.backward()

            # GRADIENT CLIPPING (KEY ADDITION!)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        # SCHEDULER STEP (KEY ADDITION!)
        self.scheduler.step()

        avg_loss = epoch_loss / max(num_batches, 1)
        total_samples = samples_processed + samples_skipped
        activation_rate = samples_processed / max(total_samples, 1)

        return avg_loss, activation_rate

    def validate(self):
        """Validate model."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def train(self, epochs):
        """Full training loop."""
        for epoch in range(epochs):
            epoch_start = time.time()

            train_loss, act_rate = self.train_epoch()
            val_acc = self.validate()

            epoch_time = time.time() - epoch_start

            self.metrics["epoch_times"].append(epoch_time)
            self.metrics["train_losses"].append(train_loss)
            self.metrics["val_accuracies"].append(val_acc)
            self.metrics["activation_rates"].append(act_rate)

            print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:5.2f}% | Act: {act_rate:5.1%} | "
                  f"Time: {epoch_time:5.1f}s")

        return self.metrics


# ============================================================================
# DATA LOADERS
# ============================================================================

def get_cifar10_loaders(batch_size=128):
    """CIFAR-10 with proper augmentation."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader


# ============================================================================
# MAIN COMPARISON TEST
# ============================================================================

def main():
    print("="*70)
    print("P100 GPU - IMPROVED TEST (With Scheduler & Gradient Clipping)")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    train_loader, val_loader = get_cifar10_loaders(batch_size=128)

    # Test 1: Baseline (no gating)
    print("\n" + "="*70)
    print("TEST 1: BASELINE (No Gating)")
    print("="*70)

    model_baseline = VisionTransformer(embed_dim=256, depth=6, num_heads=8)
    config_baseline = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "epochs": 1,
        "use_gating": False
    }
    trainer_baseline = ImprovedBatchedTrainer(model_baseline, train_loader, val_loader, device, config_baseline)
    metrics_baseline = trainer_baseline.train(epochs=1)

    print(f"\n✅ Baseline Complete:")
    print(f"   Time/Epoch: {metrics_baseline['epoch_times'][0]:.1f}s")
    print(f"   Val Accuracy: {metrics_baseline['val_accuracies'][0]:.2f}%")

    # Test 2: AST with gating
    print("\n" + "="*70)
    print("TEST 2: AST with 6% Activation Target")
    print("="*70)

    model_ast = VisionTransformer(embed_dim=256, depth=6, num_heads=8)
    config_ast = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "epochs": 1,
        "use_gating": True,
        "target_activation_rate": 0.06,
        "activation_threshold": 0.6,  # T4-calibrated value
        "gate_temperature": 0.15,
        "adapt_kp": 0.12,
        "adapt_ki": 0.008,
    }
    trainer_ast = ImprovedBatchedTrainer(model_ast, train_loader, val_loader, device, config_ast)
    metrics_ast = trainer_ast.train(epochs=1)

    print(f"\n✅ AST Complete:")
    print(f"   Time/Epoch: {metrics_ast['epoch_times'][0]:.1f}s")
    print(f"   Val Accuracy: {metrics_ast['val_accuracies'][0]:.2f}%")
    print(f"   Activation Rate: {metrics_ast['activation_rates'][0]:.1%}")

    # Test 3: Full 10-epoch training
    print("\n" + "="*70)
    print("TEST 3: Full 10-Epoch Training with AST")
    print("="*70)

    model_full = VisionTransformer(embed_dim=256, depth=6, num_heads=8)
    config_full = {
        "lr": 1e-3,
        "weight_decay": 0.01,
        "epochs": 10,
        "use_gating": True,
        "target_activation_rate": 0.06,
        "activation_threshold": 0.6,
        "gate_temperature": 0.15,
        "adapt_kp": 0.12,
        "adapt_ki": 0.008,
    }
    trainer_full = ImprovedBatchedTrainer(model_full, train_loader, val_loader, device, config_full)
    metrics_full = trainer_full.train(epochs=10)

    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Baseline time/epoch: {metrics_baseline['epoch_times'][0]:.1f}s")
    print(f"AST time/epoch: {metrics_ast['epoch_times'][0]:.1f}s")
    print(f"Speedup: {metrics_baseline['epoch_times'][0] / metrics_ast['epoch_times'][0]:.1f}×")
    print(f"\nBaseline accuracy (1 epoch): {metrics_baseline['val_accuracies'][0]:.2f}%")
    print(f"AST accuracy (1 epoch): {metrics_ast['val_accuracies'][0]:.2f}%")
    print(f"AST final accuracy (10 epochs): {metrics_full['val_accuracies'][-1]:.2f}%")
    print(f"AST best accuracy (10 epochs): {max(metrics_full['val_accuracies']):.2f}%")
    print(f"\nAST average activation rate: {np.mean(metrics_ast['activation_rates']):.1%}")
    print(f"AST 10-epoch average activation: {np.mean(metrics_full['activation_rates']):.1%}")


if __name__ == "__main__":
    main()
