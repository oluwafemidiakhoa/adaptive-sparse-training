#!/usr/bin/env python3
"""
Kaggle P100 GPU Test Suite - Comprehensive Benchmarking
========================================================

Tests to run with 25 hours P100 GPU allocation:
1. Baseline timing (no gating)
2. AST with different activation rates (3%, 6%, 10%, 20%)
3. Larger ViT models (scaling test)
4. Full 10-epoch training (final accuracy)

Expected P100 performance: 2-3× faster than T4
Estimated total runtime: ~8-10 hours (leaves buffer for reruns)

Usage in Kaggle:
    1. Create new notebook with P100 GPU
    2. Copy this entire file to a code cell
    3. Run the cell
    4. Results saved to CSV + visualizations
"""

import time
import json
from typing import Dict, List
from dataclasses import dataclass
import csv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ============================================================================
# SUNDEW CONFIG & ALGORITHM (Embedded)
# ============================================================================

@dataclass
class SundewConfig:
    """Sundew adaptive gating configuration."""
    activation_threshold: float = 0.7
    target_activation_rate: float = 0.06
    gate_temperature: float = 0.15
    energy_pressure: float = 0.4
    max_energy: float = 100.0
    dormancy_regen: tuple = (1.0, 3.0)
    adapt_kp: float = 0.15
    adapt_ki: float = 0.01


class SundewAlgorithm:
    """Lightweight Sundew gating for adaptive processing."""

    def __init__(self, config: SundewConfig):
        self.config = config
        self.threshold = config.activation_threshold
        self.energy = config.max_energy
        self.integral_error = 0.0
        self.total_processed = 0
        self.total_activated = 0

    def process(self, features: Dict[str, float]) -> bool:
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
                return True

        return False


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
# BATCHED ADAPTIVE TRAINER
# ============================================================================

class BatchedAdaptiveTrainer:
    """Batched trainer with Sundew gating."""

    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.get("lr", 1e-3),
            weight_decay=config.get("weight_decay", 0.01)
        )
        self.criterion = config.get("criterion", nn.CrossEntropyLoss())

        # Sundew gating
        self.use_gating = config.get("use_gating", True)
        if self.use_gating:
            sundew_cfg = SundewConfig(
                target_activation_rate=config.get("target_activation_rate", 0.06),
                activation_threshold=config.get("activation_threshold", 0.7),
                gate_temperature=config.get("gate_temperature", 0.15),
                energy_pressure=config.get("energy_pressure", 0.4),
                adapt_kp=config.get("adapt_kp", 0.15),
                adapt_ki=config.get("adapt_ki", 0.01),
            )
            self.gating = SundewAlgorithm(sundew_cfg)
        else:
            self.gating = None

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
            losses = nn.functional.cross_entropy(outputs, targets, reduction='none')

        mean_intensity = inputs.mean(dim=[1, 2, 3])
        std_intensity = inputs.std(dim=[1, 2, 3])

        significance = (
            0.5 * (losses / (losses.max() + 1e-6)) +
            0.3 * (std_intensity / (std_intensity.max() + 1e-6)) +
            0.2 * torch.ones_like(losses)
        )
        return significance

    def train_epoch(self):
        """Train one epoch with optional gating."""
        self.model.train()
        total_loss = 0.0
        total_activated = 0
        total_samples = 0
        num_batches = 0

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            batch_size_orig = len(inputs)

            if self.use_gating:
                # Compute significance and gate
                significance = self._compute_batch_significance(inputs, targets)

                # Gate decisions
                activated_mask = torch.zeros(len(inputs), dtype=torch.bool, device=self.device)
                for i in range(len(inputs)):
                    if self.gating.process({"significance": significance[i].item()}):
                        activated_mask[i] = True

                if activated_mask.sum() == 0:
                    total_samples += batch_size_orig
                    continue  # Skip batch if no activations

                inputs = inputs[activated_mask]
                targets = targets[activated_mask]
                total_activated += activated_mask.sum().item()
                total_samples += batch_size_orig
            else:
                total_activated += len(inputs)
                total_samples += len(inputs)

            # Forward + backward
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        activation_rate = total_activated / max(total_samples, 1) if self.use_gating else 1.0

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

            print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | Act Rate: {act_rate:.1%} | "
                  f"Time: {epoch_time:.1f}s")

        return self.metrics


# ============================================================================
# DATA LOADERS
# ============================================================================

def get_cifar10_loaders(batch_size=128):
    """CIFAR-10 data loaders."""
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
# TEST FUNCTIONS
# ============================================================================

def test_baseline(device, epochs=1):
    """Test 1: Baseline (no gating)."""
    print("\n" + "="*70)
    print("TEST 1: BASELINE (No Gating)")
    print("="*70)

    train_loader, val_loader = get_cifar10_loaders(batch_size=128)
    model = VisionTransformer(embed_dim=256, depth=6, num_heads=8)

    config = {"lr": 1e-3, "use_gating": False, "criterion": nn.CrossEntropyLoss()}
    trainer = BatchedAdaptiveTrainer(model, train_loader, val_loader, device, config)

    metrics = trainer.train(epochs=epochs)

    result = {
        "test_name": "baseline",
        "epochs": epochs,
        "avg_epoch_time": np.mean(metrics["epoch_times"]),
        "final_val_acc": metrics["val_accuracies"][-1],
        "activation_rate": 1.0,
        "target_rate": 1.0,
    }

    return result


def test_ast_activation_rate(device, target_rate, epochs=1):
    """Test AST with specific activation rate."""
    print(f"\n" + "="*70)
    print(f"TEST: AST with {target_rate:.0%} Activation Rate")
    print("="*70)

    train_loader, val_loader = get_cifar10_loaders(batch_size=128)
    model = VisionTransformer(embed_dim=256, depth=6, num_heads=8)

    # Tuned parameters for different rates
    params = {
        0.03: {"threshold": 0.80, "temp": 0.08, "kp": 0.20, "ki": 0.015},
        0.06: {"threshold": 0.75, "temp": 0.10, "kp": 0.18, "ki": 0.012},
        0.10: {"threshold": 0.70, "temp": 0.12, "kp": 0.15, "ki": 0.010},
        0.20: {"threshold": 0.65, "temp": 0.15, "kp": 0.12, "ki": 0.008},
    }

    p = params.get(target_rate, params[0.06])

    config = {
        "lr": 1e-3,
        "use_gating": True,
        "target_activation_rate": target_rate,
        "activation_threshold": p["threshold"],
        "gate_temperature": p["temp"],
        "adapt_kp": p["kp"],
        "adapt_ki": p["ki"],
        "criterion": nn.CrossEntropyLoss(),
    }

    trainer = BatchedAdaptiveTrainer(model, train_loader, val_loader, device, config)
    metrics = trainer.train(epochs=epochs)

    result = {
        "test_name": f"ast_{int(target_rate*100)}pct",
        "target_rate": target_rate,
        "actual_rate": np.mean(metrics["activation_rates"]),
        "epochs": epochs,
        "avg_epoch_time": np.mean(metrics["epoch_times"]),
        "final_val_acc": metrics["val_accuracies"][-1],
    }

    return result


def test_large_vit(device, epochs=1):
    """Test larger ViT (d_model=512, 12 layers)."""
    print("\n" + "="*70)
    print("TEST: Large ViT (d512_L12) with AST")
    print("="*70)

    train_loader, val_loader = get_cifar10_loaders(batch_size=64)
    model = VisionTransformer(embed_dim=512, depth=12, num_heads=16)

    config = {
        "lr": 1e-3,
        "use_gating": True,
        "target_activation_rate": 0.06,
        "activation_threshold": 0.75,
        "gate_temperature": 0.10,
        "adapt_kp": 0.18,
        "adapt_ki": 0.012,
        "criterion": nn.CrossEntropyLoss(),
    }

    trainer = BatchedAdaptiveTrainer(model, train_loader, val_loader, device, config)
    metrics = trainer.train(epochs=epochs)

    result = {
        "test_name": "large_vit",
        "model_size": "d512_L12",
        "epochs": epochs,
        "avg_epoch_time": np.mean(metrics["epoch_times"]),
        "final_val_acc": metrics["val_accuracies"][-1],
        "activation_rate": np.mean(metrics["activation_rates"]),
        "target_rate": 0.06,
    }

    return result


def test_full_training(device, epochs=10):
    """Test full 10-epoch training."""
    print("\n" + "="*70)
    print("TEST: Full 10-Epoch Training with AST")
    print("="*70)

    train_loader, val_loader = get_cifar10_loaders(batch_size=128)
    model = VisionTransformer(embed_dim=256, depth=6, num_heads=8)

    config = {
        "lr": 1e-3,
        "use_gating": True,
        "target_activation_rate": 0.06,
        "activation_threshold": 0.75,
        "gate_temperature": 0.10,
        "adapt_kp": 0.18,
        "adapt_ki": 0.012,
        "criterion": nn.CrossEntropyLoss(),
    }

    trainer = BatchedAdaptiveTrainer(model, train_loader, val_loader, device, config)
    metrics = trainer.train(epochs=epochs)

    result = {
        "test_name": "full_training",
        "epochs": epochs,
        "avg_epoch_time": np.mean(metrics["epoch_times"]),
        "final_val_acc": metrics["val_accuracies"][-1],
        "best_val_acc": max(metrics["val_accuracies"]),
        "activation_rate": np.mean(metrics["activation_rates"]),
        "target_rate": 0.06,
        "all_metrics": metrics,
    }

    return result


# ============================================================================
# MAIN TEST SUITE
# ============================================================================

def test_all():
    """Run complete P100 test suite."""
    print("="*70)
    print("KAGGLE P100 GPU BENCHMARK SUITE")
    print("="*70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = []
    start_time = time.time()

    # Test 1: Baseline
    print("\n📊 Running Test 1/7: Baseline...")
    results.append(test_baseline(device, epochs=1))

    # Test 2-5: Different activation rates
    for i, rate in enumerate([0.03, 0.06, 0.10, 0.20], start=2):
        print(f"\n📊 Running Test {i}/7: AST @ {int(rate*100)}% activation...")
        results.append(test_ast_activation_rate(device, rate, epochs=1))

    # Test 6: Large ViT
    print("\n📊 Running Test 6/7: Large ViT...")
    results.append(test_large_vit(device, epochs=1))

    # Test 7: Full training
    print("\n📊 Running Test 7/7: Full 10-epoch training...")
    full_result = test_full_training(device, epochs=10)
    results.append(full_result)

    total_time = time.time() - start_time

    # Save and visualize results
    save_results(results)
    plot_results(results)

    # Print summary
    print("\n" + "="*70)
    print("✅ ALL TESTS COMPLETE")
    print("="*70)
    print(f"Total runtime: {total_time/3600:.2f} hours")
    print(f"Results saved to: p100_benchmark_results.csv")
    print(f"Visualizations: p100_benchmark_plots.png")
    print("\nQuick Summary:")
    print(f"  Baseline time/epoch: {results[0]['avg_epoch_time']:.1f}s")
    print(f"  AST@6% time/epoch: {results[2]['avg_epoch_time']:.1f}s")
    print(f"  Speedup: {results[0]['avg_epoch_time']/results[2]['avg_epoch_time']:.1f}×")
    print(f"  Final accuracy (10 epochs): {results[-1]['final_val_acc']:.2f}%")

    return results


def save_results(results):
    """Save results to CSV."""
    with open("p100_benchmark_results.csv", "w", newline="") as f:
        fieldnames = ["test_name", "epochs", "avg_epoch_time", "final_val_acc",
                      "activation_rate", "target_rate", "model_size", "best_val_acc"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)
    print("\n💾 Results saved to CSV")


def plot_results(results):
    """Plot benchmark results."""
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Filter out full training for first plots
    short_results = [r for r in results if "all_metrics" not in r]

    # Panel 1: Epoch time comparison
    ax1 = fig.add_subplot(gs[0, 0])
    test_names = [r["test_name"] for r in short_results]
    epoch_times = [r["avg_epoch_time"] for r in short_results]
    colors = ['red' if r["test_name"] == "baseline" else 'steelblue' for r in short_results]
    ax1.bar(range(len(test_names)), epoch_times, color=colors)
    ax1.set_xticks(range(len(test_names)))
    ax1.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax1.set_ylabel('Time (s)', fontsize=11)
    ax1.set_title('Average Epoch Time', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Speedup vs baseline
    ax2 = fig.add_subplot(gs[0, 1])
    baseline_time = results[0]["avg_epoch_time"]
    speedups = [baseline_time / r["avg_epoch_time"] for r in short_results]
    ax2.bar(range(len(test_names)), speedups, color='green')
    ax2.set_xticks(range(len(test_names)))
    ax2.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Speedup (×)', fontsize=11)
    ax2.set_title('Speedup vs Baseline', fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Baseline')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Accuracy comparison
    ax3 = fig.add_subplot(gs[0, 2])
    accuracies = [r["final_val_acc"] for r in short_results]
    ax3.bar(range(len(test_names)), accuracies, color='orange')
    ax3.set_xticks(range(len(test_names)))
    ax3.set_xticklabels(test_names, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Accuracy (%)', fontsize=11)
    ax3.set_title('Validation Accuracy (1 epoch)', fontsize=12, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Activation rate vs target
    ax4 = fig.add_subplot(gs[1, 0])
    ast_results = [r for r in short_results if r.get("target_rate") and r["target_rate"] < 1.0]
    if ast_results:
        target_rates = [r["target_rate"] * 100 for r in ast_results]
        actual_rates = [r.get("actual_rate", r["activation_rate"]) * 100 for r in ast_results]
        x = np.arange(len(target_rates))
        width = 0.35
        ax4.bar(x - width/2, target_rates, width, label='Target', color='lightblue')
        ax4.bar(x + width/2, actual_rates, width, label='Actual', color='steelblue')
        ax4.set_ylabel('Activation Rate (%)', fontsize=11)
        ax4.set_title('Target vs Actual Activation Rate', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{int(r*100)}%" for r in [r["target_rate"] for r in ast_results]])
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

    # Panel 5: Full training progress
    ax5 = fig.add_subplot(gs[1, 1])
    full_training = [r for r in results if r.get("all_metrics")]
    if full_training:
        metrics = full_training[0]["all_metrics"]
        epochs = range(1, len(metrics["val_accuracies"]) + 1)
        ax5.plot(epochs, metrics["val_accuracies"], marker='o', color='green', linewidth=2, markersize=6)
        ax5.set_xlabel('Epoch', fontsize=11)
        ax5.set_ylabel('Validation Accuracy (%)', fontsize=11)
        ax5.set_title('10-Epoch Training Progress', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)

    # Panel 6: Summary table
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_text = "📊 P100 GPU Benchmark Summary\n"
    summary_text += "="*35 + "\n\n"
    baseline = results[0]
    ast_6pct = [r for r in results if r.get("target_rate") == 0.06]
    if ast_6pct:
        ast_6pct = ast_6pct[0]
        speedup = baseline["avg_epoch_time"] / ast_6pct["avg_epoch_time"]
        summary_text += f"Baseline:\n"
        summary_text += f"  Time: {baseline['avg_epoch_time']:.1f}s/epoch\n"
        summary_text += f"  Acc:  {baseline['final_val_acc']:.1f}%\n\n"
        summary_text += f"AST @ 6% activation:\n"
        summary_text += f"  Time: {ast_6pct['avg_epoch_time']:.1f}s/epoch\n"
        summary_text += f"  Acc:  {ast_6pct['final_val_acc']:.1f}%\n"
        summary_text += f"  Speedup: {speedup:.1f}×\n\n"

    if full_training:
        ft = full_training[0]
        summary_text += f"10-Epoch Training:\n"
        summary_text += f"  Final:  {ft['final_val_acc']:.1f}%\n"
        summary_text += f"  Best:   {ft['best_val_acc']:.1f}%\n"
        summary_text += f"  Avg Time: {ft['avg_epoch_time']:.1f}s\n"

    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
             fontsize=10, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Kaggle P100 GPU - Adaptive Sparse Training Benchmarks',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.savefig('p100_benchmark_plots.png', dpi=150, bbox_inches='tight')
    print("📈 Visualizations saved")


if __name__ == "__main__":
    print("\n🚀 Starting Kaggle P100 GPU Benchmark Suite...")
    print("⏱️  Estimated runtime: 8-10 hours\n")
    results = test_all()
