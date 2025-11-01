"""
🚨 EMERGENCY FIX - Use RAW significance scores (NO normalization!)
====================================================================
PROBLEM: Normalizing by max/mean destroys variance → all scores ≈ 1.0
SOLUTION: Use RAW loss + entropy → let PI controller find threshold

Expected: Accuracy 15-25% (better than 0.98%!)
Runtime: 30 min for 5 epochs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder  # FASTER than custom Dataset!
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experiment configuration"""
    data_dir = "/kaggle/input/imagenet100/ImageNet100"
    num_classes = 100
    image_size = 224

    # Training
    batch_size = 64
    num_epochs = 5
    learning_rate = 0.001
    weight_decay = 1e-4

    # AST - FIXED VALUES
    target_activation_rate = 0.10
    initial_threshold = 3.5  # Higher for RAW scores

    # PI Controller - GENTLER
    adapt_kp = 0.01  # Reduced
    adapt_ki = 0.0005  # Reduced
    ema_alpha = 0.15  # Slower

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4  # Increased!
    pin_memory = True

# ============================================================================
# DATASET
# ============================================================================

def get_dataloaders(config):
    """Create ImageNet-100 dataloaders using ImageFolder (FASTER!)"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Use ImageFolder (10× faster than custom Dataset!)
    train_dataset = ImageFolder(str(Path(config.data_dir) / 'train'), transform=train_transform)
    val_dataset = ImageFolder(str(Path(config.data_dir) / 'val'), transform=val_transform)

    print(f"Loaded {len(train_dataset):,} training images")
    print(f"Loaded {len(val_dataset):,} validation images")
    print(f"Found {len(train_dataset.classes)} classes")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers,
                           pin_memory=config.pin_memory)

    return train_loader, val_loader

# ============================================================================
# SUNDEW ALGORITHM - RAW SIGNIFICANCE (NO NORMALIZATION!)
# ============================================================================

class SundewAlgorithm:
    """Adaptive gating with RAW significance scores"""

    def __init__(self, config):
        self.target_activation_rate = config.target_activation_rate
        self.activation_threshold = config.initial_threshold

        # PI controller
        self.kp = config.adapt_kp
        self.ki = config.adapt_ki
        self.integral_error = 0.0

        # EMA smoothing
        self.ema_alpha = config.ema_alpha
        self.activation_rate_ema = config.target_activation_rate

        # Energy tracking
        self.energy_per_activation = config.energy_per_activation
        self.energy_per_skip = config.energy_per_skip
        self.total_baseline_energy = 0.0
        self.total_actual_energy = 0.0

    def compute_significance(self, losses, outputs):
        """
        🔧 EMERGENCY FIX: RAW SCORES (NO NORMALIZATION!)

        Problem: Normalization by max/mean → all scores collapse to ~1.0
        Solution: Use RAW loss + RAW entropy → real variance!
        """
        # Factor 1: RAW loss (no normalization!)
        # Cross-entropy loss for ImageNet-100: typically 2.0-6.0
        # Higher = harder sample
        loss_component = losses  # Keep RAW!

        # Factor 2: RAW entropy
        # Max entropy for 100 classes = log(100) ≈ 4.6
        # Higher entropy = more uncertain = more important
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # Weighted combination: 80% loss, 20% entropy
        # Typical range: 2.0×0.8 + 4.6×0.2 = 1.6 + 0.92 = 2.5 to 6.0
        significance = 0.8 * loss_component + 0.2 * entropy

        return significance

    def select_samples(self, losses, outputs):
        """Select important samples using RAW significance"""
        batch_size = losses.size(0)

        # Compute RAW significance scores
        significance = self.compute_significance(losses, outputs)

        # Select samples above threshold
        active_mask = significance > self.activation_threshold
        num_active = active_mask.sum().item()

        # Fallback mechanism
        if num_active == 0:
            # Select top 5% by significance
            k = max(2, batch_size // 20)
            _, top_indices = torch.topk(significance, k)
            active_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            active_mask[top_indices] = True
            num_active = k

        # Update activation rate EMA
        current_activation_rate = num_active / batch_size
        self.activation_rate_ema = (
            self.ema_alpha * current_activation_rate +
            (1 - self.ema_alpha) * self.activation_rate_ema
        )

        # PI controller
        error = self.activation_rate_ema - self.target_activation_rate
        proportional = self.kp * error

        # Integral with anti-windup
        if 0.1 < self.activation_threshold < 20.0:
            self.integral_error += error
            self.integral_error = max(-100, min(100, self.integral_error))
        else:
            self.integral_error *= 0.90

        # Update threshold
        new_threshold = self.activation_threshold + proportional + self.ki * self.integral_error
        self.activation_threshold = max(0.1, min(20.0, new_threshold))

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

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch_ast(model, train_loader, criterion, optimizer, sundew, config, epoch):
    """Train one epoch with AST"""
    model.train()

    running_loss = 0.0
    correct = 0
    total_samples = 0
    total_active = 0

    epoch_stats = {
        'activation_rates': [],
        'thresholds': [],
        'energy_savings': [],
    }

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(config.device)
        labels = labels.to(config.device)
        batch_size = images.size(0)

        # Forward pass to get losses and outputs
        with torch.no_grad():
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')

        # Select important samples (RAW significance)
        active_mask, energy_info = sundew.select_samples(losses, outputs)

        # Train only on active samples
        if active_mask.sum() > 0:
            active_images = images[active_mask]
            active_labels = labels[active_mask]

            optimizer.zero_grad()
            active_outputs = model(active_images)
            loss = criterion(active_outputs, active_labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * active_mask.sum().item()

            # Track accuracy
            _, predicted = active_outputs.max(1)
            correct += predicted.eq(active_labels).sum().item()
            total_active += active_mask.sum().item()

        total_samples += batch_size

        # Track stats
        epoch_stats['activation_rates'].append(energy_info['activation_rate_ema'])
        epoch_stats['thresholds'].append(energy_info['threshold'])
        epoch_stats['energy_savings'].append(energy_info['energy_savings'])

        # Print progress
        if (batch_idx + 1) % 100 == 0:  # Every 100 batches
            train_acc = 100.0 * correct / max(total_active, 1)
            print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                  f"Act: {100*energy_info['activation_rate_ema']:5.1f}% | "
                  f"Train Acc: {train_acc:5.1f}% | "
                  f"⚡ Energy: {energy_info['energy_savings']:5.1f}% | "
                  f"Threshold: {energy_info['threshold']:.2f}")

    avg_loss = running_loss / max(total_active, 1)
    avg_activation = total_active / total_samples
    train_accuracy = 100.0 * correct / max(total_active, 1)

    return avg_loss, avg_activation, train_accuracy, epoch_stats

def validate(model, val_loader, criterion, config):
    """Validation loop"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(config.device)
            labels = labels.to(config.device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function"""

    print("=" * 70)
    print("🚨 EMERGENCY FIX - RAW Significance Scoring")
    print("Testing for 5 epochs")
    print("=" * 70)

    config = Config()
    print(f"Device: {config.device}")
    print(f"Target activation rate: {config.target_activation_rate*100:.1f}%")
    print(f"Using RAW significance (NO normalization!)")
    print(f"num_workers: {config.num_workers}")
    print()

    # Load data
    train_loader, val_loader = get_dataloaders(config)
    print()

    # Load model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(config.device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded ResNet50 ({num_params:.1f}M params)")
    print()

    # Setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    sundew = SundewAlgorithm(config)

    # Training loop
    start_time = time.time()
    best_accuracy = 0.0
    all_epoch_stats = []

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        print()
        print("=" * 60)
        print(f"Epoch {epoch}/{config.num_epochs}")
        print("=" * 60)

        # Train
        train_loss, train_activation, train_acc, epoch_stats = train_epoch_ast(
            model, train_loader, criterion, optimizer, sundew, config, epoch
        )
        all_epoch_stats.append(epoch_stats)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, config)

        # Energy savings
        energy_savings = 0.0
        if sundew.total_baseline_energy > 0:
            energy_savings = ((sundew.total_baseline_energy - sundew.total_actual_energy) /
                             sundew.total_baseline_energy * 100)

        epoch_time = time.time() - epoch_start

        # Print summary
        print()
        print(f"✅ Epoch {epoch} | "
              f"Val Acc: {val_accuracy:5.2f}% | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Loss: {train_loss:.4f} | "
              f"⚡ Energy: {energy_savings:5.1f}% | "
              f"Time: {epoch_time:.1f}s")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

    total_time = time.time() - start_time

    # Final results
    print()
    print("=" * 70)
    print("🚨 EMERGENCY FIX RESULTS")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Total Energy Savings: {energy_savings:.2f}%")
    print(f"Average Activation Rate: {100*train_activation:.2f}%")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print()

    # Comparison
    print("📊 COMPARISON:")
    print(f"   Broken version: 0.98%")
    print(f"   This version:   {best_accuracy:.2f}%")
    print(f"   Improvement:    {best_accuracy - 0.98:+.2f}%")
    print()

    if best_accuracy >= 15:
        print("✅ MAJOR SUCCESS! RAW significance works!")
        print("   → Proceed to PRODUCTION version")
    elif best_accuracy >= 8:
        print("⚠️  PARTIAL SUCCESS - needs more tuning")
    else:
        print("❌ STILL BROKEN - deeper issues")
    print("=" * 70)

if __name__ == "__main__":
    main()
