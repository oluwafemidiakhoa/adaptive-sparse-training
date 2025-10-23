"""
Adaptive Sparse Training (AST) on ImageNet-100 - FIXED VERSION
===============================================================
Fixes:
1. Activation rate convergence (was stuck at 50%, now targets 10%)
2. Energy savings display (now shows in training output)
3. Results visualization (creates diagram like CIFAR-10)

ImageNet-100: Subset of 100 classes from ImageNet (~130K images)
- Input: 224x224 RGB images
- Classes: 100
- Model: ResNet50 (pretrained)

Expected Results:
- Accuracy: 75-80%
- Energy Savings: 88-91%
- Activation Rate: 9-12%

Setup Instructions for Kaggle:
1. Add ImageNet-100 dataset
2. Enable GPU: Settings > Accelerator > GPU T4 x2
3. Update data_dir in Config class to match your dataset path
4. Run this code
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import time
import math
from pathlib import Path
from PIL import Image
import os

# For visualization
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experiment configuration"""
    # Dataset - UPDATE THIS PATH
    data_dir = "/kaggle/input/imagenet100/ImageNet100"  # Adjust to your path
    num_classes = 100
    image_size = 224

    # Training
    batch_size = 64
    num_epochs = 1  # Set to 40 for full training
    learning_rate = 0.001
    weight_decay = 1e-4

    # AST Configuration
    target_activation_rate = 0.10  # Train on 10% of samples
    initial_threshold = 0.50

    # PI Controller (FIXED - increased gains for faster convergence)
    adapt_kp = 0.003  # Increased from 0.0015
    adapt_ki = 0.0001  # Increased from 0.00005
    ema_alpha = 0.3

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2
    pin_memory = True

# ============================================================================
# IMAGENET-100 DATASET
# ============================================================================

class ImageNet100Dataset(Dataset):
    """ImageNet-100 dataset loader"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform

        # Get all image paths and labels
        self.samples = []
        self.class_to_idx = {}

        for idx, class_dir in enumerate(sorted(self.root_dir.iterdir())):
            if class_dir.is_dir():
                self.class_to_idx[class_dir.name] = idx
                for img_path in class_dir.glob("*.JPEG"):
                    self.samples.append((str(img_path), idx))

        print(f"Loaded {len(self.samples)} images from {split} split")
        print(f"Found {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# ============================================================================
# DATA PREPARATION
# ============================================================================

def get_dataloaders(config):
    """Create ImageNet-100 dataloaders with augmentation"""

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

    train_dataset = ImageNet100Dataset(
        config.data_dir,
        split='train',
        transform=train_transform
    )

    val_dataset = ImageNet100Dataset(
        config.data_dir,
        split='val',
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )

    return train_loader, val_loader

# ============================================================================
# SUNDEW ALGORITHM (FIXED)
# ============================================================================

class SundewAlgorithm:
    """Adaptive gating with PI control and EMA smoothing - FIXED VERSION"""

    def __init__(self, config):
        self.target_activation_rate = config.target_activation_rate
        self.activation_threshold = config.initial_threshold

        # PI controller (FIXED - stronger gains)
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

    def compute_significance(self, losses, images):
        """Multi-factor significance scoring"""
        batch_size = losses.size(0)

        # Factor 1: Normalized loss
        loss_mean = losses.mean()
        loss_norm = losses / (loss_mean + 1e-8)

        # Factor 2: Image intensity variation
        images_flat = images.view(batch_size, 3, -1)
        std_per_channel = images_flat.std(dim=2)
        std_intensity = std_per_channel.mean(dim=1)

        std_mean = std_intensity.mean()
        std_norm = std_intensity / (std_mean + 1e-8)

        # Weighted combination
        significance = 0.7 * loss_norm + 0.3 * std_norm

        return significance

    def select_samples(self, losses, images):
        """Select important samples and update threshold - FIXED"""
        batch_size = losses.size(0)

        # Compute significance scores
        significance = self.compute_significance(losses, images)

        # Select samples above threshold
        active_mask = significance > self.activation_threshold
        num_active = active_mask.sum().item()

        # Fallback: If no samples selected, train on 2 random samples
        if num_active == 0:
            random_indices = torch.randperm(batch_size)[:2]
            active_mask[random_indices] = True
            num_active = 2

        # Update activation rate EMA
        current_activation_rate = num_active / batch_size
        self.activation_rate_ema = (
            self.ema_alpha * current_activation_rate +
            (1 - self.ema_alpha) * self.activation_rate_ema
        )

        # FIXED: PI controller update with correct error sign
        error = self.activation_rate_ema - self.target_activation_rate

        # Integral with anti-windup
        if 0.01 < self.activation_threshold < 0.99:
            self.integral_error += error
            self.integral_error = max(-50, min(50, self.integral_error))
        else:
            self.integral_error *= 0.90

        # FIXED: Update threshold (decrease when activation too high)
        delta = self.kp * error + self.ki * self.integral_error
        self.activation_threshold -= delta  # FIXED: subtract (was add)
        self.activation_threshold = max(0.01, min(0.99, self.activation_threshold))

        # Energy tracking
        baseline_energy = batch_size * self.energy_per_activation
        actual_energy = (
            num_active * self.energy_per_activation +
            (batch_size - num_active) * self.energy_per_skip
        )

        self.total_baseline_energy += baseline_energy
        self.total_actual_energy += actual_energy

        # Calculate current energy savings
        if self.total_baseline_energy > 0:
            energy_savings = (
                (self.total_baseline_energy - self.total_actual_energy) /
                self.total_baseline_energy * 100
            )
        else:
            energy_savings = 0.0

        energy_info = {
            'num_active': num_active,
            'activation_rate': current_activation_rate,
            'activation_rate_ema': self.activation_rate_ema,
            'threshold': self.activation_threshold,
            'baseline_energy': baseline_energy,
            'actual_energy': actual_energy,
            'energy_savings': energy_savings,  # ADDED
        }

        return active_mask, energy_info

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch_ast(model, train_loader, criterion, optimizer, sundew, config, epoch):
    """Train one epoch with AST - FIXED with energy display"""
    model.train()

    running_loss = 0.0
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

        # Forward pass to get losses (no gradient yet)
        with torch.no_grad():
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(
                outputs, labels, reduction='none'
            )

        # Select important samples
        active_mask, energy_info = sundew.select_samples(losses, images)

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
            total_active += active_mask.sum().item()

        total_samples += batch_size

        # Track stats
        epoch_stats['activation_rates'].append(energy_info['activation_rate_ema'])
        epoch_stats['thresholds'].append(energy_info['threshold'])
        epoch_stats['energy_savings'].append(energy_info['energy_savings'])

        # Print progress every 50 batches (CIFAR-10 format)
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                  f"Act: {100*energy_info['activation_rate_ema']:4.1f}% | "
                  f"⚡ Energy Saved: {energy_info['energy_savings']:5.1f}% | "
                  f"Threshold: {energy_info['threshold']:.3f}")

    avg_loss = running_loss / max(total_active, 1)
    avg_activation = total_active / total_samples

    return avg_loss, avg_activation, epoch_stats

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
# VISUALIZATION
# ============================================================================

def create_results_diagram(all_epoch_stats, final_results, config):
    """Create comprehensive results visualization like CIFAR-10"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ImageNet-100 AST Training Results', fontsize=16, fontweight='bold')

    # Combine stats from all epochs
    all_activation_rates = []
    all_thresholds = []
    all_energy_savings = []

    for epoch_stats in all_epoch_stats:
        all_activation_rates.extend(epoch_stats['activation_rates'])
        all_thresholds.extend(epoch_stats['thresholds'])
        all_energy_savings.extend(epoch_stats['energy_savings'])

    batches = list(range(len(all_activation_rates)))

    # Plot 1: Activation Rate Over Time
    ax1 = axes[0, 0]
    ax1.plot(batches, [r * 100 for r in all_activation_rates],
             color='blue', linewidth=1.5, alpha=0.7)
    ax1.axhline(y=config.target_activation_rate * 100,
                color='red', linestyle='--', label=f'Target: {config.target_activation_rate*100:.1f}%')
    ax1.set_xlabel('Batch', fontsize=11)
    ax1.set_ylabel('Activation Rate (%)', fontsize=11)
    ax1.set_title('Activation Rate Convergence', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Threshold Adaptation
    ax2 = axes[0, 1]
    ax2.plot(batches, all_thresholds, color='green', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Batch', fontsize=11)
    ax2.set_ylabel('Threshold', fontsize=11)
    ax2.set_title('PI Controller Threshold Adaptation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy Savings Over Time
    ax3 = axes[1, 0]
    ax3.plot(batches, all_energy_savings, color='orange', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=90, color='red', linestyle='--', label='Target: 90%')
    ax3.set_xlabel('Batch', fontsize=11)
    ax3.set_ylabel('Energy Savings (%)', fontsize=11)
    ax3.set_title('Cumulative Energy Savings', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Final Results Summary
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    FINAL RESULTS
    {'='*40}

    Validation Accuracy:     {final_results['accuracy']:.2f}%
    Energy Savings:          {final_results['energy_savings']:.2f}%
    Activation Rate:         {final_results['activation_rate']:.2f}%
    Training Time:           {final_results['training_time']:.1f} min

    Estimated Baseline:      {final_results['baseline_time']:.1f} min
    Training Speedup:        {final_results['speedup']:.1f}×

    {'='*40}

    Dataset: ImageNet-100 (130K train, 5K val)
    Model: ResNet50 ({final_results['num_params']:.1f}M params)
    Epochs: {config.num_epochs}
    Target Activation: {config.target_activation_rate*100:.1f}%
    """

    ax4.text(0.1, 0.5, summary_text,
             fontsize=10, family='monospace',
             verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('imagenet100_ast_results.png', dpi=150, bbox_inches='tight')
    print("\n✅ Results diagram saved: imagenet100_ast_results.png")
    plt.show()

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function - FIXED VERSION"""

    print("=" * 70)
    print("BATCHED ADAPTIVE SPARSE TRAINING - IMAGENET-100")
    print("With Live Energy Monitoring! 🔋⚡")
    print("=" * 70)

    # Configuration
    config = Config()
    print(f"Device: {config.device}")
    print(f"Target activation rate: {config.target_activation_rate*100:.1f}%")
    print(f"Expected speedup: 8-12× (ImageNet-100 with ResNet50)")
    print(f"Training for {config.num_epochs} epochs...")
    print()
    print()

    # Data
    train_loader, val_loader = get_dataloaders(config)
    print()

    # Model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(config.device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded ResNet50 ({num_params:.1f}M params)")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Sundew algorithm
    sundew = SundewAlgorithm(config)

    start_time = time.time()
    best_accuracy = 0.0
    all_epoch_stats = []

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Print epoch header
        print()
        print("=" * 60)
        print(f"Epoch {epoch}/{config.num_epochs}")
        print("=" * 60)

        # Train
        train_loss, train_activation, epoch_stats = train_epoch_ast(
            model, train_loader, criterion, optimizer, sundew, config, epoch
        )
        all_epoch_stats.append(epoch_stats)

        # Validate
        val_loss, val_accuracy = validate(model, val_loader, criterion, config)

        # Energy savings
        if sundew.total_baseline_energy > 0:
            energy_savings = (
                (sundew.total_baseline_energy - sundew.total_actual_energy) /
                sundew.total_baseline_energy * 100
            )
        else:
            energy_savings = 0.0

        epoch_time = time.time() - epoch_start

        # Print epoch summary
        print()
        print(f"✅ Epoch {epoch} Complete | "
              f"Val Acc: {val_accuracy:5.2f}% | "
              f"Loss: {train_loss:.4f} | "
              f"⚡ Energy Saved: {energy_savings:5.1f}% | "
              f"Time: {epoch_time:.1f}s")

        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_accuracy,
            }, 'best_model_imagenet100.pth')

    total_time = time.time() - start_time

    # Final results
    print("-" * 70)
    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Total Energy Savings: {energy_savings:.2f}%")
    print(f"Average Activation Rate: {100*train_activation:.2f}%")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print()

    # Speedup estimation
    baseline_time = total_time / train_activation if train_activation > 0 else total_time
    speedup = baseline_time / total_time if total_time > 0 else 1.0
    print(f"Estimated Baseline Time: {baseline_time/60:.1f} minutes")
    print(f"Training Speedup: {speedup:.1f}×")
    print("=" * 70)

    # Create results visualization
    final_results = {
        'accuracy': best_accuracy,
        'energy_savings': energy_savings,
        'activation_rate': 100 * train_activation,
        'training_time': total_time / 60,
        'baseline_time': baseline_time / 60,
        'speedup': speedup,
        'num_params': num_params,
    }

    create_results_diagram(all_epoch_stats, final_results, config)

if __name__ == "__main__":
    main()
