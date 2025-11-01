"""
Adaptive Sparse Training (AST) on ImageNet-100
===============================================
Adapts CIFAR-10 AST implementation for ImageNet-100 validation.

ImageNet-100: Subset of 100 classes from ImageNet (~130K images)
- Input: 224x224 RGB images (vs 32x32 CIFAR-10)
- Classes: 100 (vs 10 CIFAR-10)
- Model: ResNet50 (vs SimpleCNN)

Expected Results:
- Baseline ResNet50: ~75-80% top-1 accuracy
- AST ResNet50: 75-80% accuracy with ~90% energy savings

Setup Instructions for Kaggle:
1. Create new notebook
2. Add ImageNet-100 dataset: Search "imagenet100" or use:
   https://www.kaggle.com/datasets/ambityga/imagenet100
3. Enable GPU: Settings > Accelerator > GPU T4 x2
4. Copy this code and run
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

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experiment configuration"""
    # Dataset
    data_dir = "/kaggle/input/imagenet100"  # Adjust based on Kaggle dataset path
    num_classes = 100
    image_size = 224

    # Training
    batch_size = 64  # Reduced for 224x224 images (vs 128 for CIFAR-10)
    num_epochs = 40
    learning_rate = 0.001
    weight_decay = 1e-4

    # AST Configuration
    target_activation_rate = 0.10  # Train on 10% of samples
    initial_threshold = 0.50

    # PI Controller (same as CIFAR-10)
    adapt_kp = 0.0015
    adapt_ki = 0.00005
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
    """ImageNet-100 dataset loader

    Expected structure:
    imagenet100/
        train/
            class1/
                img1.JPEG
                img2.JPEG
            class2/
                ...
        val/
            class1/
                ...
    """
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

    # ImageNet normalization (standard)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Training augmentation (standard ImageNet)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        normalize,
    ])

    # Validation transform (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    # Create datasets
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

    # Create dataloaders
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
# SUNDEW ALGORITHM (Same as CIFAR-10)
# ============================================================================

class SundewAlgorithm:
    """Adaptive gating with PI control and EMA smoothing"""

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

    def compute_significance(self, losses, images):
        """Multi-factor significance scoring

        Args:
            losses: Per-sample losses [batch_size]
            images: Batch images [batch_size, 3, 224, 224]

        Returns:
            significance: [batch_size]
        """
        batch_size = losses.size(0)

        # Factor 1: Normalized loss
        loss_mean = losses.mean()
        loss_norm = losses / (loss_mean + 1e-8)

        # Factor 2: Image intensity variation (spatial std)
        # Compute std across spatial dimensions [batch_size, 3, 224, 224] -> [batch_size]
        images_flat = images.view(batch_size, 3, -1)  # [batch_size, 3, 224*224]
        std_per_channel = images_flat.std(dim=2)  # [batch_size, 3]
        std_intensity = std_per_channel.mean(dim=1)  # [batch_size]

        std_mean = std_intensity.mean()
        std_norm = std_intensity / (std_mean + 1e-8)

        # Weighted combination (70% loss, 30% intensity)
        significance = 0.7 * loss_norm + 0.3 * std_norm

        return significance

    def select_samples(self, losses, images):
        """Select important samples and update threshold

        Returns:
            active_mask: Boolean mask [batch_size]
            energy_info: Dict with energy statistics
        """
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

        # PI controller update
        error = self.activation_rate_ema - self.target_activation_rate

        # Integral with anti-windup
        if 0.01 < self.activation_threshold < 0.99:
            self.integral_error += error
            self.integral_error = max(-50, min(50, self.integral_error))
        else:
            self.integral_error *= 0.90  # Decay when saturated

        # Update threshold
        self.activation_threshold += self.kp * error + self.ki * self.integral_error
        self.activation_threshold = max(0.01, min(0.99, self.activation_threshold))

        # Energy tracking
        baseline_energy = batch_size * self.energy_per_activation
        actual_energy = (
            num_active * self.energy_per_activation +
            (batch_size - num_active) * self.energy_per_skip
        )

        self.total_baseline_energy += baseline_energy
        self.total_actual_energy += actual_energy

        energy_info = {
            'num_active': num_active,
            'activation_rate': current_activation_rate,
            'activation_rate_ema': self.activation_rate_ema,
            'threshold': self.activation_threshold,
            'baseline_energy': baseline_energy,
            'actual_energy': actual_energy,
        }

        return active_mask, energy_info

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch_ast(model, train_loader, criterion, optimizer, sundew, config, epoch):
    """Train one epoch with AST"""
    model.train()

    running_loss = 0.0
    total_samples = 0
    total_active = 0

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

        # Print progress every 50 batches
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {running_loss/max(total_active,1):.4f} | "
                  f"Act: {100*energy_info['activation_rate_ema']:.1f}% | "
                  f"Thr: {energy_info['threshold']:.3f}")

    avg_loss = running_loss / max(total_active, 1)
    avg_activation = total_active / total_samples

    return avg_loss, avg_activation

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
# MAIN TRAINING LOOP
# ============================================================================

def main():
    """Main training function"""

    print("=" * 70)
    print("Adaptive Sparse Training (AST) - ImageNet-100 Validation")
    print("=" * 70)
    print()

    # Configuration
    config = Config()
    print(f"Device: {config.device}")
    print(f"Batch size: {config.batch_size}")
    print(f"Target activation rate: {config.target_activation_rate*100:.1f}%")
    print()

    # Data
    print("Loading ImageNet-100 dataset...")
    train_loader, val_loader = get_dataloaders(config)
    print()

    # Model: ResNet50 pretrained on ImageNet (fine-tune for ImageNet-100)
    print("Initializing ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Replace final layer for 100 classes
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(config.device)
    print(f"Model: ResNet50 ({sum(p.numel() for p in model.parameters())/1e6:.1f}M params)")
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

    # Training loop
    print("Starting training...")
    print("-" * 70)

    start_time = time.time()
    best_accuracy = 0.0

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_activation = train_epoch_ast(
            model, train_loader, criterion, optimizer, sundew, config, epoch
        )

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
        print(f"Epoch {epoch:2d}/{config.num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Val Acc: {val_accuracy:5.2f}% | "
              f"Act: {100*train_activation:4.1f}% | "
              f"Save: {energy_savings:4.1f}% | "
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

    # Comparison to baseline (estimated)
    baseline_time = total_time / train_activation  # Estimated if training on 100%
    speedup = baseline_time / total_time
    print(f"Estimated Baseline Time: {baseline_time/60:.1f} minutes")
    print(f"Training Speedup: {speedup:.1f}×")
    print("=" * 70)

if __name__ == "__main__":
    main()
