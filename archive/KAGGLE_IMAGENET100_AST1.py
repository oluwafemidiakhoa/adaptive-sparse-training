"""
PRODUCTION - IMAGENET-100 AST (80 EPOCHS)
====================================================================
PURPOSE: Full production training with ALL optimizations

Key Optimizations:
1. 80 epochs training for full convergence
2. Fixed significance scoring (loss magnitude + entropy)
3. Cosine annealing LR with 5-epoch warmup
4. Advanced augmentation (RandAugment, strong augmentation)
5. Batch size 128 (up from 64) for better GPU utilization
6. Target activation 15% (up from 10%) for better accuracy/efficiency balance
7. Tuned PI controller (Kp=0.006, Ki=0.0001) for stability
8. Label smoothing (0.1) for better generalization
9. Gradient clipping (max_norm=1.0) for stability
10. All monitoring dashboards and architecture diagrams

Expected Results:
- Validation Accuracy: 50-55%
- Energy Savings: 85-87%
- Activation Rate: ~15%
- Training Time: ~10-12 hours on Kaggle GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
from PIL import Image
import time
import numpy as np
import math

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Production configuration with all optimizations"""
    # Dataset - UPDATE THIS PATH
    data_dir = "/kaggle/input/imagenet100/ImageNet100"
    num_classes = 100
    image_size = 224

    # Training - PRODUCTION SETTINGS
    batch_size = 128  # Increased from 64
    num_epochs = 80  # Full training
    learning_rate = 0.001  # Initial LR (will use cosine annealing)
    weight_decay = 1e-4
    warmup_epochs = 5  # LR warmup

    # Label smoothing for better generalization
    label_smoothing = 0.1

    # Gradient clipping for stability
    max_grad_norm = 1.0

    # AST Configuration - OPTIMIZED
    target_activation_rate = 0.15  # Increased from 0.10 for better accuracy
    initial_threshold = 2.0  # Lower initial threshold for 15% target

    # PI Controller - TUNED FOR STABILITY
    adapt_kp = 0.006  # Reduced from 0.02 for gentler adjustments
    adapt_ki = 0.0001  # Reduced from 0.001 for stability
    ema_alpha = 0.2

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2
    pin_memory = True

    # Checkpointing
    save_checkpoints = True
    checkpoint_dir = "checkpoints"
    save_every = 10  # Save every 10 epochs

# ============================================================================
# DATASET WITH ADVANCED AUGMENTATION
# ============================================================================

class RandAugment:
    """Simple RandAugment implementation"""
    def __init__(self, n=2, m=10):
        self.n = n  # Number of augmentation transformations to apply
        self.m = m  # Magnitude for all transformations (0-10)

    def __call__(self, img):
        """Apply n random augmentations with magnitude m"""
        import random
        from PIL import ImageEnhance, ImageOps

        ops = [
            lambda img: ImageEnhance.Color(img).enhance(1 + self.m/10 * random.choice([-1, 1])),
            lambda img: ImageEnhance.Contrast(img).enhance(1 + self.m/10 * random.choice([-1, 1])),
            lambda img: ImageEnhance.Brightness(img).enhance(1 + self.m/10 * random.choice([-1, 1])),
            lambda img: ImageEnhance.Sharpness(img).enhance(1 + self.m/10 * random.choice([-1, 1])),
            lambda img: ImageOps.autocontrast(img),
            lambda img: ImageOps.equalize(img),
        ]

        for _ in range(self.n):
            op = random.choice(ops)
            try:
                img = op(img)
            except:
                pass

        return img

class ImageNet100Dataset(Dataset):
    """ImageNet-100 dataset loader"""
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
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

def get_dataloaders(config):
    """Create ImageNet-100 dataloaders with ADVANCED AUGMENTATION"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # ADVANCED TRAINING AUGMENTATION
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        RandAugment(n=2, m=10),  # RandAugment
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),  # Random erasing
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageNet100Dataset(config.data_dir, 'train', train_transform)
    val_dataset = ImageNet100Dataset(config.data_dir, 'val', val_transform)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=config.pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers,
                           pin_memory=config.pin_memory)

    return train_loader, val_loader

# ============================================================================
# SUNDEW ALGORITHM - FIXED SIGNIFICANCE SCORING
# ============================================================================

class SundewAlgorithm:
    """Adaptive gating with FIXED gradient-magnitude based significance"""

    def __init__(self, config):
        self.target_activation_rate = config.target_activation_rate
        self.activation_threshold = config.initial_threshold

        # PI controller - TUNED FOR STABILITY
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
        FIXED SIGNIFICANCE SCORING

        Uses:
        - Absolute loss magnitude (harder samples = higher loss)
        - Prediction entropy (uncertain samples = high entropy)
        - No batch normalization that destroys variance
        """
        batch_size = losses.size(0)

        # Factor 1: Loss magnitude (scale to [0,1] using max, NOT mean)
        loss_max = losses.max()
        if loss_max > 0:
            loss_significance = losses / loss_max
        else:
            loss_significance = torch.ones_like(losses)

        # Factor 2: Prediction uncertainty (entropy)
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        entropy_max = entropy.max()
        if entropy_max > 0:
            entropy_significance = entropy / entropy_max
        else:
            entropy_significance = torch.zeros_like(entropy)

        # Weighted combination: 70% loss, 30% uncertainty
        significance = 0.7 * loss_significance + 0.3 * entropy_significance

        return significance

    def select_samples(self, losses, images, outputs):
        """Select important samples using fixed significance scoring"""
        batch_size = losses.size(0)

        # Compute significance scores
        significance = self.compute_significance(losses, outputs)

        # Select samples above threshold
        active_mask = significance > self.activation_threshold
        num_active = active_mask.sum().item()

        # Fallback mechanism
        if num_active == 0:
            random_indices = torch.randperm(batch_size)[:max(2, int(batch_size * 0.05))]
            active_mask[random_indices] = True
            num_active = len(random_indices)

        # Update activation rate EMA
        current_activation_rate = num_active / batch_size
        self.activation_rate_ema = (
            self.ema_alpha * current_activation_rate +
            (1 - self.ema_alpha) * self.activation_rate_ema
        )

        # PI controller with tuned gains
        error = self.activation_rate_ema - self.target_activation_rate
        proportional = self.kp * error

        # Integral with anti-windup
        if 0.01 < self.activation_threshold < 10.0:
            self.integral_error += error
            self.integral_error = max(-50, min(50, self.integral_error))
        else:
            self.integral_error *= 0.90

        new_threshold = self.activation_threshold + proportional + self.ki * self.integral_error
        self.activation_threshold = max(0.01, new_threshold)

        # Energy tracking
        baseline_energy = batch_size * self.energy_per_activation
        actual_energy = (num_active * self.energy_per_activation +
                        (batch_size - num_active) * self.energy_per_skip)

        self.total_baseline_energy += baseline_energy
        self.total_actual_energy += actual_energy

        # Calculate energy savings
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
# LEARNING RATE SCHEDULER
# ============================================================================

class CosineAnnealingWarmup:
    """Cosine annealing with warmup"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self, epoch):
        """Update learning rate"""
        self.current_epoch = epoch

        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

# ============================================================================
# TRAINING
# ============================================================================

def train_epoch_ast(model, train_loader, criterion, optimizer, sundew, config, epoch):
    """Train one epoch with AST and gradient clipping"""
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

        # Forward pass to get losses and outputs
        with torch.no_grad():
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')

        # Select important samples
        active_mask, energy_info = sundew.select_samples(losses, images, outputs)

        # Train only on active samples
        if active_mask.sum() > 0:
            active_images = images[active_mask]
            active_labels = labels[active_mask]

            optimizer.zero_grad()
            active_outputs = model(active_images)
            loss = criterion(active_outputs, active_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            running_loss += loss.item() * active_mask.sum().item()
            total_active += active_mask.sum().item()

        total_samples += batch_size

        # Track stats
        epoch_stats['activation_rates'].append(energy_info['activation_rate_ema'])
        epoch_stats['thresholds'].append(energy_info['threshold'])
        epoch_stats['energy_savings'].append(energy_info['energy_savings'])

        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                  f"Act: {100*energy_info['activation_rate_ema']:4.1f}% | "
                  f"Energy Saved: {energy_info['energy_savings']:5.1f}% | "
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
# ARCHITECTURE DIAGRAM
# ============================================================================

def create_architecture_diagram():
    """Create production AST architecture diagram"""
    fig, ax = plt.subplots(figsize=(18, 12))
    ax.axis('off')

    # Title
    fig.suptitle('PRODUCTION - ImageNet-100 AST with All Optimizations',
                 fontsize=22, fontweight='bold', y=0.98)

    # Define positions
    y_start = 0.85
    box_height = 0.08
    box_width = 0.18

    # Row 1: Input Stage
    ax.add_patch(plt.Rectangle((0.05, y_start), box_width, box_height,
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(0.14, y_start + box_height/2, 'Input Batch\n[128, 3, 224, 224]',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow
    ax.arrow(0.23, y_start + box_height/2, 0.05, 0, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Fixed Significance
    ax.add_patch(plt.Rectangle((0.30, y_start), box_width, box_height,
                                facecolor='lime', edgecolor='green', linewidth=3))
    ax.text(0.39, y_start + box_height/2, 'FIXED\nSignificance\n(Loss + Entropy)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')

    # Arrow
    ax.arrow(0.48, y_start + box_height/2, 0.05, 0, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Sundew Gating
    ax.add_patch(plt.Rectangle((0.55, y_start), box_width, box_height,
                                facecolor='yellow', edgecolor='black', linewidth=2))
    ax.text(0.64, y_start + box_height/2, 'Sundew Gating\nPI Controller',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow
    ax.arrow(0.73, y_start + box_height/2, 0.05, 0, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Active Mask
    ax.add_patch(plt.Rectangle((0.80, y_start), box_width, box_height,
                                facecolor='orange', edgecolor='black', linewidth=2))
    ax.text(0.89, y_start + box_height/2, 'Active Mask\n[128] → [~19]',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Row 2: Training Stage
    y_row2 = y_start - 0.15

    ax.add_patch(plt.Rectangle((0.30, y_row2), 0.43, box_height,
                                facecolor='lightcoral', edgecolor='black', linewidth=3))
    ax.text(0.515, y_row2 + box_height/2,
            'ResNet50 Training (GPU Parallel)\nLabel Smoothing + Gradient Clipping',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow down
    ax.arrow(0.89, y_start, 0, -0.06, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Row 3: Optimizations
    y_row3 = y_row2 - 0.22

    # Box 1: Data Augmentation
    ax.add_patch(plt.Rectangle((0.05, y_row3), 0.27, 0.14,
                                facecolor='#E6F3FF', edgecolor='blue', linewidth=2))
    ax.text(0.185, y_row3 + 0.11, 'Advanced Augmentation',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkblue')
    ax.text(0.185, y_row3 + 0.06, 'RandAugment\nColorJitter, RandomErasing',
            ha='center', va='center', fontsize=9)
    ax.text(0.185, y_row3 + 0.02, 'Batch Size: 128',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Box 2: Training Optimizations
    ax.add_patch(plt.Rectangle((0.35, y_row3), 0.27, 0.14,
                                facecolor='#FFE6F0', edgecolor='red', linewidth=2))
    ax.text(0.485, y_row3 + 0.11, 'Training Optimizations',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkred')
    ax.text(0.485, y_row3 + 0.06, 'Cosine LR + 5-epoch warmup\nLabel smoothing (0.1)',
            ha='center', va='center', fontsize=9)
    ax.text(0.485, y_row3 + 0.02, 'Gradient clipping (1.0)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Box 3: AST Optimizations
    ax.add_patch(plt.Rectangle((0.65, y_row3), 0.27, 0.14,
                                facecolor='#E6FFE6', edgecolor='green', linewidth=2))
    ax.text(0.785, y_row3 + 0.11, 'AST Optimizations',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')
    ax.text(0.785, y_row3 + 0.06, 'Target Activation: 15%\nPI Gains: Kp=0.006, Ki=0.0001',
            ha='center', va='center', fontsize=9)
    ax.text(0.785, y_row3 + 0.02, 'Fixed Significance Scoring',
            ha='center', va='center', fontsize=9, fontweight='bold')

    # Bottom: Expected Results
    y_bottom = 0.08

    ax.add_patch(plt.Rectangle((0.15, y_bottom), 0.7, 0.12,
                                facecolor='gold', edgecolor='orange', linewidth=3))
    ax.text(0.5, y_bottom + 0.08, 'EXPECTED RESULTS (80 Epochs)',
            ha='center', va='center', fontsize=13, fontweight='bold')
    ax.text(0.5, y_bottom + 0.04, 'Accuracy: 50-55% | Energy Savings: 85-87% | Activation: ~15% | Time: 10-12 hours',
            ha='center', va='center', fontsize=11)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.savefig('production_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Architecture diagram saved to: production_architecture.png")
    plt.show()

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def create_results_dashboard(all_epoch_stats, final_results, config, training_history):
    """Create comprehensive production results dashboard"""

    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Combine all epoch stats
    all_activation_rates = []
    all_thresholds = []
    all_energy_savings = []

    for epoch_stats in all_epoch_stats:
        all_activation_rates.extend(epoch_stats['activation_rates'])
        all_thresholds.extend(epoch_stats['thresholds'])
        all_energy_savings.extend(epoch_stats['energy_savings'])

    batches = list(range(len(all_activation_rates)))

    # Plot 1: Activation Rate over Time
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(batches, [r * 100 for r in all_activation_rates],
             color='blue', linewidth=1.5, alpha=0.7, label='Activation Rate')
    ax1.axhline(y=config.target_activation_rate * 100, color='red',
                linestyle='--', linewidth=2, label=f'Target: {config.target_activation_rate*100:.0f}%')
    ax1.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Activation Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Activation Rate Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Threshold Adaptation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(batches, all_thresholds, color='green', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threshold', fontsize=11, fontweight='bold')
    ax2.set_title('PI Controller Threshold', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy Savings
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(batches, all_energy_savings, color='orange', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target: 85%')
    ax3.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Energy Savings (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Cumulative Energy Savings', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Validation Accuracy over Epochs
    ax4 = fig.add_subplot(gs[1, 0])
    epochs = list(range(1, len(training_history['val_acc']) + 1))
    ax4.plot(epochs, training_history['val_acc'], 'o-', color='purple',
             linewidth=2, markersize=6, label='Val Accuracy')
    ax4.axhline(y=50, color='red', linestyle='--', linewidth=2, label='Target: 50%')
    ax4.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Validation Accuracy Progress', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Learning Rate Schedule
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(epochs, training_history['lr'], 'o-', color='teal',
             linewidth=2, markersize=6, label='Learning Rate')
    ax5.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
    ax5.set_title('Cosine Annealing LR Schedule', fontsize=13, fontweight='bold')
    ax5.set_yscale('log')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Epoch Energy Savings
    ax6 = fig.add_subplot(gs[1, 2])
    if len(all_epoch_stats) > 0:
        batch_size = len(all_epoch_stats[0]['energy_savings'])
        epoch_energy_savings = []
        for i in range(len(all_epoch_stats)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(all_energy_savings))
            avg = np.mean(all_energy_savings[start_idx:end_idx])
            epoch_energy_savings.append(avg)

        ax6.plot(epochs, epoch_energy_savings, 'o-', color='darkorange',
                linewidth=2, markersize=6, label='Energy Saved')
        ax6.axhline(y=85, color='red', linestyle='--', linewidth=2, label='Target: 85%')
        ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Energy Savings (%)', fontsize=11, fontweight='bold')
        ax6.set_title('Energy Savings per Epoch', fontsize=13, fontweight='bold')
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3)

    # Plot 7: Activation Distribution
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist([r * 100 for r in all_activation_rates], bins=50,
             color='skyblue', edgecolor='black', alpha=0.7)
    ax7.axvline(x=config.target_activation_rate * 100, color='red',
                linestyle='--', linewidth=2, label='Target')
    ax7.set_xlabel('Activation Rate (%)', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('Activation Rate Distribution', fontsize=13, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')

    # Plot 8: Energy Distribution
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(all_energy_savings, bins=50, color='lightcoral',
             edgecolor='black', alpha=0.7)
    ax8.axvline(x=85, color='red', linestyle='--', linewidth=2, label='Target')
    ax8.set_xlabel('Energy Savings (%)', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('Energy Savings Distribution', fontsize=13, fontweight='bold')
    ax8.legend(fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')

    # Plot 9: Training Loss
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.plot(epochs, training_history['train_loss'], 'o-', color='darkblue',
             linewidth=2, markersize=6, label='Train Loss')
    ax9.plot(epochs, training_history['val_loss'], 'o-', color='darkred',
             linewidth=2, markersize=6, label='Val Loss')
    ax9.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax9.set_title('Training and Validation Loss', fontsize=13, fontweight='bold')
    ax9.legend(fontsize=9)
    ax9.grid(True, alpha=0.3)

    # Plot 10: Summary
    ax10 = fig.add_subplot(gs[3, :])
    ax10.axis('off')

    summary_text = f"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════╗
║                       PRODUCTION RESULTS (80 EPOCHS)                                             ║
╠══════════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                                  ║
║  Final Validation Accuracy:      {final_results['accuracy']:6.2f}%     {'✅ EXCELLENT!' if final_results['accuracy'] >= 50 else '⚠️  Check training'}                                    ║
║  Energy Savings:                 {final_results['energy_savings']:6.2f}%     {'✅ ON TARGET' if final_results['energy_savings'] >= 85 else '⚠️  LOW'}                                       ║
║  Activation Rate:                {final_results['activation_rate']:6.2f}%     {'✅ PERFECT' if 14 <= final_results['activation_rate'] <= 16 else '⚠️  OFF TARGET'}                                       ║
║  Training Time:                  {final_results['training_time']:6.1f} hrs                                                   ║
║                                                                                                  ║
║  OPTIMIZATIONS APPLIED:                                                                          ║
║     ✅ Fixed Significance Scoring (Loss magnitude + Entropy)                                     ║
║     ✅ Advanced Augmentation (RandAugment, ColorJitter, RandomErasing)                           ║
║     ✅ Cosine Annealing LR with 5-epoch warmup                                                   ║
║     ✅ Label Smoothing (0.1) for better generalization                                           ║
║     ✅ Gradient Clipping (1.0) for training stability                                            ║
║     ✅ Batch Size 128 for better GPU utilization                                                 ║
║     ✅ Target Activation 15% for accuracy/efficiency balance                                     ║
║     ✅ Tuned PI Controller (Kp=0.006, Ki=0.0001)                                                 ║
║                                                                                                  ║
║  Dataset:  ImageNet-100 | Model: ResNet50 ({final_results['num_params']:.1f}M) | Epochs: {config.num_epochs}                            ║
║                                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax10.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                      edgecolor='orange', linewidth=3))

    plt.suptitle('PRODUCTION - ImageNet-100 AST Complete Results',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('production_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("\nResults dashboard saved: production_results.png")
    plt.show()

# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, scheduler, sundew, epoch, accuracy, config, is_best=False):
    """Save training checkpoint"""
    if not config.save_checkpoints:
        return

    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': accuracy,
        'threshold': sundew.activation_threshold,
        'total_baseline_energy': sundew.total_baseline_energy,
        'total_actual_energy': sundew.total_actual_energy,
    }

    # Save regular checkpoint
    checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # Save best checkpoint
    if is_best:
        best_path = checkpoint_dir / 'best_model.pt'
        torch.save(checkpoint, best_path)
        print(f"Best model saved: {best_path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function"""

    print("=" * 80)
    print("PRODUCTION - IMAGENET-100 AST (80 EPOCHS)")
    print("With ALL optimizations for maximum accuracy and energy efficiency")
    print("=" * 80)

    config = Config()
    print(f"Device: {config.device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Target Activation Rate: {config.target_activation_rate*100:.1f}%")
    print(f"Learning Rate: {config.learning_rate} (with cosine annealing + warmup)")
    print(f"Label Smoothing: {config.label_smoothing}")
    print(f"Gradient Clipping: {config.max_grad_norm}")
    print()

    # Create architecture diagram
    print("Generating architecture diagram...")
    create_architecture_diagram()
    print()

    # Load data
    print("Loading ImageNet-100 dataset with advanced augmentation...")
    train_loader, val_loader = get_dataloaders(config)
    print()

    # Load model
    print("Loading ResNet50 model...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(config.device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"ResNet50 loaded: {num_params:.1f}M parameters")
    print()

    # Setup
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                           weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmup(optimizer, config.warmup_epochs, config.num_epochs,
                                      config.learning_rate)
    sundew = SundewAlgorithm(config)

    # Training loop
    start_time = time.time()
    best_accuracy = 0.0
    all_epoch_stats = []
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'lr': [],
    }

    print("Starting training...")
    print("=" * 80)

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        print()
        print("=" * 70)
        print(f"Epoch {epoch}/{config.num_epochs}")
        print("=" * 70)

        # Update learning rate
        current_lr = scheduler.step(epoch - 1)
        print(f"Learning Rate: {current_lr:.6f}")

        # Train
        train_loss, train_activation, epoch_stats = train_epoch_ast(
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

        # Save history
        training_history['train_loss'].append(train_loss)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_accuracy)
        training_history['lr'].append(current_lr)

        # Print summary
        print()
        print(f"Epoch {epoch} Complete | "
              f"Val Acc: {val_accuracy:5.2f}% | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Energy Saved: {energy_savings:5.1f}% | "
              f"Time: {epoch_time/60:.1f}min")

        # Save checkpoint
        is_best = val_accuracy > best_accuracy
        if is_best:
            best_accuracy = val_accuracy

        if epoch % config.save_every == 0 or is_best:
            save_checkpoint(model, optimizer, scheduler, sundew, epoch, val_accuracy, config, is_best)

    total_time = time.time() - start_time

    # Final results
    print()
    print("=" * 80)
    print("PRODUCTION TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Total Energy Savings: {energy_savings:.2f}%")
    print(f"Average Activation Rate: {100*train_activation:.2f}%")
    print(f"Total Training Time: {total_time/3600:.1f} hours")
    print()

    # Performance analysis
    print("PERFORMANCE ANALYSIS:")
    print(f"   Target Accuracy: 50-55%")
    print(f"   Achieved:        {best_accuracy:.2f}%")
    print(f"   Target Energy:   85-87%")
    print(f"   Achieved:        {energy_savings:.2f}%")
    print(f"   Target Activation: ~15%")
    print(f"   Achieved:        {100*train_activation:.2f}%")
    print()

    if best_accuracy >= 50 and energy_savings >= 85:
        print("SUCCESS! All targets achieved!")
    elif best_accuracy >= 45:
        print("GOOD RESULTS - Close to target")
    else:
        print("Results below target - may need more tuning")

    print("=" * 80)

    # Create dashboard
    final_results = {
        'accuracy': best_accuracy,
        'energy_savings': energy_savings,
        'activation_rate': 100 * train_activation,
        'training_time': total_time / 3600,
        'num_params': num_params,
    }

    print("\nGenerating comprehensive results dashboard...")
    create_results_dashboard(all_epoch_stats, final_results, config, training_history)

    print("\nAll results saved!")
    print("  - production_architecture.png")
    print("  - production_results.png")
    if config.save_checkpoints:
        print(f"  - {config.checkpoint_dir}/best_model.pt")

if __name__ == "__main__":
    main()
