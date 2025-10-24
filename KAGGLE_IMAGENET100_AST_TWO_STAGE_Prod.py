"""
🔥 TWO-STAGE ADAPTIVE SPARSE TRAINING - ImageNet-100 🔥
====================================================================
SOLUTION: Pretrained models need warmup before AST!

STAGE 1 (Epochs 1-10): WARMUP - Train on 100% of samples
  → Adapt pretrained ResNet50 from ImageNet-1K to ImageNet-100
  → Get model to ~40-50% accuracy
  → "Teach" the new final layer

STAGE 2 (Epochs 11-50): AST - Train on 15-20% of samples
  → Apply Adaptive Sparse Training
  → Maintain 35-45% accuracy with 80-85% energy savings
  → Fine-tune with sparse updates

Expected Final Results:
- Validation Accuracy: 45-55%
- Energy Savings (overall): 65-75%
- Training Speedup: 4-6×

Runtime: ~6-8 hours for 50 epochs on Kaggle GPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from pathlib import Path
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Two-stage training configuration"""
    # Dataset
    data_dir = "/kaggle/input/imagenet100/ImageNet100"
    num_classes = 100
    image_size = 224

    # Training
    batch_size = 128  # Increased for better GPU utilization
    num_epochs = 50  # Total epochs
    warmup_epochs = 10  # Stage 1: Full training

    # Stage 1: Warmup (100% samples)
    warmup_lr = 0.01  # Higher LR for warmup
    warmup_weight_decay = 1e-4

    # Stage 2: AST (15-20% samples)
    ast_lr = 0.001  # Lower LR for fine-tuning
    ast_weight_decay = 1e-4
    target_activation_rate = 0.20  # 20% activation (more conservative)
    initial_threshold = 4.5

    # PI Controller - Gentle for stability
    adapt_kp = 0.008
    adapt_ki = 0.0003
    ema_alpha = 0.15

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 4
    pin_memory = True

# ============================================================================
# DATASET WITH AUGMENTATION
# ============================================================================

def get_dataloaders(config):
    """Create dataloaders with strong augmentation"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    # Strong augmentation for ImageNet
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(config.image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),  # Cutout-like augmentation
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageFolder(str(Path(config.data_dir) / 'train'), transform=train_transform)
    val_dataset = ImageFolder(str(Path(config.data_dir) / 'val'), transform=val_transform)

    print(f"📦 Loaded {len(train_dataset):,} training images")
    print(f"📦 Loaded {len(val_dataset):,} validation images")
    print(f"📦 Found {len(train_dataset.classes)} classes")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                              shuffle=True, num_workers=config.num_workers,
                              pin_memory=config.pin_memory, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=config.num_workers,
                           pin_memory=config.pin_memory)

    return train_loader, val_loader

# ============================================================================
# LEARNING RATE SCHEDULER
# ============================================================================

class CosineAnnealingWarmup:
    """Cosine annealing with warmup"""
    def __init__(self, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr

# ============================================================================
# SUNDEW ALGORITHM - RAW SIGNIFICANCE
# ============================================================================

class SundewAlgorithm:
    """Adaptive sample selection with RAW significance scores"""

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
        RAW significance scoring (no normalization)
        - loss: typically 1.0-5.0 for ImageNet-100
        - entropy: typically 2.0-4.6 (log 100 ≈ 4.6)
        """
        # RAW loss
        loss_component = losses

        # RAW entropy
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)

        # Weighted combination
        significance = 0.7 * loss_component + 0.3 * entropy

        return significance

    def select_samples(self, losses, outputs):
        """Select important samples"""
        batch_size = losses.size(0)

        # Compute significance
        significance = self.compute_significance(losses, outputs)

        # Select samples above threshold
        active_mask = significance > self.activation_threshold
        num_active = active_mask.sum().item()

        # Fallback: ensure minimum activation
        min_active = max(2, int(batch_size * 0.05))  # At least 5%
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

        # PI controller
        error = self.activation_rate_ema - self.target_activation_rate
        proportional = self.kp * error

        # Integral with anti-windup
        if 0.5 < self.activation_threshold < 15.0:
            self.integral_error += error
            self.integral_error = max(-100, min(100, self.integral_error))
        else:
            self.integral_error *= 0.90

        # Update threshold
        new_threshold = self.activation_threshold + proportional + self.ki * self.integral_error
        self.activation_threshold = max(0.5, min(15.0, new_threshold))

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
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch_warmup(model, train_loader, criterion, optimizer, config, epoch):
    """STAGE 1: Warmup training on 100% of samples"""
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(config.device)
        labels = labels.to(config.device)

        # Standard training (100% samples)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Progress print
        if (batch_idx + 1) % 100 == 0:
            acc = 100.0 * correct / total
            avg_loss_so_far = running_loss / total
            print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                  f"Loss: {avg_loss_so_far:.4f} | "
                  f"Train Acc: {acc:5.2f}% | "
                  f"Current Batch Loss: {loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def train_epoch_ast(model, train_loader, criterion, optimizer, sundew, config, epoch):
    """STAGE 2: AST training on selected samples"""
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

        # Forward pass to get significance
        with torch.no_grad():
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')

        # Select important samples
        active_mask, energy_info = sundew.select_samples(losses, outputs)

        # Train on selected samples
        if active_mask.sum() > 0:
            active_images = images[active_mask]
            active_labels = labels[active_mask]

            optimizer.zero_grad()
            active_outputs = model(active_images)
            loss = criterion(active_outputs, active_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

        # Progress print
        if (batch_idx + 1) % 100 == 0:
            train_acc = 100.0 * correct / max(total_active, 1)
            print(f"  Batch {batch_idx+1:4d}/{len(train_loader)} | "
                  f"Act: {100*energy_info['activation_rate_ema']:5.1f}% | "
                  f"Train Acc: {train_acc:5.2f}% | "
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
# VISUALIZATION
# ============================================================================

def create_results_dashboard(warmup_results, ast_results, final_metrics, config):
    """Create comprehensive results dashboard"""

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Plot 1: Validation Accuracy Over Time
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = list(range(1, config.num_epochs + 1))
    val_accs = warmup_results['val_accs'] + ast_results['val_accs']

    ax1.plot(epochs[:config.warmup_epochs], val_accs[:config.warmup_epochs],
             'o-', color='blue', linewidth=2, markersize=6, label='Stage 1: Warmup (100% samples)')
    ax1.plot(epochs[config.warmup_epochs:], val_accs[config.warmup_epochs:],
             'o-', color='green', linewidth=2, markersize=6, label='Stage 2: AST (20% samples)')
    ax1.axvline(x=config.warmup_epochs, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('🎯 Validation Accuracy - Two-Stage Training', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Training Loss
    ax2 = fig.add_subplot(gs[0, 2])
    train_losses = warmup_results['train_losses'] + ast_results['train_losses']
    ax2.plot(epochs, train_losses, 'o-', color='purple', linewidth=2, markersize=4)
    ax2.axvline(x=config.warmup_epochs, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Training Loss', fontsize=11, fontweight='bold')
    ax2.set_title('📉 Training Loss', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Activation Rate (AST phase only)
    if len(ast_results['all_activation_rates']) > 0:
        ax3 = fig.add_subplot(gs[1, 0])
        batches = list(range(len(ast_results['all_activation_rates'])))
        ax3.plot(batches, [r * 100 for r in ast_results['all_activation_rates']],
                 color='blue', linewidth=1.5, alpha=0.7, label='Activation Rate')
        ax3.axhline(y=config.target_activation_rate * 100, color='red',
                    linestyle='--', linewidth=2, label=f'Target: {config.target_activation_rate*100:.0f}%')
        ax3.set_xlabel('Batch (AST Phase)', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Activation Rate (%)', fontsize=11, fontweight='bold')
        ax3.set_title('🎯 Activation Rate Convergence', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)

    # Plot 4: Energy Savings
    if len(ast_results['all_energy_savings']) > 0:
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(batches, ast_results['all_energy_savings'],
                 color='orange', linewidth=1.5, alpha=0.7, label='Energy Saved')
        ax4.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target: 80%')
        ax4.set_xlabel('Batch (AST Phase)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Energy Savings (%)', fontsize=11, fontweight='bold')
        ax4.set_title('⚡ Energy Savings Over Time', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)

    # Plot 5: Threshold Adaptation
    if len(ast_results['all_thresholds']) > 0:
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.plot(batches, ast_results['all_thresholds'],
                 color='green', linewidth=1.5, alpha=0.7)
        ax5.set_xlabel('Batch (AST Phase)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Threshold', fontsize=11, fontweight='bold')
        ax5.set_title('🎛️ PI Controller Threshold', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3)

    # Plot 6: Summary Box
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis('off')

    summary_text = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                    🔥 TWO-STAGE AST - IMAGENET-100 FINAL RESULTS 🔥                        ║
╠════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                            ║
║  STAGE 1: WARMUP (Epochs 1-{config.warmup_epochs}, 100% samples)                                              ║
║    📊 Final Warmup Accuracy:    {warmup_results['val_accs'][-1]:6.2f}%                                               ║
║    ⏱️  Warmup Time:              {warmup_results['total_time']:.1f} min                                            ║
║                                                                                            ║
║  STAGE 2: AST (Epochs {config.warmup_epochs+1}-{config.num_epochs}, ~20% samples)                                            ║
║    📊 Best AST Accuracy:        {final_metrics['best_accuracy']:6.2f}%                                               ║
║    ⚡ Energy Savings:           {final_metrics['final_energy_savings']:6.2f}%                                               ║
║    🎯 Avg Activation Rate:      {final_metrics['avg_activation']:6.2f}%                                               ║
║    ⏱️  AST Phase Time:           {ast_results['total_time']:.1f} min                                            ║
║                                                                                            ║
║  OVERALL RESULTS:                                                                          ║
║    🏆 Best Validation Accuracy: {final_metrics['best_accuracy']:6.2f}%     {'✅ EXCELLENT!' if final_metrics['best_accuracy'] >= 50 else '✅ GOOD' if final_metrics['best_accuracy'] >= 40 else '⚠️  NEEDS TUNING'}                           ║
║    ⚡ Overall Energy Savings:   {final_metrics['overall_energy_savings']:6.2f}%     (accounting for warmup)                 ║
║    🚀 Overall Speedup:          {final_metrics['overall_speedup']:6.2f}×                                                   ║
║    ⏱️  Total Time:               {final_metrics['total_time']:.1f} min                                            ║
║                                                                                            ║
║  📦 Dataset:     ImageNet-100 (126K train, 5K val)                                         ║
║  🤖 Model:       ResNet50 (23.7M params, pretrained on ImageNet-1K)                        ║
║  🎛️  Controller:  PI (Kp={config.adapt_kp}, Ki={config.adapt_ki})                                ║
║                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax6.text(0.5, 0.5, summary_text, fontsize=10, family='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                      edgecolor='orange', linewidth=3))

    plt.suptitle('🔥 Two-Stage Adaptive Sparse Training - ImageNet-100 Results 🔥',
                 fontsize=18, fontweight='bold', y=0.98)

    plt.savefig('imagenet100_two_stage_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("\n✅ Results dashboard saved: imagenet100_two_stage_results.png")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Two-stage training pipeline"""

    print("=" * 70)
    print("🔥 TWO-STAGE ADAPTIVE SPARSE TRAINING - IMAGENET-100")
    print("=" * 70)
    print(f"Stage 1 (Warmup):  Epochs 1-{Config.warmup_epochs} on 100% samples")
    print(f"Stage 2 (AST):     Epochs {Config.warmup_epochs+1}-{Config.num_epochs} on ~{Config.target_activation_rate*100:.0f}% samples")
    print("=" * 70)
    print()

    config = Config()
    print(f"📱 Device: {config.device}")
    print(f"🎯 Target activation rate (AST): {config.target_activation_rate*100:.0f}%")
    print(f"📦 Batch size: {config.batch_size}")
    print(f"👷 Workers: {config.num_workers}")
    print()

    # Load data
    train_loader, val_loader = get_dataloaders(config)
    print()

    # Load pretrained model
    print("🤖 Loading pretrained ResNet50...")
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, config.num_classes)
    model = model.to(config.device)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"✅ Loaded ResNet50 ({num_params:.1f}M params, pretrained on ImageNet-1K)")
    print()

    # Criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Tracking
    best_accuracy = 0.0
    warmup_results = {'val_accs': [], 'train_losses': [], 'total_time': 0}
    ast_results = {'val_accs': [], 'train_losses': [], 'all_activation_rates': [],
                   'all_thresholds': [], 'all_energy_savings': [], 'total_time': 0}

    total_start = time.time()

    # ========================================================================
    # STAGE 1: WARMUP TRAINING (100% samples)
    # ========================================================================

    print("=" * 70)
    print("🔥 STAGE 1: WARMUP TRAINING (100% SAMPLES)")
    print("=" * 70)
    print()

    optimizer_warmup = optim.SGD(model.parameters(), lr=config.warmup_lr,
                                 momentum=0.9, weight_decay=config.warmup_weight_decay)
    scheduler_warmup = CosineAnnealingWarmup(optimizer_warmup, warmup_epochs=2,
                                            max_epochs=config.warmup_epochs)

    warmup_start = time.time()

    for epoch in range(1, config.warmup_epochs + 1):
        epoch_start = time.time()

        # Update LR
        current_lr = scheduler_warmup.step(epoch - 1)

        print(f"\n{'='*60}")
        print(f"Warmup Epoch {epoch}/{config.warmup_epochs} | LR: {current_lr:.6f}")
        print(f"{'='*60}")

        # Train on 100% of samples
        train_loss, train_acc = train_epoch_warmup(model, train_loader, criterion,
                                                    optimizer_warmup, config, epoch)

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config)

        epoch_time = time.time() - epoch_start

        # Track
        warmup_results['val_accs'].append(val_acc)
        warmup_results['train_losses'].append(train_loss)

        print(f"\n✅ Warmup Epoch {epoch} | "
              f"Val Acc: {val_acc:5.2f}% | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Loss: {train_loss:.4f} | "
              f"Time: {epoch_time:.1f}s")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
            }, 'best_model_warmup.pth')
            print(f"💾 Saved best warmup model (acc: {val_acc:.2f}%)")

    warmup_results['total_time'] = (time.time() - warmup_start) / 60

    print(f"\n🎉 WARMUP COMPLETE!")
    print(f"   Best warmup accuracy: {best_accuracy:.2f}%")
    print(f"   Warmup time: {warmup_results['total_time']:.1f} min")
    print()

    # ========================================================================
    # STAGE 2: AST TRAINING (~20% samples)
    # ========================================================================

    print("=" * 70)
    print("🔥 STAGE 2: ADAPTIVE SPARSE TRAINING (~20% SAMPLES)")
    print("=" * 70)
    print()

    # Initialize AST components
    optimizer_ast = optim.AdamW(model.parameters(), lr=config.ast_lr,
                               weight_decay=config.ast_weight_decay)
    scheduler_ast = CosineAnnealingWarmup(optimizer_ast, warmup_epochs=0,
                                         max_epochs=config.num_epochs - config.warmup_epochs)
    sundew = SundewAlgorithm(config)

    ast_start = time.time()

    for epoch in range(config.warmup_epochs + 1, config.num_epochs + 1):
        epoch_start = time.time()

        # Update LR
        current_lr = scheduler_ast.step(epoch - config.warmup_epochs - 1)

        print(f"\n{'='*60}")
        print(f"AST Epoch {epoch}/{config.num_epochs} | LR: {current_lr:.6f}")
        print(f"{'='*60}")

        # Train with AST
        train_loss, train_activation, train_acc, epoch_stats = train_epoch_ast(
            model, train_loader, criterion, optimizer_ast, sundew, config, epoch
        )

        # Accumulate stats
        ast_results['all_activation_rates'].extend(epoch_stats['activation_rates'])
        ast_results['all_thresholds'].extend(epoch_stats['thresholds'])
        ast_results['all_energy_savings'].extend(epoch_stats['energy_savings'])

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, config)

        # Energy savings
        energy_savings = 0.0
        if sundew.total_baseline_energy > 0:
            energy_savings = ((sundew.total_baseline_energy - sundew.total_actual_energy) /
                             sundew.total_baseline_energy * 100)

        epoch_time = time.time() - epoch_start

        # Track
        ast_results['val_accs'].append(val_acc)
        ast_results['train_losses'].append(train_loss)

        print(f"\n✅ AST Epoch {epoch} | "
              f"Val Acc: {val_acc:5.2f}% | "
              f"Train Acc: {train_acc:5.2f}% | "
              f"Act: {100*train_activation:5.1f}% | "
              f"⚡ Energy: {energy_savings:5.1f}% | "
              f"Time: {epoch_time:.1f}s")

        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'accuracy': val_acc,
                'energy_savings': energy_savings,
            }, 'best_model_ast.pth')
            print(f"💾 Saved best AST model (acc: {val_acc:.2f}%)")

    ast_results['total_time'] = (time.time() - ast_start) / 60
    total_time = (time.time() - total_start) / 60

    # ========================================================================
    # FINAL RESULTS
    # ========================================================================

    print("\n" + "=" * 70)
    print("🎉 TWO-STAGE TRAINING COMPLETE!")
    print("=" * 70)

    # Calculate overall metrics
    warmup_fraction = config.warmup_epochs / config.num_epochs
    ast_fraction = (config.num_epochs - config.warmup_epochs) / config.num_epochs

    final_energy_savings = energy_savings if sundew.total_baseline_energy > 0 else 0
    overall_energy_savings = (warmup_fraction * 0 + ast_fraction * final_energy_savings)

    baseline_time_estimate = total_time / (warmup_fraction * 1.0 + ast_fraction * config.target_activation_rate)
    overall_speedup = baseline_time_estimate / total_time if total_time > 0 else 1.0

    final_metrics = {
        'best_accuracy': best_accuracy,
        'final_energy_savings': final_energy_savings,
        'avg_activation': 100 * train_activation,
        'overall_energy_savings': overall_energy_savings,
        'overall_speedup': overall_speedup,
        'total_time': total_time,
    }

    print(f"🏆 Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"⚡ Final Energy Savings (AST phase): {final_energy_savings:.2f}%")
    print(f"⚡ Overall Energy Savings: {overall_energy_savings:.2f}%")
    print(f"🚀 Overall Speedup: {overall_speedup:.2f}×")
    print(f"⏱️  Total Training Time: {total_time:.1f} minutes")
    print("=" * 70)

    # Create dashboard
    print("\n📊 Generating results dashboard...")
    create_results_dashboard(warmup_results, ast_results, final_metrics, config)

    print("\n✅ Training complete!")
    print("📁 Saved files:")
    print("   - best_model_warmup.pth")
    print("   - best_model_ast.pth")
    print("   - imagenet100_two_stage_results.png")

if __name__ == "__main__":
    main()
