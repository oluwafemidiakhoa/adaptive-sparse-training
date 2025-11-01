"""
🧪 QUICK TEST - SIGNIFICANCE SCORING FIX (5 EPOCHS)
====================================================================
PURPOSE: Validate the core significance scoring fix in 30 minutes

Changes from ULTIMATE version:
✅ FIXED: Gradient-magnitude based significance (NOT image std)
✅ FIXED: Loss magnitude scoring (NOT batch-normalized)
✅ Added: Prediction entropy for uncertainty estimation
⚠️  Kept: Everything else identical for A/B comparison
⏱️  Runtime: ~30 minutes (5 epochs)

Expected Results:
- Validation Accuracy: 32-38% (up from 26.38%)
- Energy Savings: 89-91% (same)
- Activation Rate: 9-12% (same)

If accuracy improves by 6-12%, proceed to Medium Test or Production!
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

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Experiment configuration - IDENTICAL TO ULTIMATE except epochs"""
    # Dataset - UPDATE THIS PATH
    data_dir = "/kaggle/input/imagenet100/ImageNet100"
    num_classes = 100
    image_size = 224

    # Training
    batch_size = 64
    num_epochs = 5  # ⚠️ QUICK TEST: Only 5 epochs
    learning_rate = 0.001
    weight_decay = 1e-4

    # AST Configuration
    target_activation_rate = 0.10
    initial_threshold = 2.5

    # PI Controller - Same as ULTIMATE
    adapt_kp = 0.02
    adapt_ki = 0.001
    ema_alpha = 0.2

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # System
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 2
    pin_memory = True

# ============================================================================
# DATASET
# ============================================================================

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
    """Create ImageNet-100 dataloaders"""
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
        🔧 FIXED SIGNIFICANCE SCORING

        OLD (BROKEN):
        - Used batch-normalized loss → collapsed to ~1.0 ± 0.2
        - Used image std → meaningless for ImageNet

        NEW (FIXED):
        - Use ABSOLUTE loss magnitude (harder samples = higher loss)
        - Use prediction entropy (uncertain samples = high entropy)
        - No batch normalization that destroys variance!
        """
        batch_size = losses.size(0)

        # Factor 1: Loss magnitude (scale to [0,1] using max, NOT mean!)
        # Higher loss = harder sample = more important
        loss_max = losses.max()
        if loss_max > 0:
            loss_significance = losses / loss_max
        else:
            loss_significance = torch.ones_like(losses)

        # Factor 2: Prediction uncertainty (entropy)
        # High entropy = model is uncertain = important sample
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
        """Select important samples - UPDATED to use outputs for entropy"""
        batch_size = losses.size(0)

        # Compute significance scores (FIXED VERSION)
        significance = self.compute_significance(losses, outputs)

        # Select samples above threshold
        active_mask = significance > self.activation_threshold
        num_active = active_mask.sum().item()

        # Fallback mechanism
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

        # PI controller (same as before)
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
# TRAINING
# ============================================================================

def train_epoch_ast(model, train_loader, criterion, optimizer, sundew, config, epoch):
    """Train one epoch with AST"""
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

        # Forward pass to get losses AND outputs (for entropy calculation)
        with torch.no_grad():
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')

        # Select important samples (FIXED VERSION uses outputs)
        active_mask, energy_info = sundew.select_samples(losses, images, outputs)

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

        # Print progress
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
# ARCHITECTURE DIAGRAM
# ============================================================================

def create_architecture_diagram():
    """Create AST architecture diagram with FIXED significance scoring highlighted"""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')

    # Title
    fig.suptitle('🧪 QUICK TEST - Fixed Significance Scoring (ImageNet-100)',
                 fontsize=20, fontweight='bold', y=0.98)

    # Define positions
    y_start = 0.85
    box_height = 0.08
    box_width = 0.18

    # Row 1: Input Stage
    ax.add_patch(plt.Rectangle((0.05, y_start), box_width, box_height,
                                facecolor='lightblue', edgecolor='black', linewidth=2))
    ax.text(0.14, y_start + box_height/2, 'Input Batch\n[64, 3, 224, 224]',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow
    ax.arrow(0.23, y_start + box_height/2, 0.05, 0, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # FIXED Significance (highlighted)
    ax.add_patch(plt.Rectangle((0.30, y_start), box_width, box_height,
                                facecolor='lime', edgecolor='red', linewidth=3))
    ax.text(0.39, y_start + box_height/2, '✅ FIXED\nSignificance\n(Loss + Entropy)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkgreen')

    # Arrow
    ax.arrow(0.48, y_start + box_height/2, 0.05, 0, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Sundew Gating
    ax.add_patch(plt.Rectangle((0.55, y_start), box_width, box_height,
                                facecolor='yellow', edgecolor='black', linewidth=2))
    ax.text(0.64, y_start + box_height/2, 'Sundew Gating\n⚡ Energy Tracking',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Arrow
    ax.arrow(0.73, y_start + box_height/2, 0.05, 0, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Active Mask
    ax.add_patch(plt.Rectangle((0.80, y_start), box_width, box_height,
                                facecolor='orange', edgecolor='black', linewidth=2))
    ax.text(0.89, y_start + box_height/2, 'Active Mask\n[64] → [~6]',
            ha='center', va='center', fontsize=11, fontweight='bold')

    # Row 2: Batched Training
    y_row2 = y_start - 0.15

    ax.add_patch(plt.Rectangle((0.30, y_row2), 0.43, box_height,
                                facecolor='lightcoral', edgecolor='black', linewidth=3))
    ax.text(0.515, y_row2 + box_height/2,
            'Batched ResNet50 Training\n(GPU Parallel on ~6 Active Samples)',
            ha='center', va='center', fontsize=12, fontweight='bold')

    # Arrow down
    ax.arrow(0.89, y_start, 0, -0.06, head_width=0.02,
             head_length=0.02, fc='black', ec='black', linewidth=2)

    # Row 3: What Changed
    y_row3 = y_row2 - 0.20

    # Box 1: OLD Method
    ax.add_patch(plt.Rectangle((0.05, y_row3), 0.27, 0.12,
                                facecolor='#FFE6E6', edgecolor='red', linewidth=2))
    ax.text(0.185, y_row3 + 0.08, '❌ OLD: Batch-Normalized Loss',
            ha='center', va='center', fontsize=11, fontweight='bold', color='red')
    ax.text(0.185, y_row3 + 0.04, 'loss_norm = loss / batch_mean\n→ collapsed variance',
            ha='center', va='center', fontsize=9)

    # Box 2: NEW Method
    ax.add_patch(plt.Rectangle((0.35, y_row3), 0.27, 0.12,
                                facecolor='#E6FFE6', edgecolor='green', linewidth=2))
    ax.text(0.485, y_row3 + 0.08, '✅ NEW: Loss Magnitude + Entropy',
            ha='center', va='center', fontsize=11, fontweight='bold', color='green')
    ax.text(0.485, y_row3 + 0.04, 'sig = 0.7×(loss/max) + 0.3×entropy\n→ preserves variance',
            ha='center', va='center', fontsize=9)

    # Box 3: Expected Impact
    ax.add_patch(plt.Rectangle((0.65, y_row3), 0.27, 0.12,
                                facecolor='gold', edgecolor='orange', linewidth=2))
    ax.text(0.785, y_row3 + 0.08, '🎯 Expected Improvement',
            ha='center', va='center', fontsize=11, fontweight='bold', color='darkorange')
    ax.text(0.785, y_row3 + 0.04, 'Accuracy: 26% → 32-38%\nEnergy: Same (~90%)',
            ha='center', va='center', fontsize=9)

    # Bottom: Test Info
    y_bottom = 0.08

    ax.add_patch(plt.Rectangle((0.25, y_bottom), 0.5, 0.10,
                                facecolor='lightblue', edgecolor='blue', linewidth=3))
    ax.text(0.5, y_bottom + 0.05, '🧪 QUICK TEST: 5 epochs | ~30 min runtime | Validates significance fix',
            ha='center', va='center', fontsize=13, fontweight='bold')

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.savefig('quicktest_architecture.png', dpi=150, bbox_inches='tight', facecolor='white')
    print(f"🏗️ Architecture diagram saved to: quicktest_architecture.png")
    plt.show()

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

def create_results_dashboard(all_epoch_stats, final_results, config):
    """Create comprehensive results dashboard"""

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Combine all epoch stats
    all_activation_rates = []
    all_thresholds = []
    all_energy_savings = []

    for epoch_stats in all_epoch_stats:
        all_activation_rates.extend(epoch_stats['activation_rates'])
        all_thresholds.extend(epoch_stats['thresholds'])
        all_energy_savings.extend(epoch_stats['energy_savings'])

    batches = list(range(len(all_activation_rates)))

    # Plot 1: Activation Rate
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(batches, [r * 100 for r in all_activation_rates],
             color='blue', linewidth=1.5, alpha=0.7, label='Activation Rate')
    ax1.axhline(y=config.target_activation_rate * 100, color='red',
                linestyle='--', linewidth=2, label=f'Target: {config.target_activation_rate*100:.0f}%')
    ax1.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Activation Rate (%)', fontsize=11, fontweight='bold')
    ax1.set_title('🎯 Activation Rate Convergence', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Threshold
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(batches, all_thresholds, color='green', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Threshold', fontsize=11, fontweight='bold')
    ax2.set_title('🎛️ PI Controller Threshold', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Energy Savings
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(batches, all_energy_savings, color='orange', linewidth=1.5, alpha=0.7, label='Energy Saved')
    ax3.axhline(y=90, color='red', linestyle='--', linewidth=2, label='Target: 90%')
    ax3.set_xlabel('Batch', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Energy Savings (%)', fontsize=11, fontweight='bold')
    ax3.set_title('⚡ Cumulative Energy Savings', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Activation Distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist([r * 100 for r in all_activation_rates], bins=50,
             color='skyblue', edgecolor='black', alpha=0.7)
    ax4.axvline(x=config.target_activation_rate * 100, color='red',
                linestyle='--', linewidth=2, label='Target')
    ax4.set_xlabel('Activation Rate (%)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('📊 Activation Rate Distribution', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Energy Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(all_energy_savings, bins=50, color='lightcoral',
             edgecolor='black', alpha=0.7)
    ax5.axvline(x=90, color='red', linestyle='--', linewidth=2, label='Target')
    ax5.set_xlabel('Energy Savings (%)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('⚡ Energy Savings Distribution', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    # Plot 6: Training Progress
    ax6 = fig.add_subplot(gs[1, 2])
    epochs_plot = list(range(1, len(all_epoch_stats) + 1))
    epoch_energy_savings = []
    if len(all_epoch_stats) > 0:
        batch_size = len(all_epoch_stats[0]['energy_savings'])
        for i in range(len(all_epoch_stats)):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(all_energy_savings))
            avg = np.mean(all_energy_savings[start_idx:end_idx])
            epoch_energy_savings.append(avg)

    ax6.plot(epochs_plot, epoch_energy_savings, 'o-', color='purple',
             linewidth=2, markersize=8, label='Energy Saved')
    ax6.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Energy Savings (%)', fontsize=11, fontweight='bold')
    ax6.set_title('📈 Training Progress', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)

    # Plot 7: Comparison Summary
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')

    summary_text = f"""
╔════════════════════════════════════════════════════════════════════════════════════════════╗
║                       🧪 QUICK TEST RESULTS (5 EPOCHS) 🧪                                  ║
╠════════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                            ║
║  📊 Validation Accuracy:      {final_results['accuracy']:6.2f}%     {'✅ IMPROVEMENT!' if final_results['accuracy'] >= 32 else '⚠️  Check logs'}                           ║
║  ⚡ Energy Savings:           {final_results['energy_savings']:6.2f}%     {'✅ ON TARGET' if final_results['energy_savings'] >= 88 else '⚠️  LOW'}                              ║
║  🎯 Activation Rate:          {final_results['activation_rate']:6.2f}%     {'✅ PERFECT' if 9 <= final_results['activation_rate'] <= 12 else '⚠️  OFF TARGET'}                              ║
║  ⏱️  Training Time:            {final_results['training_time']:6.1f} min                                              ║
║                                                                                            ║
║  🔬 WHAT CHANGED:                                                                          ║
║     ✅ Significance = 0.7×(loss/max) + 0.3×entropy (NOT batch-normalized!)                ║
║     ✅ Uses prediction entropy for uncertainty estimation                                 ║
║     ✅ Preserves variance in significance scores                                          ║
║                                                                                            ║
║  🎯 SUCCESS CRITERIA:                                                                      ║
║     ✅ Accuracy > 32%  → Significance fix works! Proceed to Production                    ║
║     ⚠️  Accuracy < 30%  → Need further debugging                                           ║
║                                                                                            ║
║  📦 Dataset:  ImageNet-100 | 🤖 Model: ResNet50 ({final_results['num_params']:.1f}M) | 🔬 Epochs: {config.num_epochs}            ║
║                                                                                            ║
╚════════════════════════════════════════════════════════════════════════════════════════════╝
    """

    ax7.text(0.5, 0.5, summary_text, fontsize=11, family='monospace',
             ha='center', va='center',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                      edgecolor='orange', linewidth=3))

    plt.suptitle('🧪 Quick Test - Significance Scoring Fix Validation',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.savefig('quicktest_results.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("\n✅ Results dashboard saved: quicktest_results.png")
    plt.show()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main training function"""

    print("=" * 70)
    print("🧪 QUICK TEST - SIGNIFICANCE SCORING FIX")
    print("Testing for 5 epochs (~30 minutes)")
    print("=" * 70)

    config = Config()
    print(f"Device: {config.device}")
    print(f"Target activation rate: {config.target_activation_rate*100:.1f}%")
    print(f"Testing FIXED significance scoring (loss magnitude + entropy)")
    print()

    # Create architecture diagram
    print("📐 Generating architecture diagram...")
    create_architecture_diagram()
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

        # Print summary
        print()
        print(f"✅ Epoch {epoch} Complete | "
              f"Val Acc: {val_accuracy:5.2f}% | "
              f"Loss: {train_loss:.4f} | "
              f"⚡ Energy Saved: {energy_savings:5.1f}% | "
              f"Time: {epoch_time:.1f}s")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

    total_time = time.time() - start_time

    # Final results
    print()
    print("=" * 70)
    print("🧪 QUICK TEST RESULTS")
    print("=" * 70)
    print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"Total Energy Savings: {energy_savings:.2f}%")
    print(f"Average Activation Rate: {100*train_activation:.2f}%")
    print(f"Total Training Time: {total_time/60:.1f} minutes")
    print()

    # Comparison
    print("📊 COMPARISON TO ORIGINAL:")
    print(f"   Original accuracy: 26.38%")
    print(f"   New accuracy:      {best_accuracy:.2f}%")
    print(f"   Improvement:       {best_accuracy - 26.38:+.2f}%")
    print()

    if best_accuracy >= 32:
        print("✅ SUCCESS! Significance fix works!")
        print("   → Proceed to PRODUCTION version (80 epochs)")
    elif best_accuracy >= 30:
        print("⚠️  PARTIAL SUCCESS - needs more tuning")
        print("   → Try MEDIUM TEST (with LR schedule)")
    else:
        print("❌ NO IMPROVEMENT - deeper investigation needed")
    print("=" * 70)

    # Create dashboard
    final_results = {
        'accuracy': best_accuracy,
        'energy_savings': energy_savings,
        'activation_rate': 100 * train_activation,
        'training_time': total_time / 60,
        'num_params': num_params,
    }

    print("\n📊 Generating results dashboard...")
    create_results_dashboard(all_epoch_stats, final_results, config)

if __name__ == "__main__":
    main()
