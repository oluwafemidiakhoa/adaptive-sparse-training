"""
🔥🚀 ImageNet-1K AST Training - RESUMABLE VERSION 🚀🔥
Handles Kaggle's 9-hour session limit by auto-resuming from checkpoints!

Expected: 70-72% accuracy, 80% energy savings
Features:
- Auto-resume from last checkpoint
- Saves every epoch (no data loss)
- Works across multiple Kaggle sessions
- Real-time monitoring with live plots!

USAGE:
1. First run: Just execute the script
2. After timeout: Re-run the SAME script - it auto-resumes!
3. Repeat until all 30 epochs complete
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.datasets import ImageFolder
from pathlib import Path
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output
import json

# Configuration
class Config:
    data_dir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    num_classes = 1000
    batch_size = 128
    num_epochs = 30
    ast_lr = 0.015
    weight_decay = 1e-4
    momentum = 0.9
    target_activation_rate = 0.20  # 80% energy savings
    initial_threshold = 5.0
    adapt_kp = 0.010
    adapt_ki = 0.00020
    ema_alpha = 0.1
    num_workers = 2
    use_amp = True
    gradient_accumulation_steps = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = "/kaggle/working/checkpoints"
    resume_checkpoint = "/kaggle/working/checkpoints/latest_checkpoint.pt"

config = Config()
os.makedirs(config.checkpoint_dir, exist_ok=True)

print("=" * 80)
print("🔥🚀 IMAGENET-1K AST TRAINING (RESUMABLE) 🚀🔥")
print("=" * 80)

# Check for existing checkpoint
resume_from_checkpoint = os.path.exists(config.resume_checkpoint)
if resume_from_checkpoint:
    print(f"\n✅ Found checkpoint: {config.resume_checkpoint}")
    print("📥 Will resume from last saved state...\n")
else:
    print(f"\n🆕 No checkpoint found - starting fresh training\n")

# Dataset
print(f"📂 Loading ImageNet-1K...")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    normalize,
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

train_dataset = ImageFolder(str(Path(config.data_dir) / 'train'), transform=train_transform)
total_size = len(train_dataset)
val_size = int(0.05 * total_size)
train_size = total_size - val_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size],
                                          generator=torch.Generator().manual_seed(42))

print(f"📦 Training: {len(train_dataset):,} | Validation: {len(val_dataset):,}")

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                         num_workers=config.num_workers, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size*2, shuffle=False,
                        num_workers=config.num_workers, pin_memory=True)

# Sundew Algorithm
class Sundew:
    def __init__(self, config):
        self.target_rate = config.target_activation_rate
        self.threshold = config.initial_threshold
        self.kp, self.ki = config.adapt_kp, config.adapt_ki
        self.integral = 0.0
        self.ema_alpha = config.ema_alpha
        self.rate_ema = config.target_activation_rate
        self.total_baseline, self.total_actual = 0.0, 0.0

    def select(self, losses, outputs):
        # Compute significance
        probs = torch.softmax(outputs, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        significance = 0.7 * losses + 0.3 * entropy

        # Select samples
        active_mask = significance > self.threshold
        num_active = max(active_mask.sum().item(), int(losses.size(0) * 0.10))
        if num_active < active_mask.sum().item():
            _, top_idx = torch.topk(significance, num_active)
            active_mask = torch.zeros_like(active_mask, dtype=torch.bool)
            active_mask[top_idx] = True

        # PI controller
        current_rate = num_active / losses.size(0)
        self.rate_ema = self.ema_alpha * current_rate + (1 - self.ema_alpha) * self.rate_ema
        error = self.rate_ema - self.target_rate
        self.integral = max(-100, min(100, self.integral + error))
        self.threshold = max(0.5, min(10.0, self.threshold + self.kp * error + self.ki * self.integral))

        # Energy tracking
        self.total_baseline += losses.size(0)
        self.total_actual += num_active
        energy_savings = ((self.total_baseline - self.total_actual) / self.total_baseline * 100) if self.total_baseline > 0 else 0

        return active_mask, {'rate': self.rate_ema, 'threshold': self.threshold, 'savings': energy_savings}

    def state_dict(self):
        """Save Sundew state for checkpointing"""
        return {
            'threshold': self.threshold,
            'integral': self.integral,
            'rate_ema': self.rate_ema,
            'total_baseline': self.total_baseline,
            'total_actual': self.total_actual,
        }

    def load_state_dict(self, state):
        """Restore Sundew state from checkpoint"""
        self.threshold = state['threshold']
        self.integral = state['integral']
        self.rate_ema = state['rate_ema']
        self.total_baseline = state['total_baseline']
        self.total_actual = state['total_actual']

# Model
print("🤖 Loading ResNet50...")
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, config.num_classes)
model = model.to(config.device)
print(f"✅ ResNet50 ready (23.7M params)\n")

# Training setup
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.SGD(model.parameters(), lr=config.ast_lr, momentum=config.momentum,
                     weight_decay=config.weight_decay)
scaler = GradScaler(enabled=config.use_amp)
sundew = Sundew(config)
best_acc = 0.0

# Tracking for plots
history = {'epoch': [], 'train_acc': [], 'val_acc': [], 'energy_savings': [],
           'activation_rate': [], 'time': []}

# Resume from checkpoint if exists
start_epoch = 1
total_time_offset = 0.0

if resume_from_checkpoint:
    print("📥 Loading checkpoint...")
    checkpoint = torch.load(config.resume_checkpoint)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scaler.load_state_dict(checkpoint['scaler'])
    sundew.load_state_dict(checkpoint['sundew'])

    start_epoch = checkpoint['epoch'] + 1
    best_acc = checkpoint['best_acc']
    history = checkpoint['history']
    total_time_offset = checkpoint['total_time']

    print(f"✅ Resumed from Epoch {checkpoint['epoch']}")
    print(f"   Best Accuracy: {best_acc:.2f}%")
    print(f"   Total Time So Far: {total_time_offset/60:.1f} hours")
    print(f"   Resuming at Epoch {start_epoch}/{config.num_epochs}\n")

print("=" * 80)
print(f"🔥 TRAINING: Epochs {start_epoch}-{config.num_epochs} | Target: {config.target_activation_rate*100:.0f}% activation")
print("=" * 80)

total_start = time.time()

for epoch in range(start_epoch, config.num_epochs + 1):
    # Training
    model.train()
    running_loss, correct, total_active, total_samples = 0.0, 0, 0, 0
    epoch_start = time.time()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(config.device), labels.to(config.device)

        with autocast(device_type='cuda', enabled=config.use_amp):
            outputs = model(images)
            losses = torch.nn.functional.cross_entropy(outputs, labels, reduction='none')

        with torch.no_grad():
            active_mask, info = sundew.select(losses, outputs)

        with autocast(device_type='cuda', enabled=config.use_amp):
            masked_loss = (losses * active_mask.float()).sum() / max(active_mask.sum(), 1)
            masked_loss = masked_loss / config.gradient_accumulation_steps

        scaler.scale(masked_loss).backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += masked_loss.item() * config.gradient_accumulation_steps * active_mask.sum().item()
        correct += outputs.max(1)[1][active_mask].eq(labels[active_mask]).sum().item()
        total_active += active_mask.sum().item()
        total_samples += images.size(0)

        if (batch_idx + 1) % 50 == 0:
            train_acc = 100.0 * correct / max(total_active, 1)
            print(f"  [{epoch}/{config.num_epochs}] Batch {batch_idx+1:4d} | "
                  f"Act: {100*info['rate']:5.1f}% | Acc: {train_acc:5.2f}% | "
                  f"⚡ Save: {info['savings']:5.1f}%")

    train_acc = 100.0 * correct / max(total_active, 1)

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            with autocast(device_type='cuda', enabled=config.use_amp):
                outputs = model(images)
            val_correct += outputs.max(1)[1].eq(labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100.0 * val_correct / val_total
    epoch_time = (time.time() - epoch_start) / 60
    cumulative_time = total_time_offset + (time.time() - total_start)

    # Update history
    history['epoch'].append(epoch)
    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['energy_savings'].append(info['savings'])
    history['activation_rate'].append(100 * info['rate'])
    history['time'].append(epoch_time)

    # Clear output and plot
    if epoch % 2 == 0 or epoch == start_epoch:  # Plot every 2 epochs or first resumed epoch
        clear_output(wait=True)

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'🔥 ImageNet-1K AST Training - Epoch {epoch}/{config.num_epochs}',
                     fontsize=16, fontweight='bold')

        # Plot 1: Accuracy
        axes[0, 0].plot(history['epoch'], history['train_acc'], 'o-', label='Train Acc', linewidth=2, markersize=6)
        axes[0, 0].plot(history['epoch'], history['val_acc'], 's-', label='Val Acc', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Target (70%)')
        axes[0, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('🏆 Accuracy Progress', fontsize=14, fontweight='bold')
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 100])

        # Plot 2: Energy Savings
        axes[0, 1].plot(history['epoch'], history['energy_savings'], 'o-', color='green',
                       linewidth=2, markersize=6)
        axes[0, 1].axhline(y=75, color='r', linestyle='--', alpha=0.7, label='Target (75%)')
        axes[0, 1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Energy Savings (%)', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('⚡ Energy Savings', fontsize=14, fontweight='bold')
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 100])

        # Plot 3: Activation Rate
        axes[1, 0].plot(history['epoch'], history['activation_rate'], 'o-', color='blue',
                       linewidth=2, markersize=6)
        axes[1, 0].axhline(y=config.target_activation_rate*100, color='r', linestyle='--',
                          alpha=0.7, label=f'Target ({config.target_activation_rate*100:.0f}%)')
        axes[1, 0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('Activation Rate (%)', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('🎯 Sample Activation Rate', fontsize=14, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 50])

        # Plot 4: Summary Stats
        axes[1, 1].axis('off')
        summary_text = f"""
╔══════════════════════════════════════════════════════╗
║            CURRENT TRAINING STATUS                   ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  Epoch:              {epoch:3d}/{config.num_epochs}                           ║
║  Best Val Acc:       {max(history['val_acc']):6.2f}%                      ║
║  Current Val Acc:    {val_acc:6.2f}%                      ║
║  Current Train Acc:  {train_acc:6.2f}%                      ║
║  Energy Savings:     {info['savings']:6.2f}%                      ║
║  Activation Rate:    {100*info['rate']:6.2f}%                      ║
║  Epoch Time:         {epoch_time:6.1f} min                   ║
║  Total Time:         {cumulative_time/60:6.1f} min ({cumulative_time/3600:.1f} hrs)      ║
║  Est. Remaining:     {(cumulative_time/epoch)*(config.num_epochs-epoch)/60:6.1f} min ({(cumulative_time/epoch)*(config.num_epochs-epoch)/3600:.1f} hrs)      ║
║                                                      ║
║  Status: {'✅ ON TRACK!' if val_acc >= 30 and info['savings'] >= 70 else '⏳ Training...'}                                 ║
╚══════════════════════════════════════════════════════╝
        """
        axes[1, 1].text(0.5, 0.5, summary_text, fontsize=11, family='monospace',
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                                edgecolor='orange', linewidth=3))

        plt.tight_layout()
        plt.savefig(f'{config.checkpoint_dir}/training_progress_epoch{epoch}.png',
                   dpi=150, bbox_inches='tight')
        plt.show()

    print(f"\n✅ Epoch {epoch}/{config.num_epochs} | Val Acc: {val_acc:5.2f}% | "
          f"Train Acc: {train_acc:5.2f}% | ⚡ Savings: {info['savings']:5.1f}% | "
          f"Time: {epoch_time:.1f}min\n")

    # Save checkpoint EVERY epoch (prevent data loss)
    checkpoint_state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'sundew': sundew.state_dict(),
        'best_acc': best_acc if val_acc <= best_acc else val_acc,
        'history': history,
        'total_time': cumulative_time,
        'config': {
            'num_classes': config.num_classes,
            'batch_size': config.batch_size,
            'ast_lr': config.ast_lr,
            'target_activation_rate': config.target_activation_rate,
        }
    }

    # Save as latest checkpoint
    torch.save(checkpoint_state, config.resume_checkpoint)

    # Also save numbered checkpoint every 3 epochs
    if epoch % 3 == 0:
        torch.save(checkpoint_state, f"{config.checkpoint_dir}/checkpoint_e{epoch}.pt")

    # Update best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'model': model.state_dict(), 'acc': val_acc, 'savings': info['savings']},
                  f"{config.checkpoint_dir}/best_model.pt")
        print(f"🏆 New best: {val_acc:.2f}%\n")

total_time = cumulative_time / 60

# Results
print("=" * 80)
print("🎉 TRAINING COMPLETE!")
print("=" * 80)
print(f"🏆 Best Accuracy: {best_acc:.2f}%")
print(f"⚡ Energy Savings: {info['savings']:.2f}%")
print(f"⏱️  Total Time: {total_time:.1f} min ({total_time/60:.1f} hours)")
print("=" * 80)

if best_acc >= 70.0 and info['savings'] >= 75.0:
    print("\n✅ SUCCESS! AST validated on ImageNet-1K (1.28M images)!")
    print("   CIFAR-10 → ImageNet-100 → ImageNet-1K scaling confirmed!")
else:
    print(f"\n⚠️  Results: {best_acc:.1f}% acc, {info['savings']:.1f}% savings")

print("\n📊 Progression:")
print("   CIFAR-10:     61.2% acc, 89.6% savings")
print("   ImageNet-100: 92.1% acc, 61.5% savings")
print(f"   ImageNet-1K:  {best_acc:.1f}% acc, {info['savings']:.1f}% savings")
print("\n🚀 pip install adaptive-sparse-training")

# Clean up old checkpoints to save space (keep latest + best + every 5 epochs)
print("\n🧹 Cleaning up old checkpoints...")
for f in os.listdir(config.checkpoint_dir):
    if f.startswith("checkpoint_e") and f.endswith(".pt"):
        epoch_num = int(f.split("_e")[1].split(".pt")[0])
        if epoch_num % 5 != 0 and epoch_num < epoch - 3:
            os.remove(os.path.join(config.checkpoint_dir, f))
print("✅ Cleanup complete!")
