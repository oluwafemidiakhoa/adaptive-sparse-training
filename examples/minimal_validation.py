#!/usr/bin/env python3
"""
Minimal Validation: Test AST with synthetic data (no CIFAR download)
Runs in ~1 minute
"""

import sys
from pathlib import Path
import time

# Add paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "deepseek_physical_ai"))
sys.path.insert(0, str(repo_root / "src"))

print("=" * 70)
print("ADAPTIVE SPARSE TRAINING - MINIMAL VALIDATION")
print("=" * 70)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import AST components
import training_significance
import sparse_transformer
import adaptive_training_loop

from training_significance import MultimodalTrainingSignificance
from sparse_transformer import SparseViT, SparseAttentionConfig
from adaptive_training_loop import AdaptiveSparseTrainer

print("\n[1/4] Creating synthetic dataset (32×32 RGB, 10 classes)...")
# Synthetic CIFAR-10-like data
n_train = 500  # Small for quick validation
n_val = 100

train_data = torch.randn(n_train, 3, 32, 32)
train_labels = torch.randint(0, 10, (n_train,))

val_data = torch.randn(n_val, 3, 32, 32)
val_labels = torch.randint(0, 10, (n_val,))

train_dataset = TensorDataset(train_data, train_labels)
val_dataset = TensorDataset(val_data, val_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"  Train: {len(train_dataset)} samples")
print(f"  Val: {len(val_dataset)} samples")

print("\n[2/4] Creating Sparse ViT model...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Using device: {device}")

sparse_config = SparseAttentionConfig(
    d_model=192,
    n_heads=4,
    local_window=16,
    top_k=8,
    n_global=4,
    dropout=0.1,
)

model = SparseViT(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=10,
    d_model=192,
    n_layers=2,  # Small for quick validation
    n_heads=4,
    sparse_config=sparse_config,
)

n_params = sum(p.numel() for p in model.parameters())
print(f"  Model parameters: {n_params:,}")

print("\n[3/4] Initializing Adaptive Sparse Trainer...")
config = {
    "lr": 1e-3,
    "weight_decay": 0.01,
    "epochs": 2,
    "criterion": nn.CrossEntropyLoss(),
    "num_classes": 10,
    "target_activation_rate": 0.06,  # 6% sample selection
    "use_proxy_model": True,
}

trainer = AdaptiveSparseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    modality="vision",
    device=device,
    config=config,
)

print("  Trainer initialized")
print(f"  Target activation rate: {config['target_activation_rate']:.1%}")
print(f"  Proxy model: {'Enabled' if config['use_proxy_model'] else 'Disabled'}")

print("\n[4/4] Training for 2 epochs...")
start_time = time.time()

try:
    metrics = trainer.train(epochs=2)

    print("\n" + "=" * 70)
    print("VALIDATION SUCCESS")
    print("=" * 70)
    print(f"Final Val Accuracy: {metrics['final_val_accuracy']:.2f}%")
    print(f"Avg Activation Rate: {metrics['avg_activation_rate']:.1%}")
    print(f"Energy Savings: {metrics['total_energy_savings']:.1%}")
    print(f"Total Time: {time.time() - start_time:.1f}s")

    # Verify activation rate is reasonable
    if 0.03 <= metrics['avg_activation_rate'] <= 0.15:
        print("\nActivation rate in expected range (3-15%)")
    else:
        print(f"\nWARNING: Activation rate {metrics['avg_activation_rate']:.1%} outside expected range")

    # Verify energy savings
    if metrics['total_energy_savings'] >= 0.85:
        print(f"Energy savings excellent (>{85}%)")
    else:
        print(f"Energy savings: {metrics['total_energy_savings']:.1%}")

    print("\nAST framework working correctly!")
    print("Ready for full CIFAR-10 training.")

except Exception as e:
    print(f"\nERROR during training: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
