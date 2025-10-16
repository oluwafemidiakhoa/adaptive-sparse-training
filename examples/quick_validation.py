#!/usr/bin/env python3
"""
Quick Validation: Test AST components before CIFAR-10 run
Runs in ~30 seconds
"""

import sys
from pathlib import Path

# Add paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root / "deepseek_physical_ai"))
sys.path.insert(0, str(repo_root / "src"))

print("=" * 70)
print("ADAPTIVE SPARSE TRAINING - QUICK VALIDATION")
print("=" * 70)

# Test 1: Import Sundew
print("\n[1/5] Testing Sundew imports...")
try:
    from sundew.config import SundewConfig
    from sundew.core import SundewAlgorithm
    print("  OK - Sundew core imported")
except Exception as e:
    print(f"  ERROR - {e}")
    sys.exit(1)

# Test 2: Import AST components
print("\n[2/5] Testing AST component imports...")
try:
    # Import from parent directory
    import sys
    from pathlib import Path
    deepseek_path = Path(__file__).parent.parent
    if str(deepseek_path) not in sys.path:
        sys.path.insert(0, str(deepseek_path))

    import training_significance
    import sparse_transformer
    import adaptive_training_loop

    from training_significance import MultimodalTrainingSignificance, TrainingSampleContext
    from sparse_transformer import DeepSeekSparseAttention, SparseAttentionConfig, SparseViT
    from adaptive_training_loop import AdaptiveSparseTrainer
    print("  OK - AST components imported")
except Exception as e:
    print(f"  ERROR - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test significance model
print("\n[3/5] Testing significance model...")
try:
    import numpy as np

    sig_model = MultimodalTrainingSignificance(modality="vision")

    # Create dummy context
    context = TrainingSampleContext(
        timestamp=0.0,
        sequence_id=0,
        features={"mean_intensity": 0.5},
        history=[],
        metadata={},
        sample_id=0,
        modality="vision",
        batch_index=0,
        epoch=0,
        current_loss=1.0,
        loss_history=[1.2, 1.1, 1.0],
    )

    significance, explanation = sig_model.compute_significance(context)

    assert 0.0 <= significance <= 1.0, f"Significance out of range: {significance}"
    assert "learning_value" in explanation
    assert "difficulty" in explanation

    print(f"  OK - Significance computed: {significance:.3f}")
    print(f"       Components: learning={explanation['learning_value']:.2f}, "
          f"difficulty={explanation['difficulty']:.2f}, "
          f"novelty={explanation['novelty']:.2f}")

except Exception as e:
    print(f"  ERROR - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test sparse attention (requires torch)
print("\n[4/5] Testing sparse attention...")
try:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")

    config = SparseAttentionConfig(
        d_model=128,
        n_heads=4,
        local_window=32,
        top_k=16,
        n_global=4,
        dropout=0.0,
    )

    sparse_attn = DeepSeekSparseAttention(config).to(device)

    # Dummy input
    batch_size, seq_len = 2, 64
    x = torch.randn(batch_size, seq_len, config.d_model).to(device)

    # Forward pass
    output, _ = sparse_attn(x)

    assert output.shape == (batch_size, seq_len, config.d_model)

    print(f"  OK - Sparse attention forward pass")
    print(f"       Input: {x.shape}, Output: {output.shape}")

except Exception as e:
    print(f"  ERROR - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Test SparseViT
print("\n[5/5] Testing SparseViT model...")
try:
    sparse_config = SparseAttentionConfig(
        d_model=192,
        n_heads=4,
        local_window=16,
        top_k=8,
        n_global=4,
    )

    model = SparseViT(
        img_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        d_model=192,
        n_layers=2,
        n_heads=4,
        sparse_config=sparse_config,
    ).to(device)

    # Dummy image
    img = torch.randn(2, 3, 32, 32).to(device)

    # Forward pass
    logits = model(img)

    assert logits.shape == (2, 10)

    print(f"  OK - SparseViT forward pass")
    print(f"       Input: {img.shape}, Output: {logits.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"       Model parameters: {n_params:,}")

except Exception as e:
    print(f"  ERROR - {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("VALIDATION COMPLETE - ALL TESTS PASSED")
print("=" * 70)
print("\nReady to run CIFAR-10 demo:")
print("  python examples/cifar10_demo.py --epochs 2")
print("\nOr with sparse attention (50× speedup):")
print("  python examples/cifar10_demo.py --model vit --sparse --epochs 2")
