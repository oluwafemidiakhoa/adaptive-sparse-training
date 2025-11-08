"""
Quick Demo Script for AST Malaria Classification

This script demonstrates the complete AST workflow:
1. Creates a minimal synthetic dataset (for demo purposes)
2. Trains a small model with AST
3. Generates visualizations
4. Shows energy savings

For real usage, replace with actual NIH malaria dataset.
"""

import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import yaml


def create_demo_dataset(base_dir="demo_data", n_samples_per_class=100):
    """
    Create a minimal synthetic dataset for demonstration

    In production, use the real NIH malaria dataset from:
    https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
    """
    print("📁 Creating demo dataset...")

    for split in ["train", "val"]:
        for cls in ["Parasitized", "Uninfected"]:
            path = Path(base_dir) / split / cls
            path.mkdir(parents=True, exist_ok=True)

            n = n_samples_per_class if split == "train" else n_samples_per_class // 5

            for i in range(n):
                # Create synthetic cell images (random noise with slight class differences)
                img_size = 224
                if cls == "Parasitized":
                    # Infected cells: darker with some spots
                    img = np.random.randint(50, 150, (img_size, img_size, 3), dtype=np.uint8)
                    # Add some "parasites" (bright spots)
                    for _ in range(5):
                        x, y = np.random.randint(50, img_size-50, 2)
                        img[x-10:x+10, y-10:y+10] = [200, 100, 150]
                else:
                    # Uninfected cells: lighter, more uniform
                    img = np.random.randint(100, 200, (img_size, img_size, 3), dtype=np.uint8)

                # Save image
                Image.fromarray(img).save(path / f"{cls}_{i:04d}.png")

    print(f"✅ Created demo dataset at: {base_dir}/")
    print(f"   Train: {n_samples_per_class * 2} samples")
    print(f"   Val: {(n_samples_per_class // 5) * 2} samples")
    print("\n⚠️  NOTE: This is synthetic data for demo purposes!")
    print("   For real malaria detection, download the NIH dataset from:")
    print("   https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria\n")


def create_demo_config():
    """Create a minimal config for fast demo training"""
    config = {
        "model_name": "resnet18",  # Smaller model for faster demo
        "num_classes": 2,
        "image_size": 224,
        "epochs": 5,  # Just a few epochs for demo
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "num_workers": 0,  # 0 for Windows compatibility
        "amp": torch.cuda.is_available(),
        "train_dir": "demo_data/train",
        "val_dir": "demo_data/val",
        "save_dir": "demo_checkpoints",
        "resume": False,
        "patience": 10,
        # AST settings - moderate for demo
        "ast_target_activation_rate": 0.50,  # 50% activation = 50% savings
        "ast_initial_threshold": 3.0,
        "ast_adapt_kp": 0.005,
        "ast_adapt_ki": 0.0001,
        "ast_ema_alpha": 0.1,
        "ast_warmup_epochs": 1,
    }

    config_path = Path("configs/config_demo.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Created demo config at: {config_path}")
    return config_path


def run_demo():
    """Run the complete AST demo"""
    print("\n" + "="*80)
    print("🌿 ADAPTIVE SPARSE TRAINING (AST) - QUICK DEMO")
    print("="*80 + "\n")

    # Step 1: Create demo dataset
    if not Path("demo_data").exists():
        create_demo_dataset()
    else:
        print("📁 Using existing demo dataset\n")

    # Step 2: Create demo config
    config_path = create_demo_config()

    # Step 3: Train with AST
    print("\n🚀 Starting AST training...")
    print("="*80 + "\n")

    try:
        # Import and run training
        from train_ast import main
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        main(cfg)
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Generate visualizations
    print("\n📊 Generating visualizations...")

    metrics_path = "demo_checkpoints/metrics_ast.jsonl"
    if Path(metrics_path).exists():
        try:
            from visualize_ast import load_metrics, create_ast_comparison_plot, create_headline_graphic, print_summary_stats

            metrics = load_metrics(metrics_path)
            print_summary_stats(metrics)

            output_dir = Path("demo_visualizations")
            output_dir.mkdir(exist_ok=True)

            create_ast_comparison_plot(metrics, output_path=str(output_dir / "demo_results.png"))
            create_headline_graphic(metrics, output_path=str(output_dir / "demo_headline.png"))

            print(f"\n✅ Visualizations saved to: {output_dir}/")
        except Exception as e:
            print(f"\n⚠️  Visualization error: {e}")

    # Step 5: Summary
    print("\n" + "="*80)
    print("🎉 DEMO COMPLETE!")
    print("="*80)
    print("\n📋 What was demonstrated:")
    print("  ✅ Adaptive Sparse Training with Sundew algorithm")
    print("  ✅ Energy savings tracking (check metrics)")
    print("  ✅ Automatic checkpointing (demo_checkpoints/)")
    print("  ✅ Comprehensive visualizations (demo_visualizations/)")

    print("\n📊 Check the results:")
    print(f"  - Metrics: demo_checkpoints/metrics_ast.jsonl")
    print(f"  - Checkpoints: demo_checkpoints/best.pt")
    print(f"  - Visualizations: demo_visualizations/")

    print("\n🚀 Next steps for real usage:")
    print("  1. Download NIH malaria dataset from Kaggle")
    print("  2. Organize as data/train and data/val")
    print("  3. Run: python train_ast.py --config configs/config_ast.yaml")
    print("  4. Visualize: python visualize_ast.py")

    print("\n💡 For maximum buzz:")
    print("  - Set ast_target_activation_rate: 0.10 for 90% energy savings")
    print("  - Train for 30+ epochs on real data")
    print("  - Compare with baseline training")

    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Check if user wants to skip demo data creation
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-data":
        print("ℹ️  Skipping demo data creation (using existing data)")
    else:
        run_demo()
