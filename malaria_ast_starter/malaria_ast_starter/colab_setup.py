"""
Google Colab Setup Script for Malaria AST Training

This script automates the complete setup process in Colab:
1. Downloads NIH malaria dataset from Kaggle
2. Organizes data into train/val splits
3. Installs dependencies
4. Prepares for training

Run this first in your Colab notebook!
"""

import os
import shutil
import zipfile
from pathlib import Path
from sklearn.model_selection import train_test_split
import json


def check_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def setup_kaggle_api():
    """
    Setup Kaggle API credentials

    Instructions:
    1. Go to https://www.kaggle.com/settings
    2. Scroll to "API" section
    3. Click "Create New API Token"
    4. Upload the downloaded kaggle.json file when prompted
    """
    print("🔑 Setting up Kaggle API...")

    if check_colab():
        from google.colab import files
        print("\n📁 Please upload your kaggle.json file:")
        print("   (Download from: https://www.kaggle.com/settings -> API -> Create New Token)")
        uploaded = files.upload()

        if 'kaggle.json' not in uploaded:
            raise FileNotFoundError("kaggle.json not found. Please upload your Kaggle API token.")

        # Setup Kaggle credentials
        os.makedirs('/root/.kaggle', exist_ok=True)
        shutil.move('kaggle.json', '/root/.kaggle/kaggle.json')
        os.chmod('/root/.kaggle/kaggle.json', 0o600)
        print("✅ Kaggle API configured successfully!")
    else:
        print("⚠️  Not running in Colab. Please ensure kaggle.json is in ~/.kaggle/")


def download_dataset():
    """Download malaria dataset from Kaggle"""
    print("\n📦 Downloading malaria dataset from Kaggle...")
    print("   Dataset: cell-images-for-detecting-malaria")
    print("   Size: ~350 MB")

    # Install kaggle if not present
    os.system("pip install -q kaggle")

    # Download dataset
    dataset_name = "iarunava/cell-images-for-detecting-malaria"
    os.system(f"kaggle datasets download -d {dataset_name}")

    # Extract
    zip_file = "cell-images-for-detecting-malaria.zip"
    if os.path.exists(zip_file):
        print(f"📂 Extracting {zip_file}...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_file)
        print("✅ Dataset downloaded and extracted!")
    else:
        raise FileNotFoundError(f"Download failed. {zip_file} not found.")


def organize_data(source_dir="cell_images", dest_dir="data", val_split=0.2, seed=42):
    """
    Organize downloaded images into train/val splits

    Args:
        source_dir: Directory containing Parasitized/ and Uninfected/ folders
        dest_dir: Target directory for organized data
        val_split: Fraction of data for validation (default: 0.2 = 20%)
        seed: Random seed for reproducibility
    """
    print(f"\n🗂️  Organizing data into train/val splits ({int((1-val_split)*100)}%/{int(val_split*100)}%)...")

    source_path = Path(source_dir)
    dest_path = Path(dest_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    stats = {"train": {}, "val": {}}

    for class_name in ['Parasitized', 'Uninfected']:
        class_path = source_path / class_name

        if not class_path.exists():
            print(f"⚠️  Warning: {class_path} not found, skipping...")
            continue

        # Get all images
        images = list(class_path.glob('*.png'))
        print(f"   Found {len(images)} {class_name} images")

        # Split into train/val
        train_imgs, val_imgs = train_test_split(
            images, test_size=val_split, random_state=seed
        )

        # Copy to organized structure
        for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
            dest_class_path = dest_path / split / class_name
            dest_class_path.mkdir(parents=True, exist_ok=True)

            for img in img_list:
                shutil.copy(img, dest_class_path / img.name)

            stats[split][class_name] = len(img_list)

    # Print summary
    print("\n✅ Data organization complete!")
    print("\n📊 Dataset Statistics:")
    print(f"   Training:")
    for cls, count in stats['train'].items():
        print(f"      {cls}: {count:,} images")
    print(f"   Validation:")
    for cls, count in stats['val'].items():
        print(f"      {cls}: {count:,} images")

    total_train = sum(stats['train'].values())
    total_val = sum(stats['val'].values())
    print(f"\n   Total: {total_train:,} train / {total_val:,} val")

    # Save stats
    with open(dest_path / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def install_dependencies():
    """Install all required packages"""
    print("\n📦 Installing dependencies...")

    packages = [
        "torch",
        "torchvision",
        "tqdm",
        "numpy",
        "pillow",
        "pyyaml",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "adaptive-sparse-training"
    ]

    # Install in one go
    os.system(f"pip install -q {' '.join(packages)}")
    print("✅ All dependencies installed!")


def verify_gpu():
    """Check if GPU is available and print details"""
    print("\n🖥️  Checking GPU availability...")

    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU Available: {gpu_name}")
        print(f"   Memory: {gpu_memory:.1f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")

        # Recommend batch size based on GPU
        if "T4" in gpu_name:
            print("\n💡 Recommended batch_size: 32-64 (T4 has 16GB)")
        elif "P100" in gpu_name:
            print("\n💡 Recommended batch_size: 64-128 (P100 has 16GB)")
        elif "V100" in gpu_name:
            print("\n💡 Recommended batch_size: 128-256 (V100 has 16-32GB)")
        elif "A100" in gpu_name:
            print("\n💡 Recommended batch_size: 256+ (A100 has 40-80GB)")
    else:
        print("⚠️  No GPU detected. Training will be very slow on CPU.")
        print("   In Colab: Runtime -> Change runtime type -> GPU")


def create_colab_config():
    """Create optimized config for Colab training"""
    print("\n⚙️  Creating Colab-optimized config...")

    import torch
    import yaml

    # Determine optimal batch size based on GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if "T4" in gpu_name:
            batch_size = 64
        elif "P100" in gpu_name:
            batch_size = 128
        elif "V100" in gpu_name or "A100" in gpu_name:
            batch_size = 256
        else:
            batch_size = 64  # Safe default
    else:
        batch_size = 16  # CPU

    config = {
        "model_name": "efficientnet_b0",
        "num_classes": 2,
        "image_size": 224,
        "epochs": 30,
        "batch_size": batch_size,
        "learning_rate": 0.0003,
        "weight_decay": 0.0001,
        "num_workers": 2,
        "amp": True,
        "train_dir": "data/train",
        "val_dir": "data/val",
        "save_dir": "checkpoints_ast",
        "resume": True,
        "patience": 7,
        # AST settings - balanced for good results
        "ast_target_activation_rate": 0.40,  # 60% energy savings
        "ast_initial_threshold": 3.0,
        "ast_adapt_kp": 0.005,
        "ast_adapt_ki": 0.0001,
        "ast_ema_alpha": 0.1,
        "ast_warmup_epochs": 2,
    }

    config_path = Path("configs/config_colab.yaml")
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"✅ Config saved to: {config_path}")
    print(f"   Batch size: {batch_size}")
    print(f"   AST activation rate: {config['ast_target_activation_rate']*100:.0f}%")

    return config_path


def setup_colab_outputs():
    """Setup Google Drive integration for saving outputs"""
    if not check_colab():
        return

    print("\n💾 Setting up Google Drive integration...")
    print("   This will save checkpoints and results to your Drive")

    try:
        from google.colab import drive
        drive.mount('/content/drive')

        # Create project folder in Drive
        project_dir = Path("/content/drive/MyDrive/malaria_ast")
        project_dir.mkdir(exist_ok=True)

        print(f"✅ Drive mounted! Outputs will be saved to:")
        print(f"   {project_dir}")

        return project_dir
    except Exception as e:
        print(f"⚠️  Drive mount failed: {e}")
        print("   Continuing without Drive integration")
        return None


def full_setup():
    """Run complete setup pipeline"""
    print("="*80)
    print("🌿 MALARIA AST TRAINING - GOOGLE COLAB SETUP")
    print("="*80)

    # Step 1: Verify environment
    if check_colab():
        print("\n✅ Running in Google Colab")
    else:
        print("\n⚠️  Not detected as Colab environment")
        print("   Continuing anyway...")

    # Step 2: Install dependencies first
    install_dependencies()

    # Step 3: Verify GPU
    verify_gpu()

    # Step 4: Setup Kaggle API
    setup_kaggle_api()

    # Step 5: Download dataset
    download_dataset()

    # Step 6: Organize data
    stats = organize_data()

    # Step 7: Create config
    config_path = create_colab_config()

    # Step 8: Setup Drive (optional)
    drive_dir = setup_colab_outputs()

    # Final summary
    print("\n" + "="*80)
    print("🎉 SETUP COMPLETE! Ready to train!")
    print("="*80)

    print("\n📋 Next steps:")
    print("\n1. Train with AST (60% energy savings):")
    print("   !python train_ast.py --config configs/config_colab.yaml")

    print("\n2. Generate visualizations:")
    print("   !python visualize_ast.py --metrics checkpoints_ast/metrics_ast.jsonl")

    print("\n3. Evaluate model:")
    print("   !python eval.py --weights checkpoints_ast/best.pt")

    print("\n4. Create Grad-CAM visualization:")
    print("   !python gradcam_snapshot.py --weights checkpoints_ast/best.pt \\")
    print("       --image data/val/Parasitized/<pick_any_image.png> --out gradcam.png")

    if drive_dir:
        print(f"\n💡 To save outputs to Drive, copy checkpoints after training:")
        print(f"   !cp -r checkpoints_ast /content/drive/MyDrive/malaria_ast/")
        print(f"   !cp -r visualizations /content/drive/MyDrive/malaria_ast/")

    print("\n" + "="*80)

    return {
        "dataset_stats": stats,
        "config_path": str(config_path),
        "drive_path": str(drive_dir) if drive_dir else None
    }


if __name__ == "__main__":
    result = full_setup()
    print("\n✨ Setup result:", json.dumps(result, indent=2))
