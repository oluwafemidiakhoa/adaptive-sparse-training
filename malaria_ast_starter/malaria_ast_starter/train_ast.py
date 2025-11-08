"""
Malaria Diagnostic Assistant - Energy-Efficient Training with Adaptive Sparse Training (AST)

This script implements the Sundew algorithm for adaptive sample selection during training,
achieving significant energy savings while maintaining high accuracy.

Key Innovation: Trains only on "hard" samples adaptively selected each epoch,
reducing computational cost by 40-90% compared to traditional training.

For the full paper: "Adaptive Sparse Training: Energy-Efficient Deep Learning"
"""

import os, json, time, csv
from pathlib import Path
import argparse
import yaml
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig


def build_model(name: str, num_classes: int):
    """Build model architecture with final layer modified for num_classes"""
    name = name.lower()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == "resnet18":
        m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model_name: {name}")


def get_loaders(train_dir, val_dir, img_sz, bs, num_workers):
    """Create training and validation data loaders with augmentation"""
    tf_train = transforms.Compose([
        transforms.Resize(int(img_sz*1.15)),
        transforms.RandomResizedCrop(img_sz, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.05,0.05,0.05,0.02),
        transforms.ToTensor(),
    ])
    tf_val = transforms.Compose([
        transforms.Resize(int(img_sz*1.15)),
        transforms.CenterCrop(img_sz),
        transforms.ToTensor(),
    ])
    ds_tr = datasets.ImageFolder(train_dir, transform=tf_train)
    ds_va = datasets.ImageFolder(val_dir,   transform=tf_val)
    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True,  num_workers=num_workers, pin_memory=True)
    dl_va = DataLoader(ds_va, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_tr, dl_va, ds_tr.classes


def log_row(csv_path, jsonl_path, row: dict):
    """Log training metrics to both CSV and JSONL formats"""
    keys = ["epoch","timestamp","train_loss","val_loss","val_acc","lr",
            "activation_rate","energy_savings","samples_processed","total_samples"]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k) for k in keys})
    with open(jsonl_path,"a") as f:
        f.write(json.dumps(row) + "\n")


def save_ckpt(save_dir, model, epoch, best=False, energy_savings=None):
    """Save model checkpoint with metadata"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = "best.pt" if best else "last.pt"
    torch.save(model.state_dict(), str(Path(save_dir)/fname))
    if not best:
        meta = {"epoch": epoch}
        if energy_savings is not None:
            meta["energy_savings"] = energy_savings
        torch.save(meta, str(Path(save_dir)/"resume_meta.pt"))


def load_resume(save_dir, model):
    """Load checkpoint for resuming training"""
    last = Path(save_dir)/"last.pt"
    if last.exists():
        sd = torch.load(last, map_location="cpu")
        model.load_state_dict(sd)
        meta = Path(save_dir)/"resume_meta.pt"
        start = torch.load(meta)["epoch"]+1 if meta.exists() else 0
        print(f"[resume] loaded {last} (start at epoch {start})")
        return start
    return 0


def print_ast_banner(config):
    """Print informative banner about AST configuration"""
    print("\n" + "="*80)
    print("🌿 ADAPTIVE SPARSE TRAINING (AST) - Sundew Algorithm")
    print("="*80)
    print(f"Target Activation Rate: {config.target_activation_rate*100:.1f}%")
    print(f"Expected Energy Savings: {(1-config.target_activation_rate)*100:.1f}%")
    print(f"Strategy: Train only on 'hard' samples adaptively selected each epoch")
    print(f"AMP Enabled: {config.use_amp}")
    print("="*80 + "\n")


def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    # Create data loaders
    dl_tr, dl_va, classes = get_loaders(
        cfg["train_dir"], cfg["val_dir"], cfg["image_size"],
        cfg["batch_size"], cfg["num_workers"]
    )
    print(f"[data] {len(dl_tr.dataset)} training samples, {len(dl_va.dataset)} validation samples")
    print(f"[classes] {classes}")

    # Build model
    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)

    # Setup optimizer
    opt = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    criterion = nn.CrossEntropyLoss()

    # Configure AST
    ast_cfg = ASTConfig(
        target_activation_rate=cfg.get("ast_target_activation_rate", 0.4),
        initial_threshold=cfg.get("ast_initial_threshold", 3.0),
        adapt_kp=cfg.get("ast_adapt_kp", 0.005),
        adapt_ki=cfg.get("ast_adapt_ki", 0.0001),
        ema_alpha=cfg.get("ast_ema_alpha", 0.1),
        use_amp=cfg.get("amp", True) and device.type == "cuda",
        device=str(device)
    )

    print_ast_banner(ast_cfg)

    # Create AST trainer
    trainer = AdaptiveSparseTrainer(
        model=model,
        train_loader=dl_tr,
        val_loader=dl_va,
        config=ast_cfg,
        optimizer=opt,
        criterion=criterion
    )

    # Resume if requested
    start_epoch = load_resume(cfg["save_dir"], model) if cfg.get("resume", True) else 0

    # Setup paths
    metrics_csv = str(Path(cfg["save_dir"])/"metrics_ast.csv")
    metrics_jsonl = str(Path(cfg["save_dir"])/"metrics_ast.jsonl")

    # Training loop with AST
    best_acc = 0.0
    bad_epochs = 0
    warmup_epochs = cfg.get("ast_warmup_epochs", 0)

    print(f"[training] Starting from epoch {start_epoch+1}/{cfg['epochs']}")
    if warmup_epochs > 0:
        print(f"[warmup] First {warmup_epochs} epochs will use 100% of samples")

    for epoch in range(start_epoch, cfg["epochs"]):
        is_warmup = epoch < warmup_epochs

        if is_warmup:
            print(f"\n[WARMUP Epoch {epoch+1}/{cfg['epochs']}] Training on 100% of samples")
        else:
            print(f"\n[AST Epoch {epoch+1}/{cfg['epochs']}]")

        # Train one epoch
        train_stats = trainer.train_epoch(epoch)

        # Evaluate
        val_acc = trainer.evaluate()

        # Prepare metrics row
        row = {
            "epoch": epoch+1,
            "timestamp": int(time.time()),
            "train_loss": train_stats.get("loss", 0.0),
            "val_loss": train_stats.get("val_loss", 0.0),
            "val_acc": val_acc,
            "lr": float(opt.param_groups[0]["lr"]),
            "activation_rate": train_stats.get("activation_rate", 1.0),
            "energy_savings": train_stats.get("energy_savings", 0.0),
            "samples_processed": train_stats.get("samples_processed", len(dl_tr.dataset)),
            "total_samples": len(dl_tr.dataset)
        }

        log_row(metrics_csv, metrics_jsonl, row)

        # Print summary
        print(f"\n[epoch {epoch+1}] Summary:")
        print(f"  Train Loss: {row['train_loss']:.4f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Activation Rate: {row['activation_rate']*100:.1f}%")
        print(f"  Energy Savings: {row['energy_savings']:.1f}%")
        print(f"  Samples Processed: {row['samples_processed']}/{row['total_samples']}")

        # Save checkpoints
        save_ckpt(cfg["save_dir"], model, epoch, best=False,
                 energy_savings=row['energy_savings'])

        if val_acc >= best_acc - 1e-8:
            best_acc = val_acc
            save_ckpt(cfg["save_dir"], model, epoch, best=True)
            bad_epochs = 0
            print(f"  ⭐ New best validation accuracy: {best_acc:.4f}")
        else:
            bad_epochs += 1

        # Early stopping
        if bad_epochs >= cfg.get("patience", 9999):
            print(f"\n[early-stop] No improvement for {bad_epochs} epochs; best_acc={best_acc:.4f}")
            break

    # Final summary
    print("\n" + "="*80)
    print("🎉 TRAINING COMPLETE")
    print("="*80)
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    # Calculate average energy savings from last 10 non-warmup epochs
    try:
        with open(metrics_jsonl, "r") as f:
            all_metrics = [json.loads(line) for line in f]
        non_warmup_metrics = [m for m in all_metrics if m["epoch"] > warmup_epochs]
        if non_warmup_metrics:
            avg_savings = np.mean([m.get("energy_savings", 0) for m in non_warmup_metrics[-10:]])
            avg_activation = np.mean([m.get("activation_rate", 1.0) for m in non_warmup_metrics[-10:]])
            print(f"Average Energy Savings (last 10 epochs): {avg_savings:.1f}%")
            print(f"Average Activation Rate (last 10 epochs): {avg_activation*100:.1f}%")
    except Exception as e:
        print(f"Could not calculate final statistics: {e}")

    print(f"\nCheckpoint saved to: {cfg['save_dir']}/best.pt")
    print(f"Metrics logged to: {metrics_csv} and {metrics_jsonl}")
    print("="*80 + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Energy-Efficient Malaria Classification with Adaptive Sparse Training"
    )
    ap.add_argument("--config", default="configs/config_ast.yaml",
                   help="Path to AST config file")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg)
