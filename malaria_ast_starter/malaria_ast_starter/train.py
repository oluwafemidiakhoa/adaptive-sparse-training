\
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

def build_model(name: str, num_classes: int):
    name = name.lower()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unsupported model_name: {name}")

def get_loaders(train_dir, val_dir, img_sz, bs, num_workers):
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

def accuracy_from_logits(logits, y):
    preds = logits.argmax(1)
    return (preds == y).float().mean().item()

def log_row(csv_path, jsonl_path, row: dict):
    keys = ["epoch","timestamp","train_loss","val_loss","val_acc","lr"]
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path,"a",newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if write_header: w.writeheader()
        w.writerow({k: row.get(k) for k in keys})
    with open(jsonl_path,"a") as f:
        f.write(json.dumps(row) + "\n")

def save_ckpt(save_dir, model, epoch, best=False):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    fname = "best.pt" if best else "last.pt"
    torch.save(model.state_dict(), str(Path(save_dir)/fname))
    if not best:
        torch.save({"epoch": epoch}, str(Path(save_dir)/"resume_meta.pt"))

def load_resume(save_dir, model):
    last = Path(save_dir)/"last.pt"
    if last.exists():
        sd = torch.load(last, map_location="cpu")
        model.load_state_dict(sd)
        meta = Path(save_dir)/"resume_meta.pt"
        start = torch.load(meta)["epoch"]+1 if meta.exists() else 0
        print(f"[resume] loaded {last} (start at epoch {start})")
        return start
    return 0

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    dl_tr, dl_va, classes = get_loaders(
        cfg["train_dir"], cfg["val_dir"], cfg["image_size"], cfg["batch_size"], cfg["num_workers"]
    )

    model = build_model(cfg["model_name"], cfg["num_classes"]).to(device)
    opt = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.get("amp", True) and device.type=="cuda"))
    criterion = nn.CrossEntropyLoss()

    start_epoch = load_resume(cfg["save_dir"], model) if cfg.get("resume", True) else 0
    best_acc, bad_epochs = 0.0, 0

    metrics_csv = str(Path(cfg["save_dir"])/"metrics.csv")
    metrics_jsonl = str(Path(cfg["save_dir"])/"metrics.jsonl")

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {epoch+1}/{cfg['epochs']}")
        train_losses = []
        for xb, yb in pbar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(train_losses):.4f}")

        model.eval()
        val_losses, accs = [], []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                accs.append(accuracy_from_logits(logits, yb))
        val_loss = float(np.mean(val_losses)) if val_losses else None
        val_acc  = float(np.mean(accs)) if accs else None

        row = {
            "epoch": epoch+1,
            "timestamp": int(time.time()),
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "val_acc":  val_acc,
            "lr": float(opt.param_groups[0]["lr"]),
        }
        log_row(metrics_csv, metrics_jsonl, row)
        print(f"[epoch {epoch+1}] train_loss={row['train_loss']:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        save_ckpt(cfg["save_dir"], model, epoch, best=False)
        if val_acc is not None and val_acc >= best_acc - 1e-8:
            best_acc = val_acc
            save_ckpt(cfg["save_dir"], model, epoch, best=True)
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg.get("patience", 9999):
            print(f"[early-stop] no improvement for {bad_epochs} epochs; best_acc={best_acc:.4f}")
            break

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
