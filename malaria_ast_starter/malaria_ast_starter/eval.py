\
import argparse, json
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def build_model(name, num_classes):
    name = name.lower()
    if name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
        return m
    elif name == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError("unsupported")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--val_dir", default="data/val")
    ap.add_argument("--model_name", default="efficientnet_b0")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    tf = transforms.Compose([
        transforms.Resize(int(args.image_size*1.15)),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor()
    ])
    ds = datasets.ImageFolder(args.val_dir, transform=tf)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model_name, args.num_classes).to(device)
    state = torch.load(args.weights, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state["state_dict"])
    model.eval()

    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            ys.extend(yb.numpy()); ps.extend(preds)

    rep = classification_report(ys, ps, target_names=ds.classes, output_dict=True)
    cm  = confusion_matrix(ys, ps)

    Path("checkpoints").mkdir(exist_ok=True)
    with open("checkpoints/report.json","w") as f:
        json.dump(rep, f, indent=2)

    fig, ax = plt.subplots(figsize=(4.5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=ds.classes, yticklabels=ds.classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); fig.tight_layout()
    fig.savefig("checkpoints/cm.png", dpi=160)
    print("Saved: checkpoints/report.json and checkpoints/cm.png")
