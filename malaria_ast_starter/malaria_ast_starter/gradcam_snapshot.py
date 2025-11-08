\
import argparse
from PIL import Image
import torch, torch.nn as nn
from torchvision import models
from cam_utils import grad_cam

def build_model(name, num_classes):
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
        raise ValueError("unsupported")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", default="cam.png")
    ap.add_argument("--model_name", default="efficientnet_b0")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model_name, args.num_classes).to(device)
    state = torch.load(args.weights, map_location=device)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state["state_dict"])
    model.eval()

    img = Image.open(args.image).convert("RGB")
    res = grad_cam(model, img, img_size=args.image_size, device=device)
    Image.fromarray((res["overlay"]*255).astype("uint8")).save(args.out)
    print(f"Saved Grad-CAM overlay -> {args.out}")
