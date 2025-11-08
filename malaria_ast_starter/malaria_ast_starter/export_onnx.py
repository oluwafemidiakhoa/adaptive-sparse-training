\
import argparse
import torch, torch.nn as nn
from torchvision import models

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
    ap.add_argument("--model_name", default="efficientnet_b0")
    ap.add_argument("--num_classes", type=int, default=2)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--precision", choices=["fp32","fp16"], default="fp32")
    ap.add_argument("--out", default="model.onnx")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = build_model(args.model_name, args.num_classes).to(device)
    state = torch.load(args.weights, map_location=device)
    try:
        m.load_state_dict(state)
    except Exception:
        m.load_state_dict(state["state_dict"])
    m.eval()

    x = torch.randn(1,3,args.image_size,args.image_size, device=device)
    if args.precision == "fp16":
        m = m.half(); x = x.half()

    torch.onnx.export(
        m, x, args.out,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input":{0:"batch"}, "output":{0:"batch"}},
        opset_version=17, do_constant_folding=True
    )
    print(f"Saved ONNX -> {args.out}")
