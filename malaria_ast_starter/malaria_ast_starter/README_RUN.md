\
# Malaria Training Starter — Quickstart

## 1) Install
```bash
pip install -r requirements.txt
```

## 2) Data layout
```
data/
  train/
    Parasitized/  *.png|*.jpg
    Uninfected/   *.png|*.jpg
  val/
    Parasitized/  ...
    Uninfected/   ...
```

## 3) Train (resumable)
```bash
python train.py --config configs/config.yaml
# or explicitly resume:
python train_resume.py --config configs/config.yaml
```

## 4) Evaluate
```bash
python eval.py --weights checkpoints/best.pt
```

## 5) Grad-CAM snapshot
```bash
python gradcam_snapshot.py --weights checkpoints/best.pt \
  --image data/val/Parasitized/<somefile>.png --out cam_sample.png
```

## 6) Export ONNX
```bash
python export_onnx.py --weights checkpoints/best.pt --precision fp16 --out model_fp16.onnx
```
