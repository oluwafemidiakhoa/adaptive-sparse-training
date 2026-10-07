# ImageNet-100 Quick Start - 1 Hour to Results

> **Evidence status:** ImageNet-100 results are reported, with verification pending. Numerical examples and projections are not measurements. Energy savings are modeled and speedups are estimated. See [IMAGENET100_STATUS.md](IMAGENET100_STATUS.md) for evidence requirements.


## Pre-Flight Checklist (5 minutes)

- [ ] Kaggle account created and verified
- [ ] Phone number verified (required for GPU access)
- [ ] ImageNet-100 dataset found on Kaggle
- [ ] KAGGLE_IMAGENET100_AST.py code reviewed

## Step-by-Step Execution (30 minutes)

### 1. Create Kaggle Notebook (2 min)
```
1. Go to https://www.kaggle.com
2. Click "Create" → "New Notebook"
3. Title: "AST ImageNet-100 - Quick Test"
```

### 2. Enable GPU (1 min)
```
Settings (top right) → Accelerator → GPU T4 x2
```

### 3. Add ImageNet-100 Dataset (2 min)
```
1. Click "+ Add data" (right sidebar)
2. Search: "imagenet100"
3. Select: "ImageNet-100" by ambityga (or similar)
4. Click "Add"
5. Verify path: /kaggle/input/imagenet100
```

### 4. Install Dependencies (1 min)
**Cell 1:**
```python
# Verify GPU
!nvidia-smi

# Check PyTorch
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 5. Quick Test Run (5 min)
**Cell 2: Copy entire KAGGLE_IMAGENET100_AST.py EXCEPT change these lines:**

```python
class Config:
    # ... other settings ...
    num_epochs = 1  # QUICK TEST - just 1 epoch
    batch_size = 32  # Smaller batch to avoid OOM
```

### 6. Run (Shift+Enter)

Expected output:
```
================================================================================
Adaptive Sparse Training (AST) - ImageNet-100 Validation
================================================================================

Device: cuda
Batch size: 32
Target activation rate: 10.0%

Loading ImageNet-100 dataset...
Loaded 130000 images from train split
Found 100 classes
Loaded 5000 images from val split
Found 100 classes

Initializing ResNet50...
Model: ResNet50 (23.5M params)

Starting training...
----------------------------------------------------------------------
  Batch 50/4063 | Loss: 3.8234 | Act: 12.3% | Thr: 0.485
  ...
Epoch  1/1 | Loss: 3.2145 | Val Acc: 42.30% | Act:  9.7% | Save: 90.3% | Time: 312.5s

================================================================================
FINAL RESULTS
================================================================================
Best Validation Accuracy: 42.30%
Modeled Energy Savings: 90.3%
Average Activation Rate: 9.7%
Total Training Time: 5.2 minutes
```

### 7. Verify Success (2 min)

✅ **Green Flags** (Everything working):
- [ ] No errors in output
- [ ] Validation accuracy > 35% (reasonable for 1 epoch)
- [ ] Modeled energy savings 88-92%
- [ ] Activation rate 9-12%
- [ ] Training completed in 5-10 minutes

❌ **Red Flags** (Need to fix):
- [ ] `RuntimeError: CUDA out of memory` → Reduce batch_size to 16
- [ ] `FileNotFoundError: [Errno 2] No such file` → Check data_dir path
- [ ] Accuracy < 20% → Dataset loading issue
- [ ] Loss = 0.0 → Model not training

## Full Run (4-6 hours)

If quick test successful, run full experiment:

### Cell 3: Full Training
```python
class Config:
    # ... other settings ...
    num_epochs = 40  # Full training
    batch_size = 64  # Larger batch (if GPU memory allows)
```

Click **Run** and leave running (Kaggle allows 9-hour sessions).

**Monitor progress**: Check back every hour to see epoch results.

## Results Analysis

After full training completes:

### Download Results
```python
# Cell 4: Save results
import json

results = {
    'evidence_status': 'template_not_a_run',
    'best_accuracy': None,  # Populate from a completed run
    'modeled_energy_savings': None,
    'last_epoch_activation_rate': None,
    'training_time_minutes': None,
    'estimated_speedup': None
}

with open('imagenet100_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
```

Download from Kaggle:
1. Click "Data" tab (right sidebar)
2. Click "Output" (if not visible, refresh page)
3. Download `imagenet100_results.json` and `best_model_imagenet100.pth`

### Compare to CIFAR-10
| Metric | CIFAR-10 | ImageNet-100 | Change |
|--------|----------|--------------|--------|
| Accuracy | 61.2% | ??% | ? |
| Modeled Energy Savings | 89.6% | ??% | ? |
| Activation Rate | 10.4% | ??% | ? |
| Estimated Speedup | 11.5× | ??× | ? |

## Proposed Success Criteria (not completed evidence)

### Minimum Success (Ready to share)
- [ ] Accuracy ≥ 70%
- [ ] Modeled energy savings ≥ 85%
- [ ] Activation rate 8-13%
- [ ] No training failures

### Target Success (Update GitHub)
- [ ] Accuracy ≥ 75%
- [ ] Modeled energy savings ≥ 88%
- [ ] Activation rate 9-12%
- [ ] Estimated speedup ≥ 8×

### Excellent Success (Write paper!)
- [ ] Accuracy ≥ 78%
- [ ] Modeled energy savings ≥ 90%
- [ ] Activation rate 9.5-10.5%
- [ ] Estimated speedup ≥ 10×

## Troubleshooting

### Issue: GPU quota exceeded
**Error**: "You have used your weekly GPU quota"
**Solution**:
- Kaggle free tier: 30 hours/week
- Check usage: Settings → GPU quota
- Reset: Every Monday
- **Workaround**: Create second Kaggle account (allowed) OR wait until Monday

### Issue: Session timeout
**Error**: "Session has been terminated"
**Solution**:
- Kaggle auto-saves progress
- Restart notebook
- Model checkpoint saved in `best_model_imagenet100.pth`
- Training will continue if you implement checkpoint loading (optional)

### Issue: Low accuracy (< 60% after 40 epochs)
**Debugging steps**:
```python
# Cell: Debug dataset
!ls /kaggle/input/
!ls /kaggle/input/imagenet100/
!ls /kaggle/input/imagenet100/train/ | head -10

# Check first batch
from PIL import Image
import matplotlib.pyplot as plt

train_dataset = ImageNet100Dataset("/kaggle/input/imagenet100", split='train')
print(f"Dataset size: {len(train_dataset)}")
print(f"Classes: {len(train_dataset.class_to_idx)}")

# Visualize samples
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    img, label = train_dataset[i]
    ax.imshow(img)
    ax.set_title(f"Class {label}")
    ax.axis('off')
plt.show()
```

## Publication after evidence review

Use the safe publication wording in [IMAGENET100_STATUS.md](IMAGENET100_STATUS.md). Link actual run artifacts and label modeled and estimated metrics explicitly. Do not publish illustrative values as validated results or claim no retuning without matched evidence.

Suggested article title: "Adaptive Sparse Training on ImageNet-100: Reported Results and Verification Gaps".

## Timeline Summary

| Phase | Time | Cumulative |
|-------|------|------------|
| Setup Kaggle | 5 min | 5 min |
| Quick test (1 epoch) | 10 min | 15 min |
| Full training (40 epochs) | 4-6 hours | ~6 hours |
| Analysis & writeup | 2 hours | ~8 hours |
| **Total** | **~1 working day** | **Done!** |

## Ready? Go! 🚀

1. Open https://www.kaggle.com
2. Create notebook
3. Copy KAGGLE_IMAGENET100_AST.py
4. Change num_epochs=1 for quick test
5. Run and verify
6. If successful, run full 40 epochs
7. Share results!

**Good luck!** You already validated the concept on CIFAR-10. This is just scaling up. 💪
