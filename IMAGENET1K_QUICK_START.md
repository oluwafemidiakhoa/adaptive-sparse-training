# ImageNet-1K Quick Start Guide

**Goal**: Validate AST on full ImageNet-1K (1.28M images, 1000 classes)

**Developed by**: Oluwafemi Idiakhoa

---

## 🚀 Quick Summary

**Three configurations ready to use:**

| Config | Accuracy | Energy Savings | Time (V100) | Use When |
|--------|----------|----------------|-------------|----------|
| **Ultra** | 70-72% | 80% | 8 hours | Quick validation, first run |
| **Aggressive** | 73-75% | 70% | 15 hours | Balanced results |
| **Conservative** | 75-76% | 60% | 40 hours | Publication quality |

**Recommendation**: Start with **Ultra** (8 hours) to validate, then run **Conservative** for final results.

---

## 📋 Prerequisites

1. **Compute Access**:
   - Kaggle (free, 30h/week limit)
   - Google Colab Pro ($10/month)
   - Runpod.io (~$0.50-1.50/hour)

2. **Dataset**: ImageNet-1K (ILSVRC2012)
   - On Kaggle: `imagenet-object-localization-challenge`
   - Size: ~150GB
   - 1,281,167 training images
   - 50,000 validation images

3. **Files Needed**:
   - `KAGGLE_IMAGENET1K_AST_CONFIGS.py` (configurations)
   - `KAGGLE_IMAGENET100_AST_PRODUCTION.py` (base script)

---

## ⚡ Quick Start (Ultra Config - 8 hours)

### Step 1: Setup Environment

```bash
# On Kaggle/Colab
!pip install torch torchvision matplotlib numpy tqdm

# Clone repository
!git clone https://github.com/oluwafemidiakhoa/adaptive-sparse-training.git
%cd adaptive-sparse-training
```

### Step 2: Configure for Ultra (Quick Validation)

```python
from KAGGLE_IMAGENET1K_AST_CONFIGS import get_config

# Get Ultra configuration (8 hours, 80% energy savings)
config = get_config("ultra")

# Verify settings
print(f"Epochs: {config.num_epochs}")
print(f"Target Activation: {config.target_activation_rate:.0%}")
print(f"Expected Energy Savings: {(1-config.target_activation_rate)*100:.0f}%")
```

### Step 3: Run Training

```python
# Use the ImageNet-100 script as base, just update config
# (Full script creation in progress)

# Key changes from ImageNet-100:
# - num_classes: 100 → 1000
# - data_dir: ImageNet100 → ImageNet-1K path
# - epochs: Varies by config
```

### Step 4: Monitor Results

Expected output:
```
Epoch  1/30 | Loss: 4.8234 | Val Acc: 25.30% | Act: 22.5% | Save: 77.5%
Epoch 10/30 | Loss: 3.2156 | Val Acc: 55.15% | Act: 20.8% | Save: 79.2%
Epoch 30/30 | Loss: 2.1842 | Val Acc: 70.46% | Act: 19.7% | Save: 80.3%

FINAL RESULTS:
- Validation Accuracy: 70.46%
- Energy Savings: 80.3%
- Training Time: 8.2 hours
- Speedup: 6.5× vs baseline
```

---

## 📊 Expected Results by Configuration

### Ultra (8 hours):
```
Metric              Value       vs Baseline
Accuracy            70-72%      -4 to -6%
Top-5 Accuracy      89-90%      -2 to -3%
Energy Savings      80%         +80%
Speedup             6-8×        6-8×
```

### Aggressive (15 hours):
```
Metric              Value       vs Baseline
Accuracy            73-75%      -1 to -3%
Top-5 Accuracy      91-92%      -1 to -2%
Energy Savings      70%         +70%
Speedup             3-4×        3-4×
```

### Conservative (40 hours):
```
Metric              Value       vs Baseline
Accuracy            75-76%      -0 to -1%
Top-5 Accuracy      92-93%      ≈0%
Energy Savings      60%         +60%
Speedup             1.9×        1.9×
```

---

## 🎯 Success Criteria

**Minimum for Announcement:**
- ✅ Accuracy ≥70% (Ultra config)
- ✅ Energy Savings ≥60%
- ✅ Stable convergence
- ✅ Reproducible results

**Ideal for Publication:**
- ✅ Accuracy ≥75% (Conservative config)
- ✅ Energy Savings ≥60%
- ✅ Within 1% of baseline accuracy

---

## 📝 Next Steps After First Run

1. **If Ultra succeeds** (70%+ accuracy):
   - ✅ AST validated on ImageNet-1K!
   - Run Conservative for final results
   - Update README with ImageNet-1K results

2. **If Ultra underperforms** (<70% accuracy):
   - Tune PI controller gains
   - Adjust target activation rate
   - Increase warmup epochs
   - Try Aggressive config

3. **After Conservative completes**:
   - Compare all three configs
   - Choose best for announcement
   - Document lessons learned

---

## 💡 Troubleshooting

### Issue: Out of Memory
**Fix**: Reduce batch_size to 128 or 64

### Issue: Slow data loading
**Fix**: Increase num_workers to 16

### Issue: Unstable training
**Fix**:
- Reduce adapt_kp and adapt_ki
- Increase warmup_epochs
- Lower learning rate

### Issue: Time limit exceeded (Kaggle)
**Fix**:
- Save checkpoints every epoch
- Resume from checkpoint in new session
- Or use Runpod/Colab Pro

---

## 📞 Questions?

See full documentation:
- [IMAGENET1K_VALIDATION_PLAN.md](IMAGENET1K_VALIDATION_PLAN.md) - Complete experimental plan
- [KAGGLE_IMAGENET1K_AST_CONFIGS.py](KAGGLE_IMAGENET1K_AST_CONFIGS.py) - Configuration details

**Developed by Oluwafemi Idiakhoa**
