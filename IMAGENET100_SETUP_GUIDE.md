# ImageNet-100 Experiment Setup Guide

## Quick Start (30 minutes)

### Step 1: Create Kaggle Account (5 min)
1. Go to https://www.kaggle.com
2. Sign up with Google/GitHub
3. Verify email
4. Go to Account settings → Phone verification (required for GPU access)

### Step 2: Find ImageNet-100 Dataset (5 min)

**Option A: Use existing dataset**
1. Search "imagenet100" in Kaggle Datasets
2. Click "Add Data" on dataset page
3. Common datasets:
   - https://www.kaggle.com/datasets/ambityga/imagenet100
   - https://www.kaggle.com/datasets/ifigotin/imagenet100-224

**Option B: Upload custom dataset**
1. Download ImageNet-100 from https://www.image-net.org
2. Create new Kaggle dataset
3. Upload train/val folders

### Step 3: Create Kaggle Notebook (5 min)
1. Click "Create" → "New Notebook"
2. Title: "AST ImageNet-100 Validation"
3. Settings (top right):
   - Accelerator: **GPU T4 x2** (free tier)
   - Internet: **On** (for pip installs)
   - Persistence: **Files only**

### Step 4: Add Dataset to Notebook (2 min)
1. Click "+ Add data" (right panel)
2. Search for your ImageNet-100 dataset
3. Click "Add"
4. Note the path: `/kaggle/input/imagenet100/` (or similar)

### Step 5: Copy AST Code (5 min)
1. Copy entire `KAGGLE_IMAGENET100_AST.py` content
2. Paste into notebook cell
3. Update `Config.data_dir` to match your dataset path:
   ```python
   data_dir = "/kaggle/input/imagenet100"  # Adjust if different
   ```

### Step 6: Run Experiment (5 min setup + 4-6 hours training)

**Quick test run (1 epoch):**
```python
# In the Config class, change:
num_epochs = 1  # Quick test
batch_size = 32  # Smaller if GPU OOM
```

**Full experiment:**
```python
# Use default settings:
num_epochs = 40
batch_size = 64
```

Click **Run All** or press `Shift+Enter` on each cell.

---

## Expected Output

### During Training
```
================================================================================
Adaptive Sparse Training (AST) - ImageNet-100 Validation
================================================================================

Device: cuda
Batch size: 64
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
  Batch 50/2031 | Loss: 3.8234 | Act: 12.3% | Thr: 0.485
  Batch 100/2031 | Loss: 3.6421 | Act: 10.8% | Thr: 0.502
  ...
Epoch  1/40 | Loss: 3.2145 | Val Acc: 42.30% | Act:  9.7% | Save: 90.3% | Time: 285.3s
Epoch  2/40 | Loss: 2.8934 | Val Acc: 51.20% | Act: 10.2% | Save: 89.8% | Time: 278.1s
...
Epoch 40/40 | Loss: 1.8456 | Val Acc: 76.50% | Act: 10.1% | Save: 89.9% | Time: 276.5s
----------------------------------------------------------------------

================================================================================
FINAL RESULTS
================================================================================
Best Validation Accuracy: 77.20%
Total Energy Savings: 89.9%
Average Activation Rate: 10.05%
Total Training Time: 184.2 minutes

Estimated Baseline Time: 1832.6 minutes
Training Speedup: 9.9×
================================================================================
```

### Success Criteria
✅ **Validation Accuracy**: 75-80% (competitive with baseline ResNet50)
✅ **Energy Savings**: 88-91% (near 90% target)
✅ **Activation Rate**: 9-12% (converged to 10% target)
✅ **Training Time**: 3-5 hours (vs 30-50 hours baseline estimate)

---

## Troubleshooting

### Error: "GPU quota exceeded"
**Solution**:
- Kaggle free tier: 30 hours GPU/week
- Check usage: Account → Settings → GPU quota
- Wait for weekly reset (Monday)

### Error: "Out of memory"
**Solution**: Reduce batch size
```python
batch_size = 32  # Or even 16 if still failing
```

### Error: "Dataset not found"
**Solution**: Check dataset path
```python
# List available data
!ls /kaggle/input/
# Update data_dir in Config class
data_dir = "/kaggle/input/YOUR_DATASET_NAME"
```

### Error: "No images found"
**Solution**: Check dataset structure
```python
!ls /kaggle/input/imagenet100/
# Should show: train/ and val/ folders
!ls /kaggle/input/imagenet100/train/ | head -5
# Should show class folders: n01440764, n01443537, etc.
```

### Low accuracy (< 60%)
**Possible causes**:
1. Dataset path wrong → loading wrong data
2. Class count mismatch → check `len(class_to_idx)`
3. Not enough epochs → try 40-60 epochs

### Activation stuck at wrong rate (e.g., 30% instead of 10%)
**Solution**:
- This is rare on ImageNet (worked on CIFAR-10)
- If happens, adjust PI gains:
```python
adapt_kp = 0.0025  # Increase for faster convergence
adapt_ki = 0.0001
```

---

## Validation Checklist

Before claiming ImageNet-100 results, verify:

- [ ] Dataset loaded correctly (130K train, 5K val images)
- [ ] Model trains without errors for full 40 epochs
- [ ] Validation accuracy > 75% (competitive with baseline)
- [ ] Energy savings 88-91%
- [ ] Activation rate converges to 9-12%
- [ ] Training completes in < 5 hours
- [ ] Best model saved successfully

---

## Next Steps After ImageNet-100

### If Results Good (Accuracy ≥ 75%, Savings ≥ 88%)
1. **Publish Results**:
   - Update GitHub README with ImageNet-100 results
   - Write Medium "Part 2" article
   - Post to r/MachineLearning (you'll have karma now!)

2. **Scale to Full ImageNet**:
   - Use saved model as checkpoint
   - Budget: $50-100 for cloud GPU
   - Expected: 70-75% top-1 accuracy with 90% savings

### If Results Need Tuning
1. **Adjust PI Controller**:
   - If activation rate not converging, tune Kp/Ki
   - Try longer EMA (α=0.2 instead of 0.3)

2. **Improve Significance Scoring**:
   - Add gradient magnitude factor
   - Add prediction confidence factor
   - Experiment with weights (currently 70% loss, 30% intensity)

3. **Try Different Architectures**:
   - EfficientNet-B0 (smaller, faster)
   - ViT-Small (transformer-based)

---

## Cost Estimate

### Free Tier (Kaggle)
- **GPU**: 30 hours/week (T4 x2)
- **Cost**: $0
- **Limitations**: 9-hour session limit, need to restart
- **ImageNet-100**: 2-3 runs per week

### Paid Options (if need more)
- **Colab Pro**: $10/month, better GPUs (A100)
- **Lambda Labs**: $0.60/hour for A100
- **Runpod**: $0.44/hour for A100
- **Full ImageNet run**: ~$15-30 on paid GPU

---

## Expected Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Setup Kaggle | 30 min | Account + notebook ready |
| Test run (1 epoch) | 15 min | Verify code works |
| Full ImageNet-100 (40 epochs) | 4-6 hours | 75-80% accuracy results |
| Analysis & writeup | 2 hours | Updated README, blog post |
| **Total** | **1 day** | **Validated ImageNet results** |

---

## Success Metrics

### Minimum Viable Result
- ✅ 70% accuracy (shows concept works)
- ✅ 85% energy savings (significant efficiency)
- ✅ Completes without errors

### Target Result
- ✅ 75% accuracy (competitive)
- ✅ 89% energy savings (matches CIFAR-10)
- ✅ 10% activation rate (perfect convergence)

### Excellent Result
- ✅ 78%+ accuracy (beats baseline)
- ✅ 90%+ energy savings (exceeds goal)
- ✅ Stable training (no oscillation/failures)

---

## Support

**If stuck:**
1. Check Kaggle notebook logs for error messages
2. Verify dataset path with `!ls` commands
3. Try reducing batch_size if OOM errors
4. Post issue on GitHub: github.com/oluwafemidiakhoa/adaptive-sparse-training

**Community:**
- Kaggle Discussions: Ask in dataset comments
- Reddit: r/learnmachinelearning (friendly community)
- GitHub Issues: Technical problems

---

## Files Created

1. **KAGGLE_IMAGENET100_AST.py** - Full training code (570 lines)
2. **IMAGENET100_SETUP_GUIDE.md** - This file
3. **IMAGENET_VALIDATION_PLAN.md** - Comprehensive research plan

**Ready to start!** Open Kaggle and follow Step 1. 🚀
