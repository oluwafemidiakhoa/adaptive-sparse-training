# ImageNet-100 Troubleshooting - Quick Reference

## Common Errors and Instant Fixes

### 🔴 Error: `CUDA out of memory`

**Message**:
```
RuntimeError: CUDA out of memory. Tried to allocate 1.56 GiB
```

**Fix** (30 seconds):
```python
# In Config class, change:
batch_size = 16  # Was 64, now 16
```

**Why**: ResNet50 + 224×224 images = large memory footprint

**If still fails**: Try `batch_size = 8`

---

### 🔴 Error: `FileNotFoundError: [Errno 2] No such file`

**Message**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/imagenet100'
```

**Fix** (1 minute):
```python
# Step 1: Check actual path
!ls /kaggle/input/

# Step 2: Update Config.data_dir to match
# Example: If you see 'imagenet-100-dataset'
data_dir = "/kaggle/input/imagenet-100-dataset"
```

**Why**: Dataset name on Kaggle varies by uploader

---

### 🔴 Error: `GPU quota exceeded`

**Message**:
```
You have used your weekly GPU quota (30 hours)
```

**Fix Options**:

**A. Wait** (Free):
- Quota resets every Monday
- Check remaining: Settings → GPU quota

**B. Use Colab** (Free):
- Go to https://colab.research.google.com
- Same code works (minor path adjustments)
- 12-hour sessions

**C. Second Account** (Allowed):
- Kaggle allows multiple accounts
- Use different email
- Another 30 hours/week

---

### 🔴 Error: `Loaded 0 images from train split`

**Message**:
```
Loaded 0 images from train split
Found 0 classes
```

**Fix** (2 minutes):
```python
# Debug dataset structure
!ls /kaggle/input/imagenet100/
# Should show: train/ and val/

!ls /kaggle/input/imagenet100/train/ | head -10
# Should show: n01440764/ n01443537/ ... (class folders)

!ls /kaggle/input/imagenet100/train/n01440764/ | head -5
# Should show: *.JPEG files

# If structure is different, adjust ImageNet100Dataset class
```

**Common issues**:
- No train/val split → Add split parameter handling
- Different file extension → Change `.JPEG` to `.jpg` or `.png`
- Flat structure → Different loading logic needed

---

### 🟡 Warning: Low accuracy (< 40% after epoch 1)

**Expected**: 40-45% after epoch 1, 75-80% after epoch 40

**If < 30% after epoch 1**:

**Check 1**: Dataset loading correctly?
```python
# Verify class count
print(f"Classes found: {len(train_dataset.class_to_idx)}")
# Should be 100
```

**Check 2**: Model initialized?
```python
# Verify pretrained weights loaded
print(f"Model: {model}")
# Should show ResNet50 architecture
```

**Check 3**: Loss decreasing?
```
Epoch 1: Loss 3.2 → Good
Epoch 2: Loss 2.8 → Good (decreasing)
Epoch 1: Loss 0.0 → BAD (not training)
```

---

### 🟡 Warning: Activation rate stuck (e.g., 50% instead of 10%)

**Expected**: Activation rate should converge to 9-12% by epoch 5-10

**If stuck at 30-50%**:

**Check 1**: PI controller error sign
```python
# In SundewAlgorithm.select_samples(), verify:
error = self.activation_rate_ema - self.target_activation_rate
# NOT: target - activation (would be backwards!)
```

**Check 2**: Threshold updating?
```python
# Add debug print in select_samples():
print(f"Threshold: {self.activation_threshold:.3f}, Act: {current_activation_rate:.3f}")
# Should see threshold changing each batch
```

**Fix**: Increase PI gains (rare, but possible)
```python
adapt_kp = 0.0025  # Was 0.0015
adapt_ki = 0.0001  # Was 0.00005
```

---

### 🟡 Warning: Energy savings 0%

**Message**: Energy savings showing 0% despite activation rate at 10%

**Cause**: Energy tracking logic broken

**Fix**:
```python
# In SundewAlgorithm, check energy computation:
actual_energy = (
    num_active * self.energy_per_activation +
    (batch_size - num_active) * self.energy_per_skip  # Was missing!
)
```

**Should show**:
```
Act: 10% | Save: 90%  ✅ Correct
Act: 10% | Save: 0%   ❌ Broken
```

---

### 🔴 Error: `Session timed out`

**Message**: Kaggle session auto-terminated after 9 hours

**Impact**:
- Training stopped mid-run
- Partial results lost (unless saved checkpoint)

**Prevention**:
```python
# Add after each epoch in training loop:
if epoch % 5 == 0:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, f'checkpoint_epoch_{epoch}.pth')
```

**Recovery**:
- Best model saved automatically (`best_model_imagenet100.pth`)
- Restart notebook and continue from checkpoint (optional)
- Or just restart from scratch (only 4-6 hours total)

---

## Performance Issues

### Training very slow (> 10 hours for 40 epochs)

**Expected**: 4-6 hours on Kaggle T4 GPU

**If slower**:

**Check 1**: GPU enabled?
```python
!nvidia-smi
# Should show T4 GPU with memory usage
```

**Check 2**: Data loading bottleneck?
```python
# In Config, increase workers:
num_workers = 4  # Was 2
pin_memory = True
```

**Check 3**: Too many debug prints?
```python
# Remove frequent print statements
# Print only every 50-100 batches
```

---

## Validation Issues

### Validation accuracy not improving

**Expected progression**:
```
Epoch 1:  40% → Epoch 10: 55% → Epoch 20: 65% → Epoch 40: 77%
```

**If stuck**:
```
Epoch 1:  40% → Epoch 10: 42% → Epoch 20: 43% → Plateau!
```

**Possible causes**:

**A. Learning rate too low**:
```python
learning_rate = 0.005  # Increase from 0.001
```

**B. Overfitting** (train acc high, val acc low):
```python
weight_decay = 5e-4  # Increase from 1e-4
# Add dropout to model (advanced)
```

**C. Underfitting** (both low):
```python
num_epochs = 60  # Increase from 40
# Or use larger model (ResNet101)
```

---

## Dataset-Specific Issues

### Using different ImageNet-100 dataset

**If your dataset has different structure**:

**Example: Images in single folder with labels.txt**
```python
class ImageNet100Dataset(Dataset):
    def __init__(self, root_dir, split='train'):
        # Load labels from CSV/text file
        labels_file = Path(root_dir) / f"{split}_labels.txt"
        with open(labels_file) as f:
            self.samples = [line.strip().split() for line in f]
```

**Example: HDF5 format**
```python
import h5py
class ImageNet100Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.images = self.h5_file['images']
        self.labels = self.h5_file['labels']
```

**Tip**: Check dataset description on Kaggle for structure details

---

## Quick Debug Script

**Run this first if anything broken**:

```python
# Cell 1: Environment check
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Cell 2: Dataset check
from pathlib import Path
data_root = Path("/kaggle/input")
print("Available datasets:")
for path in data_root.iterdir():
    print(f"  - {path.name}")

# Cell 3: ImageNet-100 structure check
imagenet_path = data_root / "imagenet100"  # Adjust name
if imagenet_path.exists():
    print(f"\nImageNet-100 structure:")
    print(f"  Train: {len(list((imagenet_path / 'train').iterdir()))} classes")
    print(f"  Val: {len(list((imagenet_path / 'val').iterdir()))} classes")

    # Count images in first class
    first_class = next((imagenet_path / 'train').iterdir())
    num_images = len(list(first_class.glob("*.JPEG")))
    print(f"  Sample class: {first_class.name} ({num_images} images)")
else:
    print(f"ERROR: {imagenet_path} not found!")

# Cell 4: Model check
from torchvision.models import resnet50, ResNet50_Weights
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
print(f"\nModel loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")
```

**Expected output**:
```
PyTorch: 2.0.1
CUDA: True
GPU: Tesla T4
Memory: 15.0 GB

Available datasets:
  - imagenet100

ImageNet-100 structure:
  Train: 100 classes
  Val: 100 classes
  Sample class: n01440764 (1300 images)

Model loaded: 23.5M params
```

---

## When to Ask for Help

**Try yourself first** (15-30 min debugging):
- Check error message carefully
- Try fixes in this guide
- Google the exact error message
- Check Kaggle dataset comments

**Ask for help if**:
- Error persists after trying fixes
- Unclear what's wrong
- Results significantly different from expected (e.g., 20% accuracy instead of 75%)

**Where to ask**:
1. **GitHub Issues**: [Your repo](https://github.com/oluwafemidiakhoa/adaptive-sparse-training/issues)
2. **Kaggle Comments**: On the ImageNet-100 dataset page
3. **Reddit**: r/learnmachinelearning (helpful community)

**What to include**:
- Error message (full traceback)
- What you tried
- Relevant code snippet
- Environment info (GPU, PyTorch version)

---

## Success Checklist

After training completes, verify:

- [ ] No errors during training
- [ ] All 40 epochs completed
- [ ] Validation accuracy ≥ 70% (minimum) or ≥ 75% (target)
- [ ] Energy savings 85-92%
- [ ] Activation rate 8-13%
- [ ] Best model saved (`best_model_imagenet100.pth` file exists)
- [ ] Training time < 8 hours

**If all checked**: Success! 🎉 Update GitHub and write blog post.

**If some failed**: Review errors, try fixes, or post for help.

---

## Emergency Contacts

**Not working at all? Try this**:

1. **Simplest possible test**:
```python
# Just load dataset and print
from pathlib import Path
data_path = Path("/kaggle/input/imagenet100/train")
print(f"Classes: {len(list(data_path.iterdir()))}")
# Should print: Classes: 100
```

2. **Just train baseline ResNet50** (no AST):
```python
# Remove all Sundew/AST code
# Just train ResNet50 normally
# If this works → AST code has bug
# If this fails → dataset/setup issue
```

3. **Copy working example**:
- Search Kaggle for "ImageNet-100 ResNet"
- Find working notebook
- Compare to your code
- Identify difference

---

**Still stuck?** Open a GitHub issue with:
- Title: "ImageNet-100 setup help needed"
- Full error message
- Environment details
- What you tried

Community will help! 🤝
