# 🚀 Google Colab Quick Start Guide

## ⚡ Super Fast Setup (3 Steps)

### 1. Open Colab Notebook

**Option A: Upload notebook directly**
1. Go to https://colab.research.google.com
2. File → Upload notebook
3. Upload `Malaria_AST_Training.ipynb`

**Option B: Open from GitHub**
1. Go to https://colab.research.google.com
2. File → Open notebook → GitHub tab
3. Enter your repo URL
4. Select `Malaria_AST_Training.ipynb`

### 2. Enable GPU

1. Runtime → Change runtime type
2. Hardware accelerator → **GPU**
3. GPU type → **T4** (free) or **V100/A100** (Colab Pro)
4. Save

### 3. Run All Cells

1. Runtime → Run all
2. Upload `kaggle.json` when prompted
3. ☕ Wait 25-40 minutes
4. 🎉 Done!

---

## 📋 What the Notebook Does

| Step | Action | Time | Output |
|------|--------|------|--------|
| 1 | Clone repo / upload files | 30s | Code ready |
| 2 | Setup Kaggle API | 10s | API configured |
| 3 | Download dataset + setup | 3-5 min | 27k images organized |
| 4 | Train with AST | 20-35 min | Trained model (95%+ acc) |
| 5 | Generate visualizations | 30s | Publication graphics |
| 6 | Evaluate model | 1 min | Classification report |
| 7 | Create Grad-CAM | 30s | Interpretable AI |
| 8 | Save to Drive | 1 min | Permanent backup |

**Total**: ~25-40 minutes (mostly training)

---

## 🎛️ GPU Recommendations

### Free Tier (T4)
- **Batch size**: 64
- **Training time**: 25-30 min
- **Memory**: 16GB
- ✅ Plenty for this project!

### Colab Pro (V100)
- **Batch size**: 128-256
- **Training time**: 15-20 min
- **Memory**: 16GB
- ⚡ 30% faster

### Colab Pro+ (A100)
- **Batch size**: 256+
- **Training time**: 10-15 min
- **Memory**: 40GB
- 🚀 2x faster than T4

**Recommendation**: Free T4 is perfectly fine! V100/A100 only saves ~10 minutes.

---

## 📂 File Upload Methods

### Option 1: Git Clone (Recommended)
```python
!git clone https://github.com/YOUR_USERNAME/malaria-ast-trainer.git
%cd malaria-ast-trainer/malaria_ast_starter
```
✅ Fast, clean
❌ Requires GitHub repo

### Option 2: Manual Upload
1. Zip your `malaria_ast_starter/` folder
2. In Colab: Files tab (left sidebar)
3. Upload `malaria_ast_starter.zip`
4. Unzip:
```python
!unzip malaria_ast_starter.zip
%cd malaria_ast_starter
```
✅ No GitHub needed
❌ Slower for large files

### Option 3: Google Drive
1. Upload folder to Drive
2. In Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/malaria_ast_starter
```
✅ Persistent storage
❌ Slower file access

---

## 🔑 Kaggle API Setup

### Get your API token:
1. Go to https://www.kaggle.com/settings
2. Scroll to **API** section
3. Click **"Create New API Token"**
4. Save `kaggle.json` file

### Upload in Colab:
When prompted by the notebook, click "Choose Files" and upload `kaggle.json`

**OR** manual setup:
```python
from google.colab import files
uploaded = files.upload()  # Upload kaggle.json

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## 💾 Saving Your Results

### Option 1: Google Drive (Recommended)
```python
# Already in notebook Step 8
!cp -r checkpoints_ast /content/drive/MyDrive/malaria_ast_results/
!cp -r visualizations /content/drive/MyDrive/malaria_ast_results/
```
✅ Permanent storage
✅ Survives Colab disconnects

### Option 2: Download Directly
```python
from google.colab import files

# Download best model
files.download('checkpoints_ast/best.pt')

# Download visualizations
files.download('visualizations/ast_headline.png')
files.download('visualizations/ast_results.png')
```
✅ Immediate download
❌ Can be slow for large files

### Option 3: Zip and Download
```python
!zip -r results.zip checkpoints_ast visualizations gradcam_*.png checkpoints/report.json

from google.colab import files
files.download('results.zip')
```
✅ Single download
✅ Fast

---

## 🎯 Expected Results

### With Default Config (40% activation):

**Accuracy Metrics:**
- Validation Accuracy: **95-97%**
- Precision (Parasitized): **~96%**
- Recall (Parasitized): **~95%**
- F1-Score: **~96%**

**Efficiency Metrics:**
- Energy Savings: **~60%**
- Activation Rate: **~40%**
- Samples Processed: **~11k / 27k** per epoch
- Training Time (T4): **~25 minutes**

**Files Generated:**
- `checkpoints_ast/best.pt` - Best model (85 MB)
- `visualizations/ast_results.png` - 4-panel analysis
- `visualizations/ast_headline.png` - Social media graphic
- `checkpoints/report.json` - Classification metrics
- `checkpoints/cm.png` - Confusion matrix
- `gradcam_*.png` - Interpretable visualizations

---

## 🔧 Troubleshooting

### Problem: "No GPU available"
**Solution**: Runtime → Change runtime type → GPU

### Problem: "Kaggle API error"
**Solution**:
1. Re-download `kaggle.json` from Kaggle settings
2. Make sure you uploaded the file in Step 2
3. Check file permissions: `!ls -la ~/.kaggle/`

### Problem: "Out of memory"
**Solution**: Reduce batch size in config:
```python
!python -c "
import yaml
with open('configs/config_colab.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['batch_size'] = 32  # Reduced from 64
with open('configs/config_colab.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
```

### Problem: "Runtime disconnected"
**Solution**:
- Training auto-resumes from last checkpoint!
- Just re-run the training cell:
```python
!python train_ast.py --config configs/config_colab.yaml
```

### Problem: "Slow training"
**Solution**:
1. Verify GPU is enabled: `!nvidia-smi`
2. Check num_workers in config (should be 2)
3. Ensure amp is enabled (automatic with GPU)

---

## 🎨 Customization

### Try Different Energy Savings:

**90% Savings (Maximum Buzz)**:
```python
!python -c "
import yaml
with open('configs/config_colab.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['ast_target_activation_rate'] = 0.10  # Only 10% of samples
cfg['ast_warmup_epochs'] = 5
with open('configs/config_max_buzz.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
!python train_ast.py --config configs/config_max_buzz.yaml
```

**30% Savings (Conservative)**:
```python
!python -c "
import yaml
with open('configs/config_colab.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['ast_target_activation_rate'] = 0.70  # 70% of samples
cfg['ast_warmup_epochs'] = 0
with open('configs/config_conservative.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
!python train_ast.py --config configs/config_conservative.yaml
```

### Try Different Models:

**ResNet18 (Faster)**:
```python
# Edit config
cfg['model_name'] = 'resnet18'  # Faster, slightly lower accuracy
```

**ResNet50 (More Accurate)**:
```python
# Edit config
cfg['model_name'] = 'resnet50'  # Slower, potentially higher accuracy
```

---

## 📊 Comparing with Baseline

Want to show AST vs traditional training?

```python
# 1. Train baseline (no AST) - takes ~45 minutes
!python train.py --config configs/config.yaml

# 2. Train with AST - takes ~25 minutes
!python train_ast.py --config configs/config_colab.yaml

# 3. Compare results
!python visualize_ast.py \
    --metrics checkpoints_ast/metrics_ast.jsonl \
    --baseline-metrics checkpoints/metrics.jsonl \
    --output-dir visualizations
```

This creates `visualizations/ast_vs_baseline.png` showing the difference!

---

## 🎤 Using Your Results

### For Social Media (Twitter/LinkedIn):

1. Download `visualizations/ast_headline.png`
2. Post with caption:
```
🌿 Built energy-efficient AI that detects malaria with 96% accuracy
using 60% less computational resources than traditional training.

Making medical AI accessible to clinics with limited power!

#AI #MachineLearning #GlobalHealth #Sustainability
```

### For Blog Posts:

1. Use `visualizations/ast_results.png` as main graphic
2. Include Grad-CAM visualizations to show interpretability
3. Quote metrics from classification report
4. Link to GitHub repo

### For Presentations:

1. Start with headline graphic (impact)
2. Show 4-panel results (comprehensive)
3. Demo Grad-CAM (interpretability)
4. End with confusion matrix (validation)

---

## ⏱️ Time Budget

**Minimum (Demo only)**: 5 minutes
- Run `demo_ast.py` with synthetic data

**Quick (Pre-downloaded dataset)**: 20 minutes
- Skip download, just train

**Full (First time)**: 35 minutes
- Download + setup + train + visualize

**Complete (With baseline comparison)**: 70 minutes
- Baseline + AST + comparison

**Research (Multiple configs)**: 2-3 hours
- Try 90%, 60%, 30% savings
- Compare models (ResNet18/50, EfficientNet)

---

## 💡 Pro Tips

1. **Start with Free T4**: It's plenty fast for this project
2. **Save to Drive early**: Colab can disconnect anytime
3. **Use the notebook**: All steps automated
4. **Monitor GPU usage**: `!nvidia-smi` to check utilization
5. **Download results immediately**: Before notebook disconnects
6. **Keep kaggle.json safe**: You'll need it for future sessions
7. **Try max buzz config**: 90% savings makes great headlines!

---

## 📞 Need Help?

**Common Issues**:
- Check Troubleshooting section above
- Review `GETTING_STARTED.md` for detailed setup
- See `CLAUDE.md` for technical details

**For Questions**:
- Open GitHub issue
- Check documentation files
- Review notebook comments

---

**Ready to start? Open the notebook and run all cells! 🚀**

**Total time commitment: ~30 minutes**

**Output: Publication-ready energy-efficient malaria detector! 🎉**
