# 🚀 Getting Started with Energy-Efficient Malaria Detection AI

This guide will help you set up and run the AST-powered malaria classification system from scratch.

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended but not required)
- 8GB+ RAM
- 10GB+ free disk space (for dataset)

## 🎯 Quick Start (5 minutes)

### Option 1: Demo with Synthetic Data

Try AST immediately with synthetic data:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the demo (creates synthetic data and trains)
python demo_ast.py
```

This will:
- Create a small synthetic dataset
- Train a model for 5 epochs with AST
- Generate visualizations
- Show energy savings

**Note**: This is just a demo! For real malaria detection, use the NIH dataset (see Option 2).

### Option 2: Full Setup with Real Data

For actual malaria classification:

#### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Download the Dataset

**From Kaggle** (Recommended):
1. Go to: https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria
2. Download `cell_images.zip`
3. Extract to get `cell_images/` folder

**From NIH** (Alternative):
1. Go to: https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets
2. Download the dataset
3. Extract the images

#### Step 3: Organize the Data

The downloaded dataset needs to be split into train/val:

```python
# Quick script to organize data
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# Assuming you extracted to 'cell_images/'
source_dir = Path('cell_images')
dest_dir = Path('data')

for class_name in ['Parasitized', 'Uninfected']:
    # Get all images for this class
    class_path = source_dir / class_name
    images = list(class_path.glob('*.png'))

    # Split 80/20 train/val
    train_imgs, val_imgs = train_test_split(images, test_size=0.2, random_state=42)

    # Copy to organized structure
    for split, img_list in [('train', train_imgs), ('val', val_imgs)]:
        dest_path = dest_dir / split / class_name
        dest_path.mkdir(parents=True, exist_ok=True)
        for img in img_list:
            shutil.copy(img, dest_path / img.name)

print("✅ Data organized into data/train and data/val")
```

Your final structure should look like:
```
data/
  train/
    Parasitized/  # ~11,000 images
    Uninfected/   # ~11,000 images
  val/
    Parasitized/  # ~2,800 images
    Uninfected/   # ~2,800 images
```

#### Step 4: Train with AST

```bash
# Energy-efficient training (60% energy savings)
python train_ast.py --config configs/config_ast.yaml
```

Training will take approximately:
- **With GPU (RTX 3090)**: 15-30 minutes for 30 epochs
- **Without GPU (CPU only)**: 2-4 hours for 30 epochs

You'll see real-time output like:
```
🌿 ADAPTIVE SPARSE TRAINING (AST) - Sundew Algorithm
================================================================================
Target Activation Rate: 40.0%
Expected Energy Savings: 60.0%
Strategy: Train only on 'hard' samples adaptively selected each epoch
================================================================================

[AST Epoch 1/30]
Epoch 1/30: 100%|████████| 344/344 [00:45<00:00,  7.52it/s]

[epoch 1] Summary:
  Train Loss: 0.3456
  Val Accuracy: 0.9234
  Activation Rate: 41.2%
  Energy Savings: 58.8%
  Samples Processed: 9123/22046
  ⭐ New best validation accuracy: 0.9234
```

#### Step 5: Visualize Results

```bash
python visualize_ast.py --metrics checkpoints_ast/metrics_ast.jsonl
```

This generates:
- `visualizations/ast_results.png` - Comprehensive 4-panel analysis
- `visualizations/ast_headline.png` - Social media ready graphic

#### Step 6: Evaluate on Test Set

```bash
python eval.py --weights checkpoints_ast/best.pt
```

Outputs:
- `checkpoints/report.json` - Precision, recall, F1 scores
- `checkpoints/cm.png` - Confusion matrix

#### Step 7: Generate Grad-CAM Visualizations

```bash
# Pick a sample image from validation set
python gradcam_snapshot.py \
  --weights checkpoints_ast/best.pt \
  --image data/val/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png \
  --out gradcam_malaria.png
```

This shows **where the model is looking** to make its decision - great for interpretability!

## 🎛️ Customization

### Adjust Energy Savings

Edit `configs/config_ast.yaml`:

```yaml
# For 90% energy savings (maximum buzz!)
ast_target_activation_rate: 0.10
ast_warmup_epochs: 5

# For 50% energy savings (balanced)
ast_target_activation_rate: 0.50
ast_warmup_epochs: 2

# For 30% energy savings (conservative)
ast_target_activation_rate: 0.70
ast_warmup_epochs: 0
```

### Change Model Architecture

```yaml
# Faster training, slightly lower accuracy
model_name: resnet18

# Default - good balance
model_name: efficientnet_b0

# Higher accuracy, slower training
model_name: resnet50
```

### Tune Hyperparameters

```yaml
epochs: 30              # More epochs = better convergence
batch_size: 64          # Larger = faster but needs more VRAM
learning_rate: 0.0003   # Higher = faster learning, less stable
```

## 📊 Understanding the Output

### Metrics Files

**`metrics_ast.jsonl`** contains per-epoch metrics:
```json
{
  "epoch": 10,
  "timestamp": 1704067200,
  "train_loss": 0.1234,
  "val_loss": 0.1456,
  "val_acc": 0.9567,
  "lr": 0.0003,
  "activation_rate": 0.412,
  "energy_savings": 58.8,
  "samples_processed": 9123,
  "total_samples": 22046
}
```

**`metrics_ast.csv`** has the same data in spreadsheet-friendly format.

### Checkpoint Files

- **`best.pt`**: Model with highest validation accuracy (use for deployment)
- **`last.pt`**: Most recent model (use for resuming training)
- **`resume_meta.pt`**: Resume metadata (epoch number, etc.)

## 🔧 Troubleshooting

### Out of Memory (CUDA)

Reduce batch size in config:
```yaml
batch_size: 32  # or even 16
```

### Training Too Slow (CPU)

Use a smaller model:
```yaml
model_name: resnet18
```

### Poor Accuracy

Try:
1. Increase epochs: `epochs: 50`
2. Add warmup: `ast_warmup_epochs: 5`
3. Increase activation rate: `ast_target_activation_rate: 0.70`
4. Check data quality (corrupted images?)

### Resume Interrupted Training

Just re-run the same command:
```bash
python train_ast.py --config configs/config_ast.yaml
```

It automatically resumes from `last.pt` if `resume: true` in config.

## 🎯 Next Steps

### For Research/Publication

1. **Compare with Baseline**:
   ```bash
   # Train without AST
   python train.py --config configs/config.yaml

   # Compare results
   python visualize_ast.py \
     --metrics checkpoints_ast/metrics_ast.jsonl \
     --baseline-metrics checkpoints/metrics.jsonl
   ```

2. **Run Multiple Seeds**:
   Train 3-5 times with different random seeds to get mean ± std results.

3. **Hyperparameter Sweep**:
   Try different `ast_target_activation_rate` values: 0.1, 0.2, 0.4, 0.6, 0.8

### For Deployment

1. **Export to ONNX**:
   ```bash
   python export_onnx.py \
     --weights checkpoints_ast/best.pt \
     --precision fp16 \
     --out malaria_detector.onnx
   ```

2. **Test Inference Speed**:
   Use ONNX Runtime for fast inference on edge devices

3. **Quantize for Mobile**:
   Apply post-training quantization for deployment on smartphones

### For Media/Outreach

1. **Generate Press Kit Graphics**:
   ```bash
   python visualize_ast.py --metrics checkpoints_ast/metrics_ast.jsonl
   ```
   Use `visualizations/ast_headline.png` for social media

2. **Write Blog Post**:
   Use `PRESS_KIT.md` for headline ideas and talking points

3. **Prepare Demo**:
   Use Grad-CAM visualizations to show interpretable AI decisions

## 📚 Additional Resources

- **Dataset**: [NIH Malaria Cell Images](https://lhncbc.nlm.nih.gov/LHC-downloads/downloads.html#malaria-datasets)
- **AST Library**: [adaptive-sparse-training on PyPI](https://pypi.org/project/adaptive-sparse-training/)
- **Technical Docs**: See `CLAUDE.md` in this repository
- **Press Kit**: See `PRESS_KIT.md` for media resources

## 🤝 Getting Help

If you encounter issues:

1. **Check the documentation**:
   - `CLAUDE.md` - Technical architecture
   - `README_AST.md` - Project overview
   - This file - Setup guide

2. **Common issues**:
   - Data organization - Ensure data/train and data/val exist
   - Dependencies - Run `pip install -r requirements.txt` again
   - CUDA errors - Try CPU mode or reduce batch size

3. **Contact**:
   - Open a GitHub issue
   - Email: [your contact info]

## 🎉 Success Checklist

- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Dataset downloaded and organized (data/train, data/val)
- [ ] Training completed (checkpoints_ast/best.pt exists)
- [ ] Visualizations generated (visualizations/ folder)
- [ ] Evaluation run (checkpoints/report.json, cm.png)
- [ ] Grad-CAM tested (gradcam_malaria.png)

Once all checked, you're ready to:
- 📝 Write papers/blog posts
- 📊 Present at conferences
- 🚀 Deploy in production
- 🎤 Talk to media

**Congratulations on building energy-efficient AI for global health! 🌍💚**
