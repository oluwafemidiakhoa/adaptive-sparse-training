# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a malaria cell classification starter project using PyTorch. It trains deep learning models (EfficientNet-B0 or ResNet50) to classify microscopy images as either Parasitized or Uninfected.

## Data Structure

The project expects ImageFolder-style data organization:
```
data/
  train/
    Parasitized/  # PNG/JPG images of infected cells
    Uninfected/   # PNG/JPG images of healthy cells
  val/
    Parasitized/
    Uninfected/
```

## Core Commands

### Setup
```bash
pip install -r requirements.txt
```

### Training

#### Standard Training
```bash
# Train with automatic resume capability
python train.py --config configs/config.yaml

# Explicit resume (forces resume=True in config)
python train_resume.py --config configs/config.yaml
```

#### Energy-Efficient Training with AST (Recommended! 🌿)
```bash
# Train with Adaptive Sparse Training (40-90% energy savings!)
python train_ast.py --config configs/config_ast.yaml
```

Training automatically:
- Saves checkpoints to `checkpoints/` or `checkpoints_ast/` (configurable via `save_dir`)
- Saves `best.pt` (best validation accuracy) and `last.pt` (most recent epoch)
- Logs metrics to `metrics.csv` and `metrics.jsonl` (AST adds activation_rate, energy_savings)
- Resumes from `last.pt` if it exists and `resume: true` in config

### Evaluation
```bash
python eval.py --weights checkpoints/best.pt
```

Outputs:
- `checkpoints/report.json` - Classification report with precision/recall/F1
- `checkpoints/cm.png` - Confusion matrix visualization

### Grad-CAM Visualization
```bash
python gradcam_snapshot.py --weights checkpoints/best.pt \
  --image data/val/Parasitized/<somefile>.png --out cam_sample.png
```

### ONNX Export
```bash
# FP32
python export_onnx.py --weights checkpoints/best.pt --out model_fp32.onnx

# FP16
python export_onnx.py --weights checkpoints/best.pt --precision fp16 --out model_fp16.onnx
```

### AST Visualization & Analysis
```bash
# Generate comprehensive visualizations of AST training results
python visualize_ast.py --metrics checkpoints_ast/metrics_ast.jsonl --output-dir visualizations

# Compare AST vs baseline (optional)
python visualize_ast.py --metrics checkpoints_ast/metrics_ast.jsonl \
  --baseline-metrics checkpoints/metrics.jsonl --output-dir visualizations
```

Outputs:
- `visualizations/ast_results.png` - 4-panel comprehensive analysis
- `visualizations/ast_headline.png` - Social media / press release ready graphic
- `visualizations/ast_vs_baseline.png` - Direct comparison (if baseline provided)

## Architecture

### Training Pipeline (`train.py`)

**Data Loading** (`get_loaders`):
- Training: RandomResizedCrop, RandomHorizontalFlip, ColorJitter augmentation
- Validation: CenterCrop only
- Both: Resize to 1.15x image_size before cropping
- Uses PyTorch ImageFolder for automatic class label extraction

**Model Building** (`build_model`):
- Supports `efficientnet_b0` and `resnet50`
- Models loaded without pretrained weights (`weights=None`)
- Final classification layer replaced to match `num_classes`

**Training Loop**:
- AdamW optimizer with configurable learning rate and weight decay
- Mixed precision training (AMP) enabled by default for CUDA
- Early stopping based on validation accuracy (`patience` parameter)
- Saves both `last.pt` (for resuming) and `best.pt` (best validation)
- Resume metadata stored in `resume_meta.pt` (contains last epoch number)

**Resume Logic** (`load_resume`):
- Checks for `last.pt` in save_dir
- Loads `resume_meta.pt` to get starting epoch
- Returns 0 if no checkpoint exists

### Evaluation (`eval.py`)

- Uses scikit-learn's `classification_report` and `confusion_matrix`
- Generates JSON report with per-class metrics
- Creates confusion matrix heatmap using seaborn

### Grad-CAM (`cam_utils.py`, `gradcam_snapshot.py`)

**`cam_utils.grad_cam`**:
- Automatically selects target layer:
  - For models with `features` attribute (e.g., EfficientNet): uses last layer in features
  - Otherwise (e.g., ResNet): uses `layer4[-1]`
- Uses forward/backward hooks to capture activations and gradients
- Returns prediction class and overlay visualization

### Common Patterns

**Model Loading**: All scripts (eval, gradcam, export) try two load patterns:
```python
try:
    model.load_state_dict(state)
except Exception:
    model.load_state_dict(state["state_dict"])
```
This handles both raw state_dict saves and wrapped checkpoint formats.

**Configuration**: Training uses YAML config (`configs/config.yaml`) parsed with PyYAML. Key parameters:
- `model_name`: "efficientnet_b0" or "resnet50"
- `num_classes`: 2 (binary classification)
- `image_size`: 224
- `resume`: true/false (auto-resume from last.pt)
- `patience`: early stopping patience

## Adding New Models

To add a new model architecture:

1. Add a new branch in `build_model()` in `train.py`:
```python
elif name == "your_model":
    m = models.your_model(weights=None)
    # Replace final layer to match num_classes
    return m
```

2. Replicate the same logic in `eval.py`, `gradcam_snapshot.py`, and `export_onnx.py`

3. Update `configs/config.yaml` with `model_name: your_model`

## Metrics Logging

Metrics are logged in two formats:
- **CSV** (`metrics.csv` / `metrics_ast.csv`): Tabular format for easy spreadsheet import
- **JSONL** (`metrics.jsonl` / `metrics_ast.jsonl`): One JSON object per line for programmatic parsing

Standard training metrics: epoch, timestamp, train_loss, val_loss, val_acc, lr

AST training adds: activation_rate, energy_savings, samples_processed, total_samples

## Adaptive Sparse Training (AST) Deep Dive

### What is AST?

Adaptive Sparse Training uses the **Sundew algorithm** to intelligently select which training samples to process each epoch based on their difficulty. By focusing compute on "hard" samples and skipping "easy" ones, AST achieves 40-90% energy savings while maintaining competitive accuracy.

### Key Concept: Sample Selection

Traditional training processes 100% of samples every epoch. AST uses a dynamic threshold based on loss magnitude:
- **High loss samples** (difficult) → Always processed
- **Low loss samples** (easy, already learned) → Skipped
- Threshold adapts via PI controller to hit target activation rate

### Configuration (`config_ast.yaml`)

**Primary tuning knob:**
- `ast_target_activation_rate`: Fraction of samples to process (0.4 = 40% = 60% energy savings)

**Advanced parameters (rarely need tuning):**
- `ast_initial_threshold`: Starting threshold for sample difficulty
- `ast_adapt_kp`, `ast_adapt_ki`: PI controller gains for threshold adaptation
- `ast_ema_alpha`: Smoothing factor for activation rate tracking
- `ast_warmup_epochs`: Optional epochs with 100% activation before AST kicks in

### AST Training Flow (`train_ast.py`)

1. **Initialization**: Creates `AdaptiveSparseTrainer` wrapper around model, data loaders, optimizer
2. **Per-Epoch**:
   - Forward pass on batch → compute loss per sample
   - Sundew gate: samples with `loss > threshold` are activated
   - Backward pass only on activated samples
   - PI controller adjusts threshold to hit target activation rate
3. **Metrics**: Tracks activation_rate, energy_savings, samples_processed
4. **Checkpointing**: Same as standard training (best.pt, last.pt)

### Energy Savings Calculation

Energy is modeled as:
```
energy_per_batch = (n_activated * energy_per_activation) + (n_skipped * energy_per_skip)
```

Where `energy_per_skip << energy_per_activation` (default: 0.01 vs 1.0), since skipping samples avoids expensive backward pass and optimizer steps.

Savings % = `(baseline_energy - ast_energy) / baseline_energy * 100`

### When to Use AST

**✅ Use AST when:**
- Training on resource-constrained hardware (single GPU, limited power budget)
- Dataset has redundancy / easy samples (common in medical imaging)
- Energy efficiency is a key metric (climate/sustainability concerns)
- You want buzzworthy results ("90% energy savings!")

**⚠️ Be cautious when:**
- Dataset is very small (<1k samples) - may not have enough redundancy
- Every sample is critical (rare diseases, edge cases)
- You need absolute maximum accuracy (AST may sacrifice 1-2%)

### Recommended AST Settings

**For maximum buzz (90% savings):**
```yaml
ast_target_activation_rate: 0.10
ast_warmup_epochs: 5
```
Expect: ~1-2% accuracy drop, amazing headlines

**For balanced efficiency (60% savings):**
```yaml
ast_target_activation_rate: 0.40
ast_warmup_epochs: 0
```
Expect: <0.5% accuracy drop, strong results

**For conservative adoption (30% savings):**
```yaml
ast_target_activation_rate: 0.70
ast_warmup_epochs: 0
```
Expect: minimal accuracy impact, still significant savings
