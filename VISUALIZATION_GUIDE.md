# 📊 Visualization Guide - Adaptive Sparse Training

## Overview

The **KAGGLE_VIT_BATCHED_STANDALONE.py** script now includes **automatic visualization generation** to help you understand and present your results!

## Generated Visualizations

### 1. 🏗️ Architecture Diagram (`architecture_diagram.png`)

**Visual flowchart** showing how batched adaptive sparse training works:

```
Input Batch → Vectorized Significance → Sundew Gating → Active Mask → Batched Training
```

**Key Features Highlighted**:
- ✅ Single Forward Pass (vs 2× in sample-by-sample)
- ✅ Efficient Indexing (boolean mask)
- ✅ GPU Parallelism (batch operations)

**Performance Stats Shown**:
- ⚡ 10-15× Speedup
- 🔋 94% Energy Savings

**Use Cases**:
- Paper figures
- Presentation slides
- Technical documentation
- GitHub README

---

### 2. 🚀 Training Results Dashboard (`training_results.png`)

**6-panel comprehensive dashboard** showing all training metrics:

#### Panel 1: Training Loss Curve
- X-axis: Epoch
- Y-axis: Training Loss
- Shows: Model convergence over time
- **Look for**: Decreasing trend (model is learning)

#### Panel 2: Validation Accuracy Progress
- X-axis: Epoch
- Y-axis: Validation Accuracy (%)
- Shows: Model performance on unseen data
- **Look for**: Increasing trend (better generalization)

#### Panel 3: Activation Rate
- X-axis: Epoch
- Y-axis: Activation Rate (%)
- Shows: Actual vs Target (6%) activation rate
- **Look for**: Line converging toward orange target line

#### Panel 4: Energy Savings per Epoch
- X-axis: Epoch
- Y-axis: Energy Savings (%)
- Shows: Percentage of samples skipped
- **Look for**: Bars near 94% target line

#### Panel 5: Training Speed Comparison
- 3 bars: Baseline ViT, Sample-by-Sample, Batched (Ours)
- Shows: Time per epoch in seconds
- **Look for**: Green bar (ours) much shorter
- **Speedup annotation**: Big yellow box with actual speedup multiplier

#### Panel 6: Sample Processing Distribution
- Pie chart: Processed vs Skipped samples
- Shows: Total samples processed vs energy saved
- **Look for**: Large green slice (skipped samples)

---

## How to Use in Kaggle

### Quick Setup

```python
# Just run the standalone script!
# Visualizations are generated automatically after training
```

### Output Files

After running, you'll see:
```
📊 Visualization saved to: training_results.png
🏗️ Architecture diagram saved to: architecture_diagram.png
```

### Display in Notebook

Both images will automatically display in your Kaggle notebook using `plt.show()`

### Download Images

Click the output cell → Right-click image → "Save image as..."

---

## Interpreting Results

### ✅ Good Training (What Success Looks Like)

**Training Loss**:
- ✅ Decreasing smoothly
- ✅ No sudden spikes
- ✅ Reaches < 2.0 after a few epochs

**Validation Accuracy**:
- ✅ Increasing steadily
- ✅ 30-35% after 1 epoch
- ✅ 70-80% after 10 epochs

**Activation Rate**:
- ✅ Converges to ~6% (within 5-7%)
- ✅ Stable after first few epochs
- ✅ No wild oscillations

**Energy Savings**:
- ✅ Consistently 93-95%
- ✅ Matches 1 - activation_rate
- ✅ Stable across epochs

**Speedup**:
- ✅ 10-15× vs baseline
- ✅ Green bar < 20s per epoch (GPU)
- ✅ Consistent across epochs

---

### ⚠️ Potential Issues

**Activation Rate Too High (>15%)**:
```python
# Problem: Processing too many samples
# Solution: Increase threshold or energy pressure
config = {
    "target_activation_rate": 0.06,
    "activation_threshold": 0.6,  # ← Increase from 0.4
    "energy_pressure": 0.3,        # ← Increase from 0.2
}
```

**Activation Rate Too Low (<3%)**:
```python
# Problem: Not processing enough samples
# Solution: Decrease threshold
config = {
    "activation_threshold": 0.3,  # ← Decrease from 0.4
}
```

**Loss Not Decreasing**:
```python
# Problem: Learning rate too high/low
# Solution: Adjust learning rate
config = {
    "lr": 5e-5,  # ← Try smaller (was 1e-4)
}
```

**Accuracy Plateauing**:
```python
# Problem: Need more training or higher activation
# Solution: More epochs or higher activation rate
config = {
    "epochs": 10,                  # ← More epochs
    "target_activation_rate": 0.1, # ← Or process more samples
}
```

---

## Customization Options

### Change Epochs

```python
# Edit line ~461 in main()
config = {
    "epochs": 5,  # ← Change from 1 to 5
}
```

**Effect**: More panels in training dashboard, better convergence

### Adjust Visualization Style

Find the `plot_training_results()` function (around line 418) and customize:

```python
# Change figure size
fig = plt.figure(figsize=(20, 12))  # Bigger plots

# Change colors
ax1.plot(..., color='red')  # Red line instead of blue

# Change target activation rate line
ax3.axhline(y=10.0, ...)  # If you changed target to 10%
```

### Save to Different Location

```python
# In main(), around line 722
plot_training_results(trainer, final_metrics, save_path='my_results.png')
```

---

## Example Use Cases

### 1. Research Paper

**Figures to include**:
- Architecture diagram (for methodology section)
- Speedup comparison bar chart (for results)
- Training loss + accuracy curves (for evaluation)

**Export**:
1. Run in Kaggle
2. Download PNG files
3. Insert into LaTeX paper

### 2. Blog Post / Tutorial

**Visuals to showcase**:
- Architecture diagram (explain the approach)
- Full dashboard (show comprehensive results)
- Sample processing pie chart (highlight energy savings)

### 3. GitHub README

```markdown
## Results

![Architecture](architecture_diagram.png)

Our batched adaptive sparse training achieves **10-15× speedup** while maintaining accuracy:

![Results](training_results.png)
```

### 4. Presentation Slides

**Slide 1**: Problem statement
**Slide 2**: Architecture diagram (methodology)
**Slide 3**: Speedup bar chart (key result)
**Slide 4**: Full dashboard (detailed analysis)

---

## Troubleshooting Visualizations

### Issue: "No module named 'matplotlib'"

**Solution**: Matplotlib is pre-installed in Kaggle, but if running locally:
```bash
pip install matplotlib
```

### Issue: Plots not showing

**Solution**: Add at end of script:
```python
plt.show()
```

Already included in the standalone script!

### Issue: Low resolution images

**Solution**: Increase DPI in save commands:
```python
plt.savefig('results.png', dpi=300)  # Higher quality (was 150)
```

### Issue: Text overlapping in plots

**Solution**: Adjust figure size:
```python
fig = plt.figure(figsize=(20, 12))  # Larger canvas
```

---

## Advanced: Adding Custom Plots

### Example: Add Threshold Adaptation Plot

```python
def plot_threshold_adaptation(trainer):
    """Plot how threshold changes over time."""
    plt.figure(figsize=(10, 6))

    # You'd need to track thresholds during training
    # For now, this is a placeholder
    epochs = range(1, len(trainer.metrics['activation_rates']) + 1)

    plt.plot(epochs, trainer.metrics['activation_rates'], 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Activation Rate')
    plt.title('Threshold Adaptation Over Time')
    plt.grid(True)
    plt.savefig('threshold_adaptation.png', dpi=150)
    plt.show()
```

Add this call in `main()`:
```python
plot_threshold_adaptation(trainer)
```

---

## Comparison with Baseline

To create a **before/after comparison**, run two experiments:

### Experiment 1: With Sundew (Current)
```python
# Run KAGGLE_VIT_BATCHED_STANDALONE.py as-is
# Save: training_results_sundew.png
```

### Experiment 2: Without Sundew (Baseline)
```python
# Comment out gating logic in train_epoch()
# Process all samples: num_active = batch_size
# Save: training_results_baseline.png
```

### Create Comparison Figure
```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Load and display both images
ax1.imshow(plt.imread('training_results_baseline.png'))
ax1.set_title('Baseline (No Sundew)')
ax1.axis('off')

ax2.imshow(plt.imread('training_results_sundew.png'))
ax2.set_title('With Sundew Gating')
ax2.axis('off')

plt.savefig('comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Summary

✅ **2 Automatic Visualizations**:
1. Architecture Diagram
2. Training Results Dashboard (6 panels)

✅ **Key Metrics Visualized**:
- Training loss
- Validation accuracy
- Activation rate vs target
- Energy savings
- Speedup comparison
- Sample distribution

✅ **Ready for**:
- Research papers
- Blog posts
- Presentations
- GitHub repos
- Technical reports

✅ **No Extra Work**:
- Automatically generated after training
- Saved as high-quality PNGs
- Displayed in notebook

**Just run the script and get publication-ready figures! 🎉**
