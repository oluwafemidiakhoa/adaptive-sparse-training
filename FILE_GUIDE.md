# 📁 File Organization Guide

> **Evidence status:** ImageNet-100 results are reported, with verification pending. Numerical examples and projections are not measurements. Energy savings are modeled and speedups are estimated. See [IMAGENET100_STATUS.md](IMAGENET100_STATUS.md) for evidence requirements.


## ImageNet-100 experimental implementations

### 1. `KAGGLE_IMAGENET100_AST_PRODUCTION.py`
**Status:** Reported results; verification pending
**Purpose:** Version with the higher reported best accuracy

**Results:**
```
Reported best accuracy: 92.12%
Modeled Energy Savings: 61.49%
Estimated Speedup: 1.92×
Reported runtime: 514.7 minutes
```

**When to use:**
- Publication use requires verification artifacts
- ✅ Benchmarking comparisons
- ✅ When accuracy is critical
- ✅ Investigating accuracy preservation (verification pending)

**Features:**
- Mixed Precision (AMP)
- Gradient masking
- 40% target activation (observed rate requires run artifacts)
- 8 workers + prefetching

---

### 2. `KAGGLE_IMAGENET100_AST_TWO_STAGE_Prod.py`
**Status:** Reported results; verification pending
**Purpose:** Version with the larger activation-based speedup estimate

**Results:**
```
Reported best accuracy: 91.92%
Modeled Energy Savings: 63.36%
Estimated Speedup: 2.78×
Reported runtime: 519.8 minutes
```

**When to use:**
- Experimental investigation; deployment suitability unverified
- Study sample selection under resource constraints
- Investigate runtime against a matched baseline
- Demonstrate modeled selection metrics

**Features:**
- Standard training (no AMP)
- 20% target activation in the current script; observed rate requires run artifacts
- Higher modeled savings; actual energy savings unmeasured
- More aggressive pruning

---

## 📁 ARCHIVE FILES (EXPERIMENTAL)

### Location: `archive/`

These files were development iterations. **Don't use for production!**

#### `KAGGLE_IMAGENET100_AST_QUICK_TEST.py`
- Purpose: 5-epoch validation test
- Status: ⚠️ EXPERIMENTAL
- Use for: Quick prototyping, testing changes

#### `KAGGLE_IMAGENET100_AST_TWO_STAGE_FIXED.py`
- Purpose: Bug fix iteration
- Status: ⚠️ SUPERSEDED
- Replaced by: TWO_STAGE_Prod.py

#### `KAGGLE_IMAGENET100_AST_ULTIMATE_FAST.py`
- Purpose: Speed optimization attempt
- Status: ⚠️ SUPERSEDED
- Replaced by: PRODUCTION.py

---

## 📊 RESULTS & DOCUMENTATION

### `results/RESULTS_SUMMARY.md`
**Complete comparison of both experimental versions**
- Side-by-side metrics
- When to use which version
- Technical details

### `README.md`
**Main project documentation**
- Overview
- Quick start guide
- Method explanation
- Citation information

### `FILE_GUIDE.md` (This file)
**Quick reference for file usage**

---

## 🎯 DECISION MATRIX

### "Which file should I use?"

```
Need the higher reported best accuracy?
  → KAGGLE_IMAGENET100_AST_PRODUCTION.py (92.12%)

Need the higher modeled savings estimate?
  → KAGGLE_IMAGENET100_AST_TWO_STAGE_Prod.py (63%)

Need the higher estimated speedup?
  → KAGGLE_IMAGENET100_AST_TWO_STAGE_Prod.py (2.78×)

Testing new features?
  → Copy PRODUCTION.py and modify

Quick validation?
  → archive/KAGGLE_IMAGENET100_AST_QUICK_TEST.py
```

---

## ⚠️ IMPORTANT NOTES

### DO:
✅ Use PRODUCTION.py for papers/sharing
✅ Use TWO_STAGE_Prod.py for efficiency claims
✅ Keep all files (don't delete!)
✅ Cite appropriate version in publications

### DON'T:
❌ Mix results from different versions
❌ Use archive files for production
❌ Delete experimental files (historical record!)
❌ Modify production files directly (copy first!)

---

## 🔄 UPDATE HISTORY

| Date | Version | Change | File |
|------|---------|--------|------|
| 2025-01-24 | 2.0 | Best accuracy (92.12%) | PRODUCTION.py |
| 2025-01-24 | 2.0 | Maximum efficiency (2.78×) | TWO_STAGE_Prod.py |
| 2025-01-23 | 1.5 | Speed optimizations | ULTIMATE_FAST.py (archived) |
| 2025-01-22 | 1.0 | Initial baseline | TWO_STAGE.py (archived) |

---

## 📧 Questions?

If confused about which file to use, default to:
**`KAGGLE_IMAGENET100_AST_PRODUCTION.py`**

It has the best accuracy and most stable performance.

---

**Last Updated:** January 24, 2025
**Status:** Current and accurate
