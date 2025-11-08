# Release Instructions for v1.0.1

## Bug Fix Summary

**Fixed Critical Bug**: `IndexError` when using `warmup_epochs > 0`

The library was failing with error:
```
IndexError: Dimension specified as 0 but tensor has no dimensions
```

This occurred because the loss tensor validation was missing when warmup epochs were used.

## Changes Made

1. **sundew.py**:
   - Added scalar loss detection in `compute_significance()` method
   - Added scalar loss detection in `select_samples()` method
   - Improved error messages with clear guidance

2. **trainer.py**:
   - Added warning when criterion doesn't use `reduction='none'`
   - Better validation of criterion configuration

3. **Version bump**: 1.0.0 → 1.0.1

4. **Created**:
   - CHANGELOG.md to track all version changes
   - test_warmup_fix.py to verify the bug fix

## Test Results

Test passed successfully:
- Warmup epochs 1-2: 100% activation (as expected)
- AST epoch 3: 14% activation with 28.4% energy savings
- No errors during training with warmup_epochs=2

## Upload to PyPI

### Step 1: Set your PyPI API token

You need to configure your PyPI API token. Run:

```bash
# On Windows
set TWINE_USERNAME=__token__
set TWINE_PASSWORD=<your-pypi-token>

# Or create/update ~/.pypirc with:
[pypi]
username = __token__
password = <your-pypi-token>
```

### Step 2: Upload the package

The distribution files are already built in `dist/`:
- `adaptive_sparse_training-1.0.1-py3-none-any.whl`
- `adaptive_sparse_training-1.0.1.tar.gz`

Upload with:

```bash
python -m twine upload dist/adaptive_sparse_training-1.0.1*
```

### Step 3: Verify on PyPI

After upload, verify at:
https://pypi.org/project/adaptive-sparse-training/

### Step 4: Test installation

```bash
pip install --upgrade adaptive-sparse-training
pip show adaptive-sparse-training  # Should show version 1.0.1
```

## Git Commit and Tag

After successful PyPI upload:

```bash
git add .
git commit -m "Fix warmup_epochs bug (v1.0.1)

- Add scalar loss tensor validation in Sundew algorithm
- Add criterion reduction warning in trainer
- Add comprehensive error messages
- Add CHANGELOG.md
- Bump version to 1.0.1"

git tag -a v1.0.1 -m "Version 1.0.1 - Fix warmup_epochs bug"
git push origin main --tags
```

## Release Notes for GitHub

Title: **v1.0.1 - Fix warmup_epochs Bug**

Body:
```markdown
## Fixed
- **Critical Bug**: Fixed `IndexError: Dimension specified as 0 but tensor has no dimensions` when using `warmup_epochs > 0`
  - Added validation in `SundewAlgorithm.compute_significance()` to detect scalar loss tensors
  - Added validation in `SundewAlgorithm.select_samples()` to detect scalar loss tensors
  - Added warning in `AdaptiveSparseTrainer.__init__()` when criterion doesn't use `reduction='none'`
  - Improved error messages to guide users toward correct loss function configuration

## Changed
- Enhanced error handling to provide clear guidance when loss tensors have incorrect dimensions

## Files Changed
- `adaptive_sparse_training/sundew.py`
- `adaptive_sparse_training/trainer.py`
- `setup.py` (version bump)
- Added `CHANGELOG.md`
```
