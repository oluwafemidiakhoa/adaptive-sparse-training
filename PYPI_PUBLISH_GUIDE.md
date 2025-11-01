# PyPI Publishing Guide for Adaptive Sparse Training

**Status**: ✅ Packages built successfully! Ready to upload.

---

## What We've Done So Far

✅ Created Python package structure
✅ Installed `build` and `twine` tools
✅ Built distribution packages:
- `dist/adaptive_sparse_training-1.0.0-py3-none-any.whl` (16 KB)
- `dist/adaptive_sparse_training-1.0.0.tar.gz` (22 KB)

---

## Next Steps (Do These Yourself)

### Step 1: Create PyPI Account

1. **Go to PyPI**: https://pypi.org/account/register/
2. **Fill out registration**:
   - Username: (choose your username)
   - Email: (your email)
   - Password: (strong password)
3. **Verify email**: Check inbox and click verification link
4. **Enable 2FA** (recommended): https://pypi.org/manage/account/#two-factor

### Step 2: Create API Token

1. Go to: https://pypi.org/manage/account/token/
2. Click **"Add API token"**
3. **Token name**: `adaptive-sparse-training-upload`
4. **Scope**: Select **"Entire account"** (or specific project after first upload)
5. **Copy the token** - It looks like: `pypi-AgEIcHlwaS5vcmc...` (SAVE THIS!)

---

## Step 3: Upload to PyPI

### Option A: Using Twine (Recommended)

Open your terminal in the project directory and run:

```bash
# Upload to PyPI (will prompt for username and password)
twine upload dist/*
```

**When prompted:**
- **Username**: `__token__`  (exactly like that, with underscores)
- **Password**: Paste your API token from Step 2

### Option B: Test First on TestPyPI (Safer)

If you want to test before uploading to production PyPI:

```bash
# Upload to TestPyPI first
twine upload --repository testpypi dist/*
```

**Create TestPyPI account first**: https://test.pypi.org/account/register/

**When prompted for TestPyPI:**
- **Username**: `__token__`
- **Password**: Your TestPyPI API token

**Then test installation:**
```bash
pip install --index-url https://test.pypi.org/simple/ adaptive-sparse-training
```

If it works, proceed to upload to production PyPI (Step 3, Option A).

---

## Step 4: Verify It Works

After uploading to production PyPI, test the installation:

```bash
# Create a new virtual environment to test
python -m venv test_env
source test_env/bin/activate  # On Windows: test_env\Scripts\activate

# Install from PyPI
pip install adaptive-sparse-training

# Test import
python -c "from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig; print('Success!')"
```

---

## Step 5: Update README

After successful upload, I'll update the README to show:

```bash
pip install adaptive-sparse-training
```

---

## Troubleshooting

### Error: "403 Forbidden"
- **Cause**: Invalid API token or no upload permission
- **Fix**: Regenerate API token, ensure it's for "Entire account"

### Error: "Package already exists"
- **Cause**: Version 1.0.0 already uploaded
- **Fix**: Increment version in `setup.py` (change to 1.0.1), rebuild:
  ```bash
  rm -rf dist/ build/ *.egg-info
  python -m build
  twine upload dist/*
  ```

### Error: "Invalid distribution"
- **Cause**: Missing files or incorrect setup.py
- **Fix**: Check that `setup.py`, `README.md`, and `LICENSE` exist

---

## After Publishing

Once live on PyPI, anyone can install with:

```bash
pip install adaptive-sparse-training
```

**Package page will be at:**
https://pypi.org/project/adaptive-sparse-training/

---

## Quick Commands Summary

```bash
# 1. Build packages (already done)
python -m build

# 2. Upload to production PyPI
twine upload dist/*
# Username: __token__
# Password: <your-pypi-api-token>

# 3. Test installation
pip install adaptive-sparse-training

# 4. Verify import works
python -c "from adaptive_sparse_training import AdaptiveSparseTrainer; print('✓')"
```

---

## When You're Done

Come back and tell me once you've uploaded, and I'll:
1. Update the README to show `pip install adaptive-sparse-training`
2. Remove the "Coming Soon!" note
3. Commit and push the changes

Let me know if you run into any issues during the upload!
