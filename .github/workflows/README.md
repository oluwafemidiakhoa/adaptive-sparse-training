# GitHub Actions Workflows

This directory contains automated workflows for Adaptive Sparse Training.

## Workflows

### 1. `publish-to-pypi.yml` - Automatic PyPI Publishing

**Triggers:**
- When you create a new GitHub Release
- Manual trigger via GitHub Actions UI

**What it does:**
1. Builds Python distribution packages (wheel + source)
2. Checks package validity
3. Uploads to PyPI automatically

**Setup Required:**

#### Add PyPI Token to GitHub Secrets:

1. **Get your PyPI token** (account-scoped):
   - Go to: https://pypi.org/manage/account/token/
   - Create token with "Scope to entire account"
   - Copy the token (starts with `pypi-AgEI...`)

2. **Add to GitHub Secrets**:
   - Go to your repo: https://github.com/oluwafemidiakhoa/adaptive-sparse-training/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste your PyPI token
   - Click "Add secret"

#### How to Use:

**Option A: Create a GitHub Release (Recommended)**
```bash
# Tag and push
git tag v1.0.0
git push origin v1.0.0

# Then create release on GitHub:
# Go to: https://github.com/oluwafemidiakhoa/adaptive-sparse-training/releases/new
# - Tag: v1.0.0
# - Title: "Release 1.0.0 - Initial PyPI Release"
# - Description: Release notes
# - Click "Publish release"

# Workflow automatically triggers and publishes to PyPI!
```

**Option B: Manual Trigger**
- Go to: https://github.com/oluwafemidiakhoa/adaptive-sparse-training/actions/workflows/publish-to-pypi.yml
- Click "Run workflow"
- Select branch: main
- Click "Run workflow"

---

### 2. `test.yml` - Automated Testing

**Triggers:**
- Every push to `main` branch
- Every pull request to `main`

**What it does:**
1. Tests package on Python 3.8, 3.9, 3.10, 3.11
2. Verifies all imports work
3. Checks package metadata

**No setup required** - runs automatically!

---

## Updating Version for New Release

Before creating a new release:

1. **Update version in `setup.py`**:
   ```python
   setup(
       name="adaptive-sparse-training",
       version="1.0.1",  # ← Increment this
       ...
   )
   ```

2. **Update version in `adaptive_sparse_training/__init__.py`**:
   ```python
   __version__ = "1.0.1"  # ← Match setup.py
   ```

3. **Commit and push**:
   ```bash
   git add setup.py adaptive_sparse_training/__init__.py
   git commit -m "Bump version to 1.0.1"
   git push origin main
   ```

4. **Create release** (triggers publish workflow):
   ```bash
   git tag v1.0.1
   git push origin v1.0.1
   # Then create GitHub Release
   ```

---

## Troubleshooting

### Workflow fails with "Invalid API Token"

**Problem**: Token in GitHub Secrets is wrong or expired

**Fix**:
1. Create new token on PyPI (account-scoped)
2. Update GitHub secret `PYPI_API_TOKEN`
3. Re-run workflow

### Workflow fails with "File already exists"

**Problem**: Version already published to PyPI

**Fix**:
1. Increment version in `setup.py`
2. Commit and push
3. Create new release/tag

### Workflow doesn't trigger

**Problem**: Missing GitHub secret or workflow file issues

**Fix**:
1. Verify `PYPI_API_TOKEN` secret exists in repo settings
2. Check workflow file syntax (YAML is sensitive to indentation)
3. Ensure you pushed `.github/workflows/` directory

---

## Manual Upload (Fallback)

If GitHub Actions isn't working, you can still upload manually:

```bash
# Build
python -m build

# Upload
twine upload dist/*
# Username: __token__
# Password: [your PyPI token]
```

---

## Security Notes

- **Never commit PyPI tokens** to the repository
- Use GitHub Secrets for all sensitive tokens
- Use account-scoped tokens for first upload
- Use project-scoped tokens for subsequent uploads (more secure)
- Rotate tokens periodically

---

**Developed by Oluwafemi Idiakhoa**
