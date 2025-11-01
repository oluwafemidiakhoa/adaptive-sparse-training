# Publish to PyPI - Two Options

## ✅ Option 1: Use GitHub Actions (EASIEST!)

This is the recommended way - fully automated!

### Step 1: Add PyPI Token to GitHub Secrets

1. **Get your account-scoped token from PyPI**:
   - You need the one that shows "Scope: entire account"
   - If you don't have it, create one at: https://pypi.org/manage/account/token/

2. **Add to GitHub**:
   - Go to: https://github.com/oluwafemidiakhoa/adaptive-sparse-training/settings/secrets/actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste your account-scoped PyPI token
   - Click "Add secret"

### Step 2: Create a GitHub Release

**Option A: Via GitHub Web Interface**
1. Go to: https://github.com/oluwafemidiakhoa/adaptive-sparse-training/releases/new
2. Fill in:
   - **Tag version**: `v1.0.0` (create new tag)
   - **Release title**: `Release 1.0.0 - Initial PyPI Release`
   - **Description**:
     ```
     🎉 First release of Adaptive Sparse Training on PyPI!

     Install: pip install adaptive-sparse-training

     Features:
     - 3-line integration for energy-efficient training
     - 60%+ energy savings with zero accuracy loss
     - Validated on ImageNet-100 (92.12% accuracy)
     - Mixed precision training support
     - PI-controlled adaptive sample selection
     ```
3. Click "Publish release"

**Option B: Via Command Line**
```bash
git tag v1.0.0
git push origin v1.0.0
# Then create release on GitHub web interface
```

### Step 3: Watch It Publish!

1. Go to: https://github.com/oluwafemidiakhoa/adaptive-sparse-training/actions
2. You'll see "Publish to PyPI" workflow running
3. Wait ~2 minutes for it to complete
4. Check https://pypi.org/project/adaptive-sparse-training/ - Your package is live!

---

## 🛠️ Option 2: Manual Upload from Command Line

If you prefer manual control or GitHub Actions isn't working:

### You Need:

1. **Account-scoped PyPI token** that shows "Scope: entire account"
   - If you only have project-scoped, go create a new one:
   - https://pypi.org/manage/account/token/
   - Select "Scope to entire account" radio button

2. **Run upload command**:

```bash
twine upload dist/*
```

**When prompted:**
- Username: `__token__`
- Password: [paste your account-scoped token]

---

## After Publishing

Once live on PyPI (via either method), anyone can install with:

```bash
pip install adaptive-sparse-training
```

**I'll then update the README to show the pip install command!**

---

## Which Option Should You Choose?

**Use GitHub Actions (Option 1) if:**
- ✅ You want automated publishing
- ✅ You want a professional release workflow
- ✅ You plan to publish updates in the future
- ✅ You want version tagging and release notes

**Use Manual Upload (Option 2) if:**
- ✅ You want immediate control
- ✅ You're having issues with GitHub Actions
- ✅ You're only publishing once

---

## 🚨 Important: Token Must Be Account-Scoped!

For first upload, the token MUST say:
```
Scope: entire account
```

NOT:
```
Scope: Project: adaptive-sparse-training
```

The project-scoped token will NOT work until after the first upload.

---

## Troubleshooting

### "Invalid API Token" error

**Problem**: Token is project-scoped instead of account-scoped

**Fix**:
1. Go to: https://pypi.org/manage/account/token/
2. Delete existing tokens
3. Create NEW token
4. When creating, select radio button "Scope to entire account"
5. Copy the new token

### GitHub Actions workflow fails

**Problem**: Wrong secret name or token

**Fix**:
1. Verify secret is named exactly `PYPI_API_TOKEN`
2. Verify token in secret is account-scoped
3. Re-run workflow

---

**Let me know which option you choose and when it's published!**
