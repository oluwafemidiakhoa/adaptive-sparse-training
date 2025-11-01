# PyPI Token Issue - FIXED GUIDE

## The Problem

The token you're creating keeps being project-scoped instead of account-scoped.

---

## ✅ CORRECT Way to Create Token

### Step 1: Go to Token Page
https://pypi.org/manage/account/token/

### Step 2: Click "Add API token"

### Step 3: Fill Form EXACTLY Like This:

```
Token name: ast-full-account
```

### Step 4: FOR SCOPE - CRITICAL!

You should see TWO radio button options:

```
○ Scope to an entire account
○ Scope to a specific project
```

**SELECT THE FIRST ONE**: ○ Scope to an entire account

**DO NOT SELECT**: ○ Scope to a specific project

### Step 5: Click "Add token"

### Step 6: Copy the Token

The token should start with: `pypi-AgEI...`

---

## How to Verify You Did It Right

After creating the token, look at the token list on:
https://pypi.org/manage/account/token/

Your new token should show:
```
Token name: ast-full-account
Scope: entire account          ← Should say "entire account"
Created: ...
```

**NOT**:
```
Scope: Project: adaptive-sparse-training  ← This is WRONG for first upload
```

---

## If You Keep Getting Project-Scoped Token

This might happen if you have the project name in a field somewhere. Make sure:

1. Token name can be ANYTHING (doesn't matter)
2. The radio button says "Scope to an entire account"
3. There should be NO dropdown showing "adaptive-sparse-training"

---

## Alternative: Use TestPyPI First

If you keep having issues, we can use TestPyPI to practice:

1. Create account at: https://test.pypi.org/account/register/
2. Create token at: https://test.pypi.org/manage/account/token/
3. Upload with: `twine upload --repository testpypi dist/*`
4. Then do the real upload after practicing

---

## Next Steps

Once you have the ACCOUNT-SCOPED token:

```bash
twine upload dist/*
```

Username: `__token__`
Password: [your new account-scoped token]

Let me know when you have the token created and I'll help you upload!
