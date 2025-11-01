"""
Kaggle ImageNet-1K Competition Access Diagnostic

This script checks if you have access to the ImageNet competition dataset.
Run this in Colab BEFORE attempting the full download.
"""

import os
import subprocess
import json

print("="*70)
print("KAGGLE IMAGENET-1K ACCESS DIAGNOSTIC")
print("="*70)
print()

# Check 1: Kaggle credentials exist
print("1. Checking Kaggle credentials...")
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
if os.path.exists(kaggle_json_path):
    print("   ✅ kaggle.json found at ~/.kaggle/kaggle.json")

    # Check permissions
    stat_info = os.stat(kaggle_json_path)
    perms = oct(stat_info.st_mode)[-3:]
    if perms == '600':
        print(f"   ✅ Permissions correct (600)")
    else:
        print(f"   ⚠️  Permissions are {perms}, should be 600")
        print("      Run: !chmod 600 ~/.kaggle/kaggle.json")
else:
    print("   ❌ kaggle.json NOT found")
    print("      Upload your kaggle.json file first!")
    exit(1)

print()

# Check 2: Kaggle CLI works
print("2. Checking Kaggle CLI...")
try:
    result = subprocess.run(['kaggle', '--version'],
                          capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        print(f"   ✅ Kaggle CLI installed: {result.stdout.strip()}")
    else:
        print(f"   ❌ Kaggle CLI error: {result.stderr}")
        exit(1)
except Exception as e:
    print(f"   ❌ Kaggle CLI not working: {e}")
    exit(1)

print()

# Check 3: List competitions (test API access)
print("3. Testing Kaggle API access...")
try:
    result = subprocess.run(['kaggle', 'competitions', 'list', '--page', '1'],
                          capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print("   ✅ Kaggle API working - can list competitions")
    else:
        print(f"   ❌ API error: {result.stderr}")
        exit(1)
except Exception as e:
    print(f"   ❌ API test failed: {e}")
    exit(1)

print()

# Check 4: Test ImageNet competition access
print("4. Checking ImageNet competition access...")
competition_name = "imagenet-object-localization-challenge"

try:
    # Try to list competition files (this will fail if not accepted rules)
    result = subprocess.run(['kaggle', 'competitions', 'files', '-c', competition_name],
                          capture_output=True, text=True, timeout=30)

    if result.returncode == 0:
        print("   ✅ You have access to ImageNet competition!")
        print("\n   📁 Available files:")
        for line in result.stdout.split('\n')[:10]:  # Show first 10 lines
            if line.strip():
                print(f"      {line}")
        print()
        print("   🎉 Ready to download! Run the download cell in the notebook.")

    elif "403" in result.stderr or "Forbidden" in result.stderr:
        print("   ❌ 403 Forbidden - You haven't accepted the competition rules!")
        print()
        print("   📝 TO FIX:")
        print("      1. Open: https://www.kaggle.com/c/imagenet-object-localization-challenge")
        print("      2. Click the 'Join Competition' or 'I Understand and Accept' button")
        print("      3. Read and accept the terms")
        print("      4. Wait 1-2 minutes for permissions to propagate")
        print("      5. Re-run this diagnostic script")
        print()
        exit(1)

    else:
        print(f"   ⚠️  Unexpected response:")
        print(f"      stdout: {result.stdout[:200]}")
        print(f"      stderr: {result.stderr[:200]}")
        exit(1)

except Exception as e:
    print(f"   ❌ Check failed: {e}")
    exit(1)

print()
print("="*70)
print("✅ ALL CHECKS PASSED - READY TO DOWNLOAD IMAGENET-1K!")
print("="*70)
print()
print("Next steps:")
print("  1. Run the download cell in the Colab notebook")
print("  2. Expected download time: ~2 hours (150GB)")
print("  3. After download, training will take ~5 hours on A100")
print()
