"""
Quick diagnostic script to check ImageNet dataset structure in Kaggle
Run this FIRST before training to understand your dataset layout
"""

import os
from pathlib import Path

print("="*80)
print("IMAGENET DATASET STRUCTURE CHECKER")
print("="*80)
print()

# Check common ImageNet paths in Kaggle
base_paths = [
    "/kaggle/input/imagenet-object-localization-challenge",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC",
    "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC",
]

for base in base_paths:
    if os.path.exists(base):
        print(f"✅ Found: {base}")
        print(f"   Contents:")
        try:
            items = os.listdir(base)
            for item in sorted(items)[:10]:  # Show first 10 items
                item_path = os.path.join(base, item)
                if os.path.isdir(item_path):
                    # Count subdirectories
                    try:
                        sub_count = len(os.listdir(item_path))
                        print(f"      📁 {item}/ ({sub_count} items)")
                    except:
                        print(f"      📁 {item}/")
                else:
                    print(f"      📄 {item}")
            if len(items) > 10:
                print(f"      ... and {len(items) - 10} more items")
        except Exception as e:
            print(f"   ❌ Error listing: {e}")
        print()
    else:
        print(f"❌ Not found: {base}")
        print()

# Check specific train/val structure
print("="*80)
print("CHECKING TRAIN/VAL STRUCTURE")
print("="*80)
print()

data_dir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"

if os.path.exists(data_dir):
    # Check train directory
    train_dir = os.path.join(data_dir, "train")
    if os.path.exists(train_dir):
        try:
            train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
            print(f"✅ Train directory: {train_dir}")
            print(f"   Number of class folders: {len(train_classes)}")
            if len(train_classes) > 0:
                # Sample a class to see how many images
                sample_class = train_classes[0]
                sample_path = os.path.join(train_dir, sample_class)
                sample_images = len(os.listdir(sample_path))
                print(f"   Sample class '{sample_class}': {sample_images} images")
                print(f"   ✅ GOOD: Train is organized by class folders")
        except Exception as e:
            print(f"❌ Error checking train: {e}")
    else:
        print(f"❌ Train directory not found: {train_dir}")
    print()

    # Check val directory
    val_dir = os.path.join(data_dir, "val")
    if os.path.exists(val_dir):
        try:
            val_items = os.listdir(val_dir)
            val_dirs = [d for d in val_items if os.path.isdir(os.path.join(val_dir, d))]
            val_files = [f for f in val_items if os.path.isfile(os.path.join(val_dir, f))]

            print(f"✅ Val directory: {val_dir}")
            print(f"   Subdirectories: {len(val_dirs)}")
            print(f"   Files: {len(val_files)}")

            if len(val_dirs) > 0:
                print(f"   ✅ GOOD: Val is organized by class folders ({len(val_dirs)} classes)")
            elif len(val_files) > 0:
                print(f"   ⚠️  WARNING: Val has flat structure ({len(val_files)} files)")
                print(f"   This needs organizing or we'll use train split for validation")

                # Show sample files
                print(f"\n   Sample files:")
                for f in sorted(val_files)[:5]:
                    print(f"      {f}")
        except Exception as e:
            print(f"❌ Error checking val: {e}")
    else:
        print(f"❌ Val directory not found: {val_dir}")
    print()
else:
    print(f"❌ Base data directory not found: {data_dir}")
    print()

# Provide recommendation
print("="*80)
print("RECOMMENDATION")
print("="*80)

if os.path.exists(os.path.join(data_dir, "train")):
    train_ok = True
else:
    train_ok = False

val_path = os.path.join(data_dir, "val")
if os.path.exists(val_path):
    try:
        val_dirs = [d for d in os.listdir(val_path) if os.path.isdir(os.path.join(val_path, d))]
        val_ok = len(val_dirs) > 0
    except:
        val_ok = False
else:
    val_ok = False

if train_ok and val_ok:
    print("✅ Dataset structure is GOOD!")
    print(f"   Use data_dir = '{data_dir}'")
    print("   Both train and val are properly organized")
elif train_ok and not val_ok:
    print("⚠️  Dataset structure needs adjustment")
    print(f"   Use data_dir = '{data_dir}'")
    print("   Training is OK, but validation will use 5% split from training data")
    print("   This is fine for validation purposes!")
else:
    print("❌ Dataset structure has issues")
    print("   Please check if you added ImageNet dataset as input in Kaggle")
    print("   Click '+ Add Input' → Search 'imagenet-object-localization-challenge'")

print("="*80)
