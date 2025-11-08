"""
Test script to verify the warmup_epochs bug fix
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Import local version of AST
import sys
sys.path.insert(0, r'c:\Users\adminidiakhoa\deepseek_physical_ai_sundew')

from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig

# Create dummy dataset
torch.manual_seed(42)
X_train = torch.randn(100, 3, 32, 32)
y_train = torch.randint(0, 10, (100,))
X_val = torch.randn(20, 3, 32, 32)
y_val = torch.randint(0, 10, (20,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)

model = SimpleModel()

# Test with warmup_epochs > 0 (this should trigger the bug before fix)
print("=" * 70)
print("Testing AST with warmup_epochs=2")
print("=" * 70)

config = ASTConfig(
    target_activation_rate=0.40,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

trainer = AdaptiveSparseTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)

try:
    results = trainer.train(epochs=3, warmup_epochs=2)
    print("\n" + "=" * 70)
    print("TEST PASSED: warmup_epochs bug is fixed!")
    print(f"Final Accuracy: {results['final_accuracy']:.2f}%")
    print(f"Energy Savings: {results['energy_savings']:.1f}%")
    print("=" * 70)
except Exception as e:
    print("\n" + "=" * 70)
    print("TEST FAILED:")
    print(f"Error: {e}")
    print("=" * 70)
    import traceback
    traceback.print_exc()
