"""
ImageNet-1K AST Configuration Presets

Three configurations for different use cases:
1. Conservative: Publication-quality results (75-76% acc, 60% savings)
2. Aggressive: Balanced efficiency (73-75% acc, 70% savings)
3. Ultra: Maximum speedup (70-72% acc, 80% savings)

Developed by Oluwafemi Idiakhoa
"""

class ConfigConservative:
    """
    Configuration 1: Conservative (Publication Quality)

    Target: 75-76% accuracy, 60% energy savings
    Training Time: ~40 hours on V100
    Use Case: Publication, production deployment
    """
    # Dataset
    data_dir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    num_classes = 1000  # Full ImageNet-1K
    image_size = 224

    # Training
    batch_size = 256  # Larger for ImageNet-1K
    num_epochs = 100
    warmup_epochs = 10  # Adapt pretrained weights

    # Optimizer
    warmup_lr = 0.01
    ast_lr = 0.005
    weight_decay = 1e-4
    momentum = 0.9

    # AST settings (same as ImageNet-100)
    target_activation_rate = 0.40  # 60% energy savings
    initial_threshold = 3.0

    # PI Controller
    adapt_kp = 0.005
    adapt_ki = 0.0001
    ema_alpha = 0.1

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # Optimizations
    num_workers = 8
    pin_memory = True
    prefetch_factor = 2
    persistent_workers = True
    use_amp = True
    use_compile = False  # Disable for compatibility

    # System
    device = "cuda"

    # Logging
    log_interval = 100
    save_checkpoint_every = 10


class ConfigAggressive:
    """
    Configuration 2: Aggressive (Balanced Efficiency)

    Target: 73-75% accuracy, 70% energy savings
    Training Time: ~15 hours on V100
    Use Case: Rapid experimentation, resource-constrained
    """
    # Dataset
    data_dir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    num_classes = 1000
    image_size = 224

    # Training (shorter)
    batch_size = 256
    num_epochs = 50  # Half of conservative
    warmup_epochs = 5  # Quick adaptation

    # Optimizer (faster learning)
    warmup_lr = 0.02
    ast_lr = 0.01
    weight_decay = 1e-4
    momentum = 0.9

    # AST settings (more aggressive)
    target_activation_rate = 0.30  # 70% energy savings
    initial_threshold = 4.0

    # PI Controller (stronger gains)
    adapt_kp = 0.008
    adapt_ki = 0.00015
    ema_alpha = 0.1

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # Optimizations
    num_workers = 8
    pin_memory = True
    prefetch_factor = 2
    persistent_workers = True
    use_amp = True
    use_compile = False

    # System
    device = "cuda"

    # Logging
    log_interval = 100
    save_checkpoint_every = 5


class ConfigUltra:
    """
    Configuration 3: Ultra-Efficiency (Research/Rapid Iteration)

    Target: 70-72% accuracy, 80% energy savings
    Training Time: ~8 hours on V100
    Use Case: Quick validation, ablation studies
    """
    # Dataset
    data_dir = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
    num_classes = 1000
    image_size = 224

    # Training (minimal)
    batch_size = 256
    num_epochs = 30  # Very short
    warmup_epochs = 0  # No warmup, train from scratch

    # Optimizer (aggressive learning)
    warmup_lr = 0.03
    ast_lr = 0.015
    weight_decay = 1e-4
    momentum = 0.9

    # AST settings (extreme)
    target_activation_rate = 0.20  # 80% energy savings
    initial_threshold = 5.0

    # PI Controller (very strong)
    adapt_kp = 0.010
    adapt_ki = 0.00020
    ema_alpha = 0.1

    # Energy Model
    energy_per_activation = 1.0
    energy_per_skip = 0.01

    # Optimizations (max speed)
    num_workers = 8
    pin_memory = True
    prefetch_factor = 4  # More aggressive prefetch
    persistent_workers = True
    use_amp = True
    use_compile = False

    # Gradient accumulation for larger effective batch
    gradient_accumulation_steps = 4

    # System
    device = "cuda"

    # Logging
    log_interval = 50
    save_checkpoint_every = 3


# Helper function to get config by name
def get_config(config_name="conservative"):
    """
    Get configuration by name

    Args:
        config_name: "conservative", "aggressive", or "ultra"

    Returns:
        Config class instance
    """
    configs = {
        "conservative": ConfigConservative(),
        "aggressive": ConfigAggressive(),
        "ultra": ConfigUltra(),
    }

    if config_name.lower() not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(configs.keys())}")

    return configs[config_name.lower()]


# Configuration comparison
CONFIG_COMPARISON = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                     ImageNet-1K Configuration Comparison                     ║
╠══════════════════════════════════════════════════════════════════════════════╣
║ Metric              │ Conservative │ Aggressive   │ Ultra                   ║
╠═════════════════════╪══════════════╪══════════════╪═════════════════════════╣
║ Expected Accuracy   │ 75-76%       │ 73-75%       │ 70-72%                  ║
║ Energy Savings      │ 60%          │ 70%          │ 80%                     ║
║ Total Epochs        │ 100          │ 50           │ 30                      ║
║ Warmup Epochs       │ 10           │ 5            │ 0                       ║
║ Activation Rate     │ 40%          │ 30%          │ 20%                     ║
║ Training Time (V100)│ ~40 hours    │ ~15 hours    │ ~8 hours                ║
║ Speedup vs Baseline │ 1.9×         │ 3-4×         │ 6-8×                    ║
║ Use Case            │ Publication  │ Balanced     │ Research/Rapid          ║
╚═════════════════════╧══════════════╧══════════════╧═════════════════════════╝

Recommendation:
1. Start with Ultra (8 hours) to validate AST works on ImageNet-1K
2. If successful, run Conservative (40 hours) for publication-quality results
3. Use Aggressive for production with acceptable trade-offs
"""

if __name__ == "__main__":
    print(CONFIG_COMPARISON)

    print("\n" + "="*80)
    print("Configuration Details:")
    print("="*80)

    for name in ["conservative", "aggressive", "ultra"]:
        config = get_config(name)
        print(f"\n{name.upper()} Configuration:")
        print(f"  Epochs: {config.num_epochs} (warmup: {config.warmup_epochs})")
        print(f"  Target Activation: {config.target_activation_rate:.0%}")
        print(f"  Expected Energy Savings: {(1-config.target_activation_rate)*100:.0f}%")
        print(f"  Batch Size: {config.batch_size}")
        print(f"  Learning Rates: warmup={config.warmup_lr}, ast={config.ast_lr}")
