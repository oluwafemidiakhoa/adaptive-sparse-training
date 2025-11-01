"""
Configuration for Adaptive Sparse Training

Developed by Oluwafemi Idiakhoa
"""

class ASTConfig:
    """
    Configuration for Adaptive Sparse Training

    Args:
        target_activation_rate (float): Target percentage of samples to process per epoch (0.1 = 10%)
        initial_threshold (float): Starting threshold for sample selection
        adapt_kp (float): Proportional gain for PI controller
        adapt_ki (float): Integral gain for PI controller
        ema_alpha (float): EMA smoothing factor for activation rate (0-1)
        energy_per_activation (float): Energy cost per activated sample
        energy_per_skip (float): Energy cost per skipped sample
        use_amp (bool): Use automatic mixed precision training
        device (str): Device to use ('cuda' or 'cpu')

    Example:
        >>> config = ASTConfig(target_activation_rate=0.40)  # 40% activation, 60% savings
        >>> config = ASTConfig(target_activation_rate=0.10)  # 10% activation, 90% savings
    """

    def __init__(
        self,
        target_activation_rate=0.40,
        initial_threshold=3.0,
        adapt_kp=0.005,
        adapt_ki=0.0001,
        ema_alpha=0.1,
        energy_per_activation=1.0,
        energy_per_skip=0.01,
        use_amp=True,
        device="cuda"
    ):
        self.target_activation_rate = target_activation_rate
        self.initial_threshold = initial_threshold
        self.adapt_kp = adapt_kp
        self.adapt_ki = adapt_ki
        self.ema_alpha = ema_alpha
        self.energy_per_activation = energy_per_activation
        self.energy_per_skip = energy_per_skip
        self.use_amp = use_amp
        self.device = device

    def __repr__(self):
        return (f"ASTConfig(target_activation_rate={self.target_activation_rate}, "
                f"adapt_kp={self.adapt_kp}, adapt_ki={self.adapt_ki})")
