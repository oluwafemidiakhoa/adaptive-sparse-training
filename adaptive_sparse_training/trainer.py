"""
Adaptive Sparse Trainer - Main training loop with energy monitoring

Developed by Oluwafemi Idiakhoa
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from .sundew import SundewAlgorithm


class AdaptiveSparseTrainer:
    """
    Adaptive Sparse Training with automatic energy monitoring

    Integrates Sundew adaptive gating into your training loop with minimal changes.
    Supports mixed precision training (AMP) for additional speedup.

    Args:
        model (nn.Module): PyTorch model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        config (ASTConfig): AST configuration
        optimizer (torch.optim.Optimizer, optional): Optimizer (creates Adam if None)
        criterion (nn.Module, optional): Loss function (creates CrossEntropyLoss if None)

    Example:
        >>> from adaptive_sparse_training import AdaptiveSparseTrainer, ASTConfig
        >>> config = ASTConfig(target_activation_rate=0.40)
        >>> trainer = AdaptiveSparseTrainer(model, train_loader, val_loader, config)
        >>> results = trainer.train(epochs=100)
        >>> print(f"Energy Savings: {results['energy_savings']:.1f}%")
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        optimizer=None,
        criterion=None
    ):
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device

        # Create optimizer if not provided
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        # Create loss function if not provided
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss(reduction='none')  # Per-sample loss
        else:
            # Ensure criterion uses reduction='none' for per-sample losses
            self.criterion = criterion
            # Warn if criterion doesn't support reduction='none'
            if hasattr(criterion, 'reduction') and criterion.reduction != 'none':
                print(f"Warning: criterion.reduction is '{criterion.reduction}', expected 'none' for AST")

        # Initialize Sundew algorithm
        self.sundew = SundewAlgorithm(config)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.use_amp)

        # Tracking
        self.history = {
            'train_loss': [],
            'val_acc': [],
            'energy_savings': [],
            'activation_rates': [],
        }

    def train_epoch(self, epoch):
        """
        Train one epoch with adaptive sample selection

        Args:
            epoch (int): Current epoch number

        Returns:
            dict: Training statistics
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total_samples = 0
        total_active = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            batch_size = images.size(0)

            # Forward pass to compute losses (used for sample selection)
            with autocast(device_type='cuda', enabled=self.config.use_amp):
                with torch.no_grad():
                    outputs_selection = self.model(images)
                    losses = self.criterion(outputs_selection, labels)

            # Select important samples
            active_mask, energy_info = self.sundew.select_samples(losses, outputs_selection)
            num_active = energy_info['num_active']

            # Train only on selected samples (gradient masking)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=self.config.use_amp):
                # Create gradient mask
                mask_expanded = active_mask.unsqueeze(1).float()

                # Forward pass (reuses computation from selection)
                outputs = self.model(images)

                # Masked loss
                sample_losses = self.criterion(outputs, labels)
                masked_loss = (sample_losses * active_mask.float()).sum() / num_active

            # Backward pass
            self.scaler.scale(masked_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Track metrics
            running_loss += masked_loss.item() * num_active
            _, predicted = outputs[active_mask].max(1)
            correct += predicted.eq(labels[active_mask]).sum().item()
            total_samples += batch_size
            total_active += num_active

            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{masked_loss.item():.4f}',
                'Act': f'{energy_info["activation_rate"]:.1%}',
                'Save': f'{energy_info["energy_savings"]:.1f}%'
            })

        # Epoch statistics
        avg_loss = running_loss / total_active
        train_acc = 100.0 * correct / total_active
        activation_rate = total_active / total_samples
        energy_savings = energy_info['energy_savings']

        return {
            'loss': avg_loss,
            'train_acc': train_acc,
            'activation_rate': activation_rate,
            'energy_savings': energy_savings,
        }

    @torch.no_grad()
    def evaluate(self):
        """
        Evaluate model on validation set

        Returns:
            float: Validation accuracy
        """
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in self.val_loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            outputs = self.model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        return accuracy

    def train(self, epochs, warmup_epochs=0):
        """
        Full training loop with optional warmup

        Args:
            epochs (int): Total number of epochs
            warmup_epochs (int): Number of warmup epochs (trains on 100% of samples)

        Returns:
            dict: Final training results
        """
        print(f"Starting training: {epochs} epochs ({warmup_epochs} warmup + {epochs-warmup_epochs} AST)")
        print("=" * 70)

        for epoch in range(1, epochs + 1):
            # Warmup phase: disable AST (train on all samples)
            if epoch <= warmup_epochs:
                print(f"\n[WARMUP] Epoch {epoch}/{warmup_epochs}")
                # Temporarily set target to 100%
                original_target = self.sundew.target_activation_rate
                self.sundew.target_activation_rate = 1.0
                self.sundew.activation_threshold = 0.0  # Select all samples

            # Training
            train_stats = self.train_epoch(epoch)

            # Restore AST after warmup
            if epoch == warmup_epochs:
                self.sundew.target_activation_rate = original_target
                self.sundew.activation_threshold = self.config.initial_threshold
                print(f"\n[AST ACTIVATED] Target: {original_target:.0%} activation")

            # Validation
            val_acc = self.evaluate()

            # Log results
            print(f"Epoch {epoch}/{epochs} | "
                  f"Loss: {train_stats['loss']:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"Act: {train_stats['activation_rate']:.1%} | "
                  f"Save: {train_stats['energy_savings']:.1f}%")

            # Track history
            self.history['train_loss'].append(train_stats['loss'])
            self.history['val_acc'].append(val_acc)
            self.history['energy_savings'].append(train_stats['energy_savings'])
            self.history['activation_rates'].append(train_stats['activation_rate'])

        # Final results
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print(f"Final Validation Accuracy: {self.history['val_acc'][-1]:.2f}%")
        print(f"Final Energy Savings: {self.history['energy_savings'][-1]:.1f}%")
        print(f"Average Activation Rate: {sum(self.history['activation_rates'])/len(self.history['activation_rates']):.1%}")
        print("=" * 70)

        return {
            'final_accuracy': self.history['val_acc'][-1],
            'energy_savings': self.history['energy_savings'][-1],
            'history': self.history,
        }
