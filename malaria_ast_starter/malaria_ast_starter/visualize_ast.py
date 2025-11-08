"""
Visualization utilities for AST training metrics

Creates compelling visualizations showing energy savings, accuracy, and sample efficiency.
Perfect for presentations, papers, and social media posts!
"""

import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_metrics(jsonl_path):
    """Load metrics from JSONL file"""
    with open(jsonl_path, "r") as f:
        return [json.loads(line) for line in f]


def create_ast_comparison_plot(metrics, output_path="ast_results.png"):
    """
    Create a comprehensive 4-panel visualization of AST training

    Shows:
    1. Validation Accuracy over time
    2. Energy Savings percentage
    3. Samples Processed (absolute numbers)
    4. Training Loss
    """
    epochs = [m["epoch"] for m in metrics]
    val_acc = [m.get("val_acc", 0) * 100 for m in metrics]  # Convert to percentage
    energy_savings = [m.get("energy_savings", 0) for m in metrics]
    samples_processed = [m.get("samples_processed", m.get("total_samples", 0)) for m in metrics]
    total_samples = metrics[0].get("total_samples", max(samples_processed))
    train_loss = [m.get("train_loss", 0) for m in metrics]
    activation_rate = [m.get("activation_rate", 1.0) * 100 for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("🌿 Adaptive Sparse Training (AST) - Malaria Diagnostic AI",
                 fontsize=16, fontweight='bold')

    # Plot 1: Validation Accuracy
    ax1 = axes[0, 0]
    ax1.plot(epochs, val_acc, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.set_title("Model Accuracy", fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(val_acc) - 2, min(100, max(val_acc) + 2)])

    # Annotate best accuracy
    best_idx = np.argmax(val_acc)
    ax1.annotate(f'Best: {val_acc[best_idx]:.2f}%',
                xy=(epochs[best_idx], val_acc[best_idx]),
                xytext=(10, -10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

    # Plot 2: Energy Savings
    ax2 = axes[0, 1]
    ax2.plot(epochs, energy_savings, 'g-', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Energy Savings (%)", fontsize=11)
    ax2.set_title("Computational Energy Savings", fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 100])

    # Annotate average savings (excluding warmup)
    warmup_cutoff = 0
    for i, m in enumerate(metrics):
        if m.get("energy_savings", 0) > 0:
            warmup_cutoff = i
            break

    if warmup_cutoff < len(metrics):
        avg_savings = np.mean(energy_savings[warmup_cutoff:])
        ax2.axhline(y=avg_savings, color='r', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.text(epochs[-1] * 0.7, avg_savings + 5,
                f'Avg: {avg_savings:.1f}%',
                fontsize=10, color='r', fontweight='bold')

    # Plot 3: Samples Processed
    ax3 = axes[1, 0]
    ax3.bar(epochs, samples_processed, color='orange', alpha=0.7, label='Processed')
    ax3.axhline(y=total_samples, color='r', linestyle='--', linewidth=2,
               label=f'Total Available ({total_samples})')
    ax3.set_xlabel("Epoch", fontsize=11)
    ax3.set_ylabel("Number of Samples", fontsize=11)
    ax3.set_title("Sample Efficiency", fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # Plot 4: Training Loss
    ax4 = axes[1, 1]
    ax4.plot(epochs, train_loss, 'r-', linewidth=2, marker='^', markersize=4)
    ax4.set_xlabel("Epoch", fontsize=11)
    ax4.set_ylabel("Training Loss", fontsize=11)
    ax4.set_title("Training Loss Convergence", fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comprehensive visualization to: {output_path}")
    return output_path


def create_headline_graphic(metrics, output_path="ast_headline.png"):
    """
    Create a simple, bold graphic perfect for social media and press releases

    Shows:
    - Final accuracy
    - Average energy savings
    - Tagline
    """
    # Calculate key metrics
    final_acc = metrics[-1].get("val_acc", 0) * 100

    # Skip warmup epochs for energy savings calculation
    energy_metrics = [m for m in metrics if m.get("energy_savings", 0) > 0]
    if energy_metrics:
        avg_energy_savings = np.mean([m.get("energy_savings", 0) for m in energy_metrics])
    else:
        avg_energy_savings = 0

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('off')

    # Title
    fig.text(0.5, 0.85, "🌿 Energy-Efficient Malaria Detection",
            ha='center', fontsize=28, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='lightblue', alpha=0.8))

    # Main metrics
    fig.text(0.5, 0.65, f"{final_acc:.1f}%",
            ha='center', fontsize=72, fontweight='bold', color='blue')
    fig.text(0.5, 0.55, "Diagnostic Accuracy",
            ha='center', fontsize=20, color='darkblue')

    fig.text(0.5, 0.38, f"{avg_energy_savings:.0f}%",
            ha='center', fontsize=72, fontweight='bold', color='green')
    fig.text(0.5, 0.28, "Energy Savings vs. Traditional Training",
            ha='center', fontsize=20, color='darkgreen')

    # Tagline
    fig.text(0.5, 0.10, "AI-Powered Diagnostics for Low-Resource Clinical Settings",
            ha='center', fontsize=18, style='italic',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.6))

    # Footer
    fig.text(0.5, 0.02, "Powered by Adaptive Sparse Training (Sundew Algorithm)",
            ha='center', fontsize=12, color='gray')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved headline graphic to: {output_path}")
    return output_path


def print_summary_stats(metrics):
    """Print a summary of key statistics"""
    print("\n" + "="*80)
    print("📊 AST TRAINING SUMMARY STATISTICS")
    print("="*80)

    # Final metrics
    final = metrics[-1]
    print(f"\n🎯 Final Results:")
    print(f"  Validation Accuracy: {final.get('val_acc', 0)*100:.2f}%")
    print(f"  Training Loss: {final.get('train_loss', 0):.4f}")

    # Best accuracy
    best_acc = max(m.get('val_acc', 0) for m in metrics) * 100
    best_epoch = [m['epoch'] for m in metrics if m.get('val_acc', 0)*100 == best_acc][0]
    print(f"\n⭐ Best Performance:")
    print(f"  Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")

    # Energy savings (excluding warmup)
    energy_metrics = [m for m in metrics if m.get("energy_savings", 0) > 0]
    if energy_metrics:
        avg_savings = np.mean([m.get("energy_savings", 0) for m in energy_metrics])
        max_savings = max(m.get("energy_savings", 0) for m in energy_metrics)
        avg_activation = np.mean([m.get("activation_rate", 1.0) for m in energy_metrics])

        print(f"\n⚡ Energy Efficiency:")
        print(f"  Average Energy Savings: {avg_savings:.1f}%")
        print(f"  Maximum Energy Savings: {max_savings:.1f}%")
        print(f"  Average Activation Rate: {avg_activation*100:.1f}%")

        # Calculate total samples saved
        total_samples = final.get("total_samples", 0)
        total_epochs = len(energy_metrics)
        samples_saved = total_samples * total_epochs * (avg_savings / 100)
        print(f"  Total Samples Saved: {samples_saved:,.0f} / {total_samples * total_epochs:,.0f}")

    print("\n" + "="*80 + "\n")


def compare_ast_vs_baseline(ast_metrics_path, baseline_metrics_path=None, output_path="ast_vs_baseline.png"):
    """
    Compare AST training against baseline training

    If baseline_metrics_path is None, creates a simulated baseline assuming
    100% sample processing with similar accuracy
    """
    ast_metrics = load_metrics(ast_metrics_path)

    if baseline_metrics_path and Path(baseline_metrics_path).exists():
        baseline_metrics = load_metrics(baseline_metrics_path)
    else:
        # Simulate baseline (100% activation, no savings)
        print("ℹ️  No baseline metrics found, simulating baseline with 100% activation")
        baseline_metrics = [
            {
                "epoch": m["epoch"],
                "val_acc": m.get("val_acc", 0),
                "energy_savings": 0,
                "activation_rate": 1.0,
                "samples_processed": m.get("total_samples", 27558)
            }
            for m in ast_metrics
        ]

    # Extract data
    epochs = [m["epoch"] for m in ast_metrics]

    ast_acc = [m.get("val_acc", 0) * 100 for m in ast_metrics]
    baseline_acc = [m.get("val_acc", 0) * 100 for m in baseline_metrics]

    ast_energy = [m.get("energy_savings", 0) for m in ast_metrics]

    # Create comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("AST vs. Baseline Comparison", fontsize=16, fontweight='bold')

    # Accuracy comparison
    ax1.plot(epochs, baseline_acc, 'r--', linewidth=2, label='Baseline (100% samples)', marker='o')
    ax1.plot(epochs, ast_acc, 'b-', linewidth=2, label='AST (adaptive)', marker='s')
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Validation Accuracy (%)", fontsize=11)
    ax1.set_title("Accuracy Comparison", fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Energy savings
    ax2.bar(epochs, ast_energy, color='green', alpha=0.7)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Energy Savings (%)", fontsize=11)
    ax2.set_title("AST Energy Savings", fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 100])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize AST training results")
    parser.add_argument("--metrics", default="checkpoints_ast/metrics_ast.jsonl",
                       help="Path to AST metrics JSONL file")
    parser.add_argument("--baseline-metrics", default=None,
                       help="Path to baseline metrics JSONL file (optional)")
    parser.add_argument("--output-dir", default="visualizations",
                       help="Output directory for visualizations")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load metrics
    if not Path(args.metrics).exists():
        print(f"❌ Error: Metrics file not found: {args.metrics}")
        print("   Please run training first: python train_ast.py")
        exit(1)

    metrics = load_metrics(args.metrics)

    # Print summary
    print_summary_stats(metrics)

    # Create visualizations
    print("📊 Generating visualizations...")

    create_ast_comparison_plot(metrics, output_path=str(output_dir / "ast_results.png"))
    create_headline_graphic(metrics, output_path=str(output_dir / "ast_headline.png"))

    if args.baseline_metrics:
        compare_ast_vs_baseline(args.metrics, args.baseline_metrics,
                               output_path=str(output_dir / "ast_vs_baseline.png"))

    print(f"\n🎉 All visualizations saved to: {output_dir}/")
    print("\nReady for presentations, papers, and social media! 🚀")
