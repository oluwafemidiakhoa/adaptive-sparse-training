#!/usr/bin/env python3
"""
Create eye-catching social media visual for AST announcement.
Clean, professional design for Twitter/LinkedIn.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig = plt.figure(figsize=(12, 8), facecolor='white')
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(5, 9.2, 'Adaptive Sparse Training',
        ha='center', va='top', fontsize=32, fontweight='bold', color='#1a1a1a')
ax.text(5, 8.7, 'Energy-Efficient Deep Learning with Sundew Gating',
        ha='center', va='top', fontsize=16, color='#4a4a4a')

# Divider line
ax.plot([1, 9], [8.4, 8.4], 'k-', linewidth=2, alpha=0.3)

# ============================================================================
# KEY RESULTS (Big metrics)
# ============================================================================
results_y = 7.5
box_width = 1.8
box_height = 1.2
spacing = 0.1

metrics = [
    {"value": "61.2%", "label": "Accuracy", "color": "#2ecc71"},
    {"value": "89.6%", "label": "Energy\nSavings", "color": "#3498db"},
    {"value": "11.5×", "label": "Speedup", "color": "#e74c3c"},
    {"value": "10.4%", "label": "Activation\nRate", "color": "#f39c12"},
]

start_x = 5 - (len(metrics) * (box_width + spacing)) / 2 + box_width / 2

for i, metric in enumerate(metrics):
    x = start_x + i * (box_width + spacing)

    # Box background
    box = FancyBboxPatch(
        (x - box_width/2, results_y - box_height), box_width, box_height,
        boxstyle="round,pad=0.1",
        facecolor=metric['color'],
        edgecolor='none',
        alpha=0.15
    )
    ax.add_patch(box)

    # Value (big)
    ax.text(x, results_y - 0.3, metric['value'],
            ha='center', va='center', fontsize=28, fontweight='bold',
            color=metric['color'])

    # Label (small)
    ax.text(x, results_y - 0.85, metric['label'],
            ha='center', va='center', fontsize=12, color='#4a4a4a')

# ============================================================================
# COMPARISON (Traditional vs AST)
# ============================================================================
comp_y = 5.2
ax.text(5, comp_y + 0.5, 'Traditional vs AST Training',
        ha='center', va='bottom', fontsize=18, fontweight='bold', color='#1a1a1a')

# Traditional (left)
trad_x = 2.5
ax.text(trad_x, comp_y, 'Traditional Training',
        ha='center', va='top', fontsize=14, fontweight='bold', color='#e74c3c')
ax.text(trad_x, comp_y - 0.4, '50,000 samples/epoch',
        ha='center', va='top', fontsize=11, color='#4a4a4a')
ax.text(trad_x, comp_y - 0.7, '100% energy',
        ha='center', va='top', fontsize=11, color='#4a4a4a')
ax.text(trad_x, comp_y - 1.0, '120 min training',
        ha='center', va='top', fontsize=11, color='#4a4a4a')
ax.text(trad_x, comp_y - 1.3, '~60% accuracy',
        ha='center', va='top', fontsize=11, color='#4a4a4a')

# Arrow
ax.annotate('', xy=(6.5, comp_y - 0.65), xytext=(3.5, comp_y - 0.65),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2ecc71'))
ax.text(5, comp_y - 0.4, '90% savings', ha='center', va='bottom',
        fontsize=11, fontweight='bold', color='#2ecc71')

# AST (right)
ast_x = 7.5
ax.text(ast_x, comp_y, 'AST (This Work)',
        ha='center', va='top', fontsize=14, fontweight='bold', color='#2ecc71')
ax.text(ast_x, comp_y - 0.4, '5,200 samples/epoch',
        ha='center', va='top', fontsize=11, color='#4a4a4a')
ax.text(ast_x, comp_y - 0.7, '10.4% energy',
        ha='center', va='top', fontsize=11, color='#4a4a4a')
ax.text(ast_x, comp_y - 1.0, '10.5 min training',
        ha='center', va='top', fontsize=11, color='#4a4a4a')
ax.text(ast_x, comp_y - 1.3, '61.2% accuracy',
        ha='center', va='top', fontsize=11, color='#4a4a4a')

# Divider
ax.plot([1, 9], [3.2, 3.2], 'k-', linewidth=1, alpha=0.2)

# ============================================================================
# KEY INNOVATIONS
# ============================================================================
innov_y = 2.8
ax.text(5, innov_y, 'Key Innovations',
        ha='center', va='top', fontsize=16, fontweight='bold', color='#1a1a1a')

innovations = [
    "• EMA-smoothed PI controller",
    "• Batched GPU operations (50,000x faster)",
    "• Real-time energy monitoring",
    "• Production-ready (850 lines PyTorch)"
]

for i, innov in enumerate(innovations):
    y = innov_y - 0.5 - i * 0.35
    ax.text(5, y, innov, ha='center', va='top', fontsize=11, color='#4a4a4a')

# ============================================================================
# FOOTER
# ============================================================================
footer_y = 0.4
ax.text(5, footer_y, 'github.com/oluwafemidiakhoa/adaptive-sparse-training',
        ha='center', va='center', fontsize=13, fontweight='bold',
        color='#3498db', style='italic')
ax.text(5, footer_y - 0.35, 'Open Source • MIT License • Production Ready',
        ha='center', va='center', fontsize=10, color='#7f8c8d')

# ============================================================================
# SAVE
# ============================================================================
plt.tight_layout()
plt.savefig('AST_Social_Media_Visual.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("[OK] Created: AST_Social_Media_Visual.png (high-res for Twitter/LinkedIn)")

# Also create a compact version for Twitter
fig2 = plt.figure(figsize=(10, 6), facecolor='white')
ax2 = fig2.add_subplot(111)
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 6)
ax2.axis('off')

# Title
ax2.text(5, 5.6, 'Adaptive Sparse Training Results',
         ha='center', va='top', fontsize=24, fontweight='bold', color='#1a1a1a')

# Metrics (compact)
comp_metrics_y = 4.5
box_w = 1.6
box_h = 1.0
spacing_comp = 0.15

start_x2 = 5 - (len(metrics) * (box_w + spacing_comp)) / 2 + box_w / 2

for i, metric in enumerate(metrics):
    x = start_x2 + i * (box_w + spacing_comp)

    box = FancyBboxPatch(
        (x - box_w/2, comp_metrics_y - box_h), box_w, box_h,
        boxstyle="round,pad=0.08",
        facecolor=metric['color'],
        edgecolor='none',
        alpha=0.15
    )
    ax2.add_patch(box)

    ax2.text(x, comp_metrics_y - 0.25, metric['value'],
             ha='center', va='center', fontsize=22, fontweight='bold',
             color=metric['color'])

    ax2.text(x, comp_metrics_y - 0.7, metric['label'],
             ha='center', va='center', fontsize=10, color='#4a4a4a')

# Key stat
ax2.text(5, 3.0, 'CIFAR-10: 10.5 min vs 120 min baseline (11.5× speedup)',
         ha='center', va='center', fontsize=14, color='#2c3e50', fontweight='bold')

# Innovations (compact)
ax2.text(5, 2.3, 'Production-ready PyTorch • EMA-smoothed PI control • Real-time energy monitoring',
         ha='center', va='center', fontsize=11, color='#4a4a4a')

# GitHub
ax2.text(5, 1.5, 'github.com/oluwafemidiakhoa/adaptive-sparse-training',
         ha='center', va='center', fontsize=12, fontweight='bold',
         color='#3498db')

# Footer
ax2.text(5, 0.8, 'Open Source • MIT License',
         ha='center', va='center', fontsize=10, color='#7f8c8d')

# Tagline
ax2.text(5, 0.3, '90% Energy Savings • Green AI • Sustainable Deep Learning',
         ha='center', va='center', fontsize=11, fontweight='bold', color='#27ae60')

plt.tight_layout()
plt.savefig('AST_Twitter_Card.png', dpi=300, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print("[OK] Created: AST_Twitter_Card.png (optimized for Twitter card)")

plt.show()
print("\n[READY] Ready to post on Twitter and LinkedIn!")
print("[IMG] Attach AST_Twitter_Card.png to Tweet 4")
print("[IMG] Attach AST_Social_Media_Visual.png to LinkedIn post")
