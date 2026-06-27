import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['RF', 'ET', 'XGB']
r2_values = [0.9184, 0.9387, 0.8805]
mse_values = [0.000915, 0.000688, 0.001341]
rmse_values = [0.0303, 0.0262, 0.0366]

# SCI论文常见配色 (来自Nature/Science等期刊的经典配色)
colors = {
    'RF': '#4C72B0',      # 沉稳蓝
    'ET': '#DD8452',      # 暖橙
    'XGB': '#55A868'      # 墨绿
}

# Set font properties
plt.rcParams['font.family'] = 'Arial'  # 设置全局字体为 Arial
plt.rcParams['font.weight'] = 'bold'

# Create figure with three subplots (1 row, 3 columns)
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# ==================== Subplot 1: R² ====================
bar_colors_r2 = [colors['RF'], colors['ET'], colors['XGB']]
bars1 = axes[0].bar(models, r2_values, color=bar_colors_r2, edgecolor='black', linewidth=0.8)

# Add reference line at 0.9
axes[0].axhline(y=0.9, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)

# Set labels and title
axes[0].set_ylabel('R²', fontweight='bold', fontsize=12)
axes[0].set_xlabel('Model', fontweight='bold', fontsize=12)
axes[0].set_title('(a) R² Score', fontweight='bold', fontsize=12, pad=10)

# Set y-axis limit
axes[0].set_ylim(0.85, 0.95)

# Add value labels on bars
for bar, value in zip(bars1, r2_values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0015,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# ==================== Subplot 2: MSE ====================
bars2 = axes[1].bar(models, mse_values, color=bar_colors_r2, edgecolor='black', linewidth=0.8)

# Set labels and title
axes[1].set_ylabel('MSE', fontweight='bold', fontsize=12)
axes[1].set_xlabel('Model', fontweight='bold', fontsize=12)
axes[1].set_title('(b) Mean Squared Error', fontweight='bold', fontsize=12, pad=10)

# Add value labels on bars
for bar, value in zip(bars2, mse_values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.00005,
             f'{value:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

# ==================== Subplot 3: RMSE ====================
bars3 = axes[2].bar(models, rmse_values, color=bar_colors_r2, edgecolor='black', linewidth=0.8)

# Set labels and title
axes[2].set_ylabel('RMSE', fontweight='bold', fontsize=12)
axes[2].set_xlabel('Model', fontweight='bold', fontsize=12)
axes[2].set_title('(c) Root Mean Squared Error', fontweight='bold', fontsize=12, pad=10)

# Add value labels on bars
for bar, value in zip(bars3, rmse_values):
    axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
             f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout()

# ==================== Save the figure ====================
# 保存为高分辨率PNG格式（适合论文使用）
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')

# 同时保存为PDF格式（矢量图，适合直接用于论文）
plt.savefig('model_performance_comparison.pdf', bbox_inches='tight', facecolor='white')

# 可选：保存为SVG格式（矢量图，可编辑）
plt.savefig('model_performance_comparison.svg', bbox_inches='tight', facecolor='white')

# 显示图片
plt.show()

print("图片已保存为以下格式：")
print("  - model_performance_comparison.png (300 DPI, 适合投稿)")
print("  - model_performance_comparison.pdf (矢量格式，推荐)")
print("  - model_performance_comparison.svg (矢量格式，可编辑)")