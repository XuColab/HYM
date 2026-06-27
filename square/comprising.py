import matplotlib.pyplot as plt
import numpy as np

# 模型数据
models = ['RF', 'ET', 'XGB']
r2_scores = [0.9184, 0.9387, 0.8805]
mse_scores = [0.000915, 0.000688, 0.001341]
rmse_scores = [0.0303, 0.0262, 0.0366]

# 设置中文字体（避免中文显示问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图形，包含三个子图（1行3列）
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# 颜色设置
colors = ['#2E86AB', '#A23B72', '#F18F01']

# ==================== 子图1：R²对比 ====================
bars1 = axes[0].bar(models, r2_scores, color=colors, edgecolor='black', linewidth=1.2)
axes[0].set_ylim(0.85, 0.95)
axes[0].set_ylabel('R² (决定系数)', fontsize=12, fontweight='bold')
axes[0].set_title('模型 R² 性能对比', fontsize=13, fontweight='bold')
axes[0].axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.7, label='参考线 (0.9)')
# 添加数值标签
for bar, val in zip(bars1, r2_scores):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# ==================== 子图2：MSE对比（越小越好） ====================
bars2 = axes[1].bar(models, mse_scores, color=colors, edgecolor='black', linewidth=1.2)
axes[1].set_ylabel('MSE (均方误差)', fontsize=12, fontweight='bold')
axes[1].set_title('模型 MSE 性能对比', fontsize=13, fontweight='bold')
# 添加数值标签
for bar, val in zip(bars2, mse_scores):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002, 
                 f'{val:.6f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# ==================== 子图3：RMSE对比（越小越好） ====================
bars3 = axes[2].bar(models, rmse_scores, color=colors, edgecolor='black', linewidth=1.2)
axes[2].set_ylabel('RMSE (均方根误差)', fontsize=12, fontweight='bold')
axes[2].set_title('模型 RMSE 性能对比', fontsize=13, fontweight='bold')
# 添加数值标签
for bar, val in zip(bars3, rmse_scores):
    axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005, 
                 f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
axes[2].grid(axis='y', alpha=0.3)

# 整体标题
fig.suptitle('用于透过率预测模型性能对比', fontsize=16, fontweight='bold', y=1.02)

# 调整布局
plt.tight_layout()

# 保存图片（可选）
plt.savefig('model_performance_comparison.png', dpi=300, bbox_inches='tight')

# 显示图形
plt.show()

# ==================== 额外：综合对比雷达图 ====================
# 由于三个指标量纲不同且原始值差异大，统一归一化到0-1区间（越小越好）
# R²需要转换为 1-R² 使其方向与其他指标一致（越小越好）
r2_neg = [1 - r2 for r2 in r2_scores]
mse_norm = [mse / max(mse_scores) for mse in mse_scores]
rmse_norm = [rmse / max(rmse_scores) for rmse in rmse_scores]

# 创建雷达图
fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
ax2.set_theta_offset(np.pi / 2)
ax2.set_theta_direction(-1)

# 设置角度和标签
angles = np.linspace(0, 2 * np.pi, 4, endpoint=False).tolist()  # 3个指标，角度位置
angles += angles[:1]  # 闭合
metrics = ['1-R² (越小越好)', 'MSE (归一化)', 'RMSE (归一化)']

# 为每个模型绘制雷达图
for i, (model, r2n, msen, rmsen) in enumerate(zip(models, r2_neg, mse_norm, rmse_norm)):
    values = [r2n, msen, rmsen]
    values += values[:1]  # 闭合
    ax2.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
    ax2.fill(angles, values, alpha=0.15, color=colors[i])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(metrics, fontsize=11)
ax2.set_ylim(0, 1)
ax2.set_title('模型综合性能雷达图（归一化后，面积越小性能越优）', fontsize=14, fontweight='bold', pad=20)
ax2.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_radar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ==================== 打印总结 ====================
print("=" * 50)
print("模型性能对比总结")
print("=" * 50)
print(f"{'模型':<8} {'R²':<10} {'MSE':<12} {'RMSE':<10} {'综合排名'}")
print("-" * 50)
ranking = np.argsort([-r2_scores[0], -r2_scores[1], -r2_scores[2]])  # 按R²降序排名
for i, idx in enumerate(ranking):
    rank_icon = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")
    print(f"{models[idx]:<8} {r2_scores[idx]:<10.4f} {mse_scores[idx]:<12.6f} {rmse_scores[idx]:<10.4f} {rank_icon}")
print("=" * 50)
print(f"最佳模型: ET (R²={r2_scores[1]:.4f}, MSE={mse_scores[1]:.6f}, RMSE={rmse_scores[1]:.4f})")
print("=" * 50)