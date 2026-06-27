import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 数据
data = {
    'model': ['XGB', 'XGB', 'RF', 'RF', 'RF', 'RF', 'RF', 'XGB', 'XGB', 'XGB', 'RF', 'ET', 'ET', 'ET', 'ET', 'ET', 'ET'],
    'method': ['PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'GA', 'PSO', 'PSO', 'PSO', 'PSO', 'GA', 'PSO'],
    'pitch': [2.857, 2.842, 1.073, 1.074, 1.072, 1.062, 1.096, 0.567, 0.541, 0.540, 1.356, 0.949, 0.950, 0.949, 0.950, 0.839, 1.462],
    'a': [0.755, 0.746, 0.700, 0.702, 0.694, 0.701, 0.700, 0.702, 0.718, 0.712, 0.725, 0.700, 0.700, 0.700, 0.700, 0.699, 0.700],
    'h': [1.283, 1.389, 1.271, 1.273, 1.254, 1.253, 1.252, 1.349, 1.310, 1.324, 1.251, 1.253, 1.253, 1.253, 1.253, 1.254, 1.251],
    'predicted_y': [1.000, 1.000, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.999, 0.998, 0.995, 0.995, 0.995, 0.995, 0.994, 0.994],
    'error': [0.000, 0.000, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.005, 0.005, 0.005, 0.005, 0.006, 0.006]
}

df = pd.DataFrame(data)

# 创建组合标签
df['组合'] = df['model'] + '-' + df['method']

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 图1：误差对比柱状图（按组合分组）
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# 计算各组合的平均误差
avg_error = df.groupby('组合')['error'].mean().sort_values()
colors = ['#2E86AB' if 'XGB' in x else '#A23B72' if 'RF' in x else '#F18F01' for x in avg_error.index]

bars = ax.barh(range(len(avg_error)), avg_error.values, color=colors, edgecolor='black', alpha=0.8)
ax.set_yticks(range(len(avg_error)))
ax.set_yticklabels(avg_error.index, fontsize=10)
ax.set_xlabel('平均误差 (error)', fontsize=12, fontweight='bold')
ax.set_title('各模型-优化方法组合的平均误差对比', fontsize=14, fontweight='bold')

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars, avg_error.values)):
    ax.text(val + 0.0002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', 
            va='center', fontsize=10, fontweight='bold')

ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('1_error_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 图2：预测参数分布散点图（pitch vs h，颜色表示误差）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 8))

scatter = ax.scatter(df['pitch'], df['h'], c=df['error'], s=100, 
                      cmap='RdYlGn_r', edgecolors='black', linewidth=1.5, alpha=0.8)

# 添加标签
for i, row in df.iterrows():
    ax.annotate(f"{row['model']}-{row['method']}", 
                (row['pitch'], row['h']), 
                textcoords="offset points", xytext=(5,5), 
                fontsize=8, alpha=0.7)

ax.set_xlabel('pitch', fontsize=12, fontweight='bold')
ax.set_ylabel('h', fontsize=12, fontweight='bold')
ax.set_title('反向预测参数分布 (pitch vs h，颜色=误差)', fontsize=14, fontweight='bold')

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('误差 (error)', fontsize=11)

ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_parameter_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 图3：模型-优化方法组合的误差箱线图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

# 准备数据
combinations = df['组合'].unique()
error_data = [df[df['组合'] == comb]['error'].values for comb in combinations]
positions = range(len(combinations))

bp = ax.boxplot(error_data, positions=positions, widths=0.6, 
                patch_artist=True, showmeans=True, meanline=True)

# 设置颜色
colors_box = ['#2E86AB' if 'XGB' in c else '#A23B72' if 'RF' in c else '#F18F01' for c in combinations]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_xticks(positions)
ax.set_xticklabels(combinations, rotation=45, ha='right', fontsize=10)
ax.set_ylabel('误差 (error)', fontsize=12, fontweight='bold')
ax.set_title('各组合误差分布箱线图', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('3_error_boxplot.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 图4：分组柱状图 - 按模型分组对比不同优化方法
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

models = ['XGB', 'RF', 'ET']
methods = ['PSO', 'GA']
x = np.arange(len(models))
width = 0.35

# 计算平均误差
pso_errors = []
ga_errors = []
for model in models:
    pso_err = df[(df['model'] == model) & (df['method'] == 'PSO')]['error'].mean()
    ga_err = df[(df['model'] == model) & (df['method'] == 'GA')]['error'].mean()
    pso_errors.append(pso_err if not np.isnan(pso_err) else 0)
    ga_errors.append(ga_err if not np.isnan(ga_err) else 0)

bars1 = ax.bar(x - width/2, pso_errors, width, label='PSO', color='#4ECDC4', edgecolor='black', alpha=0.8)
bars2 = ax.bar(x + width/2, ga_errors, width, label='GA', color='#FF6B6B', edgecolor='black', alpha=0.8)

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylabel('平均误差 (error)', fontsize=12, fontweight='bold')
ax.set_title('不同优化方法在各模型上的平均误差对比', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

# 添加数值标签
for bar in bars1:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002, 
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)
for bar in bars2:
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0002, 
                f'{bar.get_height():.4f}', ha='center', va='bottom', fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('4_method_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 图5：预测参数三维散点图 (pitch, a, h)
# ============================================================
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# 按模型分配颜色
color_map = {'XGB': '#2E86AB', 'RF': '#A23B72', 'ET': '#F18F01'}
colors_3d = [color_map[m] for m in df['model']]

scatter = ax.scatter(df['pitch'], df['a'], df['h'], 
                     c=df['error'], cmap='RdYlGn_r', s=80, 
                     edgecolors='black', linewidth=0.5, alpha=0.8)

# 添加标签
for i, row in df.iterrows():
    ax.text(row['pitch'], row['a'], row['h'], 
            f"{row['model']}-{row['method']}", 
            fontsize=7, alpha=0.6)

ax.set_xlabel('pitch', fontsize=11, fontweight='bold')
ax.set_ylabel('a', fontsize=11, fontweight='bold')
ax.set_zlabel('h', fontsize=11, fontweight='bold')
ax.set_title('反向预测参数三维分布 (颜色=误差)', fontsize=14, fontweight='bold')

cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
cbar.set_label('误差 (error)', fontsize=10)

plt.tight_layout()
plt.savefig('5_3d_parameters.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================
# 打印统计总结
# ============================================================
print("=" * 60)
print("反向预测结果统计总结")
print("=" * 60)

print("\n【各组合平均误差排名（由低到高）】")
avg_error_sorted = df.groupby('组合')['error'].mean().sort_values()
for i, (comb, err) in enumerate(avg_error_sorted.items(), 1):
    medal = "🥇" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else ""))
    print(f"  {i}. {comb}: {err:.4f} {medal}")

print("\n【各模型平均误差】")
for model in models:
    model_avg = df[df['model'] == model]['error'].mean()
    print(f"  {model}: {model_avg:.4f}")

print("\n【各优化方法平均误差】")
for method in ['PSO', 'GA']:
    method_avg = df[df['method'] == method]['error'].mean()
    print(f"  {method}: {method_avg:.4f}")

print("\n【最佳预测组合】")
best = df.loc[df['error'].idxmin()]
print(f"  模型: {best['model']}, 方法: {best['method']}")
print(f"  预测参数: pitch={best['pitch']:.3f}, a={best['a']:.3f}, h={best['h']:.3f}")
print(f"  误差: {best['error']:.4f}")

print("\n" + "=" * 60)