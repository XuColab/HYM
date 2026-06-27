import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置全局字体为 Times New Roman
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# SCI论文风格配色 (Nature/Science风格)
SCI_COLORS = {
    'target': '#4D4D4D',      # 深灰色
    'prediction': '#E69F00',   # 橙色
    'simulation': '#56B4E9',   # 天蓝色
    'XGB': '#009E73',          # 墨绿色
    'RF': '#D55E00',           # 红棕色
    'ET': '#0072B2',           # 深蓝色
    'error1': '#CC79A7',       # 粉紫色
    'error2': '#F0E442',       # 金黄色
    'error3': '#999999'        # 灰色
}

# 新数据
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'model': ['RF', 'RF', 'RF', 'RF', 'RF', 'RF', 'ET', 'ET', 'ET', 'ET', 'ET', 'ET'],
    'method': ['PSO', 'PSO', 'PSO', 'PSO', 'PSO', 'GA', 'PSO', 'PSO', 'PSO', 'PSO', 'GA', 'PSO'],
    'pitch': [1.073, 1.074, 1.072, 1.062, 1.096, 1.356, 0.949, 0.950, 0.949, 0.950, 0.839, 1.462],
    'a': [0.700, 0.702, 0.694, 0.701, 0.700, 0.725, 0.700, 0.700, 0.700, 0.700, 0.699, 0.700],
    'h': [1.271, 1.273, 1.254, 1.253, 1.252, 1.251, 1.253, 1.253, 1.253, 1.253, 1.254, 1.251],
    'predicted_y': [0.998754794, 0.998754794, 0.998754794, 0.998754794, 0.998754794, 
                    0.998495182, 0.994548176, 0.994548176, 0.994548176, 0.994548176, 
                    0.994384634, 0.99409917],
    'error_pre_1': [0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.005, 0.005, 0.005, 0.005, 0.006, 0.006],
    'Simulation': [0.999878802, 0.999813142, 0.99999023, 0.999575142, 0.999726561, 
                   0.996641403, 0.999376505, 0.999389336, 0.999376505, 0.999389336, 
                   0.999216588, 0.999901536],
    'err_pre_simu': [0.001, 0.001, 0.001, 0.001, 0.001, 0.002, 0.005, 0.005, 0.005, 0.005, 0.005, 0.006],
    'err_Simu_1': [0.00012, 0.00019, 0.00001, 0.00042, 0.00027, 0.00336, 
                   0.00062, 0.00061, 0.00062, 0.00061, 0.00078, 0.00010]
}

df = pd.DataFrame(data)

# 重命名自变量
df_renamed = df.rename(columns={
    'pitch': 'Period (μm)',
    'a': 'f',
    'h': 'h (μm)'
})

df_renamed['Combination'] = df_renamed['model'] + '-' + df_renamed['method']
df_renamed['ID_Label'] = df_renamed['ID'].astype(str) + ' (' + df_renamed['Combination'] + ')'

# 计算偏差
df_renamed['pred_bias'] = df_renamed['predicted_y'] - 1.0
df_renamed['sim_bias'] = df_renamed['Simulation'] - 1.0

# ============================================================
# Figure 1: 整合图 (条形图+折线图)
# ============================================================
fig, ax1 = plt.subplots(figsize=(14, 7))

df_sorted = df_renamed.sort_values('err_Simu_1').reset_index(drop=True)
df_sorted['ID_Label_Sorted'] = df_sorted['ID'].astype(str) + ' (' + df_sorted['Combination'] + ')'
x_pos = np.arange(len(df_sorted))
width = 0.25

# 条形图部分
target_values = [1.0] * len(df_sorted)
pred_values = df_sorted['predicted_y'].values
sim_values = df_sorted['Simulation'].values

bars1 = ax1.bar(x_pos - width, target_values, width, label='Target Value (1.0)', 
                color=SCI_COLORS['target'], edgecolor='black', alpha=0.6, linewidth=0.8)
bars2 = ax1.bar(x_pos, pred_values, width, label='ML Predicted Value', 
                color=SCI_COLORS['prediction'], edgecolor='black', alpha=0.7, linewidth=0.8)
bars3 = ax1.bar(x_pos + width, sim_values, width, label='Simulation Value', 
                color=SCI_COLORS['simulation'], edgecolor='black', alpha=0.7, linewidth=0.8)

# 折线图叠加
ax1.plot(x_pos, pred_values, 'o-', color=SCI_COLORS['prediction'], 
         linewidth=1.5, markersize=6, alpha=0.6, label='_nolegend_')
ax1.plot(x_pos, sim_values, 's-', color=SCI_COLORS['simulation'], 
         linewidth=1.5, markersize=6, alpha=0.6, label='_nolegend_')

# 添加误差箭头
for i, (pred, sim) in enumerate(zip(pred_values, sim_values)):
    if abs(pred - sim) > 0.001:
        ax1.annotate('', xy=(i + width/2, sim), xytext=(i + width/2, pred),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8, alpha=0.5))
        ax1.text(i + width/2, (pred+sim)/2, f'{abs(pred-sim):.4f}', 
                ha='center', va='center', fontsize=7, color='gray', style='italic')

ax1.set_xticks(x_pos)
ax1.set_xticklabels(df_sorted['ID_Label_Sorted'], rotation=45, ha='right', fontsize=8)
ax1.set_ylabel('Value', fontsize=12, fontweight='bold')
ax1.set_xlabel('Data Point ID (Model-Optimization Combination)', fontsize=12, fontweight='bold')
ax1.set_title('Target vs Prediction vs Simulation (Sorted by Simulation Error)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim(0.99, 1.005)
ax1.legend(fontsize=10, loc='upper right', frameon=True, fancybox=True, shadow=True)
ax1.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
ax1.axhline(y=1.0, color=SCI_COLORS['target'], linestyle='-', linewidth=1.5, alpha=0.8)
ax1.axhspan(0.998, 1.002, alpha=0.1, color=SCI_COLORS['target'])

plt.tight_layout()
plt.savefig('1_integrated_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# Figure 2: 偏差箱线图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 6))

bias_data = [df_renamed['pred_bias'].values, df_renamed['sim_bias'].values]
bp = ax.boxplot(bias_data, labels=['Prediction Bias\n(Pred - Target)', 'Simulation Bias\n(Sim - Target)'], 
                patch_artist=True, showmeans=True, meanline=True, 
                medianprops=dict(linewidth=2, color='black'),
                meanprops=dict(linewidth=2, color='red', linestyle='--'))

colors_bias = [SCI_COLORS['prediction'], SCI_COLORS['simulation']]
for patch, color in zip(bp['boxes'], colors_bias):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')
    patch.set_linewidth(1)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Bias Line')
ax.set_ylabel('Bias Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Bias Type', fontsize=12, fontweight='bold')
ax.set_title('Bias Distribution: Prediction vs Simulation Relative to Target', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

pred_mean = df_renamed['pred_bias'].mean()
pred_std = df_renamed['pred_bias'].std()
sim_mean = df_renamed['sim_bias'].mean()
sim_std = df_renamed['sim_bias'].std()
ax.text(0.65, 0.0015, f'Mean={pred_mean:.6f}\nStd={pred_std:.6f}', 
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
ax.text(1.65, 0.0015, f'Mean={sim_mean:.6f}\nStd={sim_std:.6f}', 
        ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig('2_bias_boxplot.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# Figure 3: 模拟误差柱状图
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

df_sorted = df_renamed.sort_values('err_Simu_1').reset_index(drop=True)
df_sorted['ID_Label_Sorted'] = df_sorted['ID'].astype(str) + ' (' + df_sorted['Combination'] + ')'
x_pos = np.arange(len(df_sorted))

colors_bar = [SCI_COLORS['RF'] if m == 'RF' else SCI_COLORS['ET'] for m in df_sorted['model']]
bars = ax.bar(x_pos, df_sorted['err_Simu_1'], color=colors_bar, edgecolor='black', alpha=0.8, linewidth=0.8)

ax.set_xticks(x_pos)
ax.set_xticklabels(df_sorted['ID_Label_Sorted'], rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Simulation Error (err_Simu_1)', fontsize=12, fontweight='bold')
ax.set_xlabel('Data Point ID (Model-Optimization Combination)', fontsize=12, fontweight='bold')
ax.set_title('Simulation Error Compared to Target Value (Lower is Better)', 
             fontsize=14, fontweight='bold')

for bar, val in zip(bars, df_sorted['err_Simu_1']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00005, 
            f'{val:.5f}', ha='center', va='bottom', fontsize=8, rotation=90)

ax.axhline(y=0.001, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
           label='Excellent Threshold (0.001)')
ax.axhline(y=0.002, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, 
           label='Good Threshold (0.002)')
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('3_simulation_error.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# Figure 4: 三种误差对比
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

combinations = df_renamed.groupby('Combination')[['error_pre_1', 'err_pre_simu', 'err_Simu_1']].mean()
combinations = combinations.sort_values('err_Simu_1')
x = np.arange(len(combinations))
width = 0.25

bars1 = ax.bar(x - width, combinations['error_pre_1'], width, 
               label='Prediction Error (vs Target)', color=SCI_COLORS['error1'], 
               edgecolor='black', alpha=0.7, linewidth=0.8)
bars2 = ax.bar(x, combinations['err_pre_simu'], width, 
               label='Pred-Sim Deviation', color=SCI_COLORS['error2'], 
               edgecolor='black', alpha=0.7, linewidth=0.8)
bars3 = ax.bar(x + width, combinations['err_Simu_1'], width, 
               label='Simulation Error (vs Target)', color=SCI_COLORS['error3'], 
               edgecolor='black', alpha=0.7, linewidth=0.8)

ax.set_xticks(x)
ax.set_xticklabels(combinations.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Model-Optimization Combination', fontsize=12, fontweight='bold')
ax.set_title('Three Types of Errors Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('4_three_errors_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# Figure 5: 模拟值vs目标值散点图
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

markers = {'RF': 'o', 'ET': 's'}
colors_model = {'RF': SCI_COLORS['RF'], 'ET': SCI_COLORS['ET']}

for model in df_renamed['model'].unique():
    subset = df_renamed[df_renamed['model'] == model]
    ax.scatter(subset['Simulation'], [1]*len(subset), 
               s=120, marker=markers[model], c=colors_model[model], 
               edgecolors='black', linewidth=1.2, alpha=0.8, label=model, zorder=5)
    for _, row in subset.iterrows():
        ax.annotate(f"{row['method']}", (row['Simulation'], 1), 
                   textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8)

ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Target Value (1.0)')
ax.axvspan(0.999, 1.001, alpha=0.2, color='green', label='Excellent Zone (±0.1%)')
ax.axvspan(0.998, 0.999, alpha=0.2, color='yellow', label='Good Zone (0.1%-0.2%)')
ax.axvspan(0.996, 0.998, alpha=0.2, color='orange', label='Acceptable Zone (0.2%-0.4%)')

ax.set_xlabel('Simulation Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Target Value', fontsize=12, fontweight='bold')
ax.set_title('Simulation Value Distribution (Closer to 1.0 is Better)', 
             fontsize=14, fontweight='bold')
ax.set_xlim(0.995, 1.001)
ax.set_ylim(0.998, 1.002)
ax.set_yticks([1])
ax.set_yticklabels(['Target = 1.0'], fontweight='bold')
ax.legend(fontsize=10, loc='upper left', frameon=True, fancybox=True)
ax.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('5_simulation_target_scatter.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# Figure 6: 参数vs误差散点图 (使用新名称)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

params = ['Period (μm)', 'f', 'h (μm)']
param_names = ['Period (μm)', 'f', 'h (μm)']

for i, (param, name) in enumerate(zip(params, param_names)):
    for model in df_renamed['model'].unique():
        subset = df_renamed[df_renamed['model'] == model]
        axes[i].scatter(subset[param], subset['err_Simu_1'], 
                       s=80, marker=markers[model], c=colors_model[model], 
                       edgecolors='black', linewidth=1, alpha=0.7, label=model)
    axes[i].set_xlabel(name, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Simulation Error (err_Simu_1)', fontsize=12, fontweight='bold')
    axes[i].set_title(f'{name} vs Simulation Error', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    axes[i].legend(fontsize=10, loc='best')

plt.tight_layout()
plt.savefig('6_params_vs_error.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# Figure 7: 雷达图
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

models = ['RF', 'ET']
metrics_data = {}
for model in models:
    subset = df_renamed[df_renamed['model'] == model]
    metrics_data[model] = {
        'err_Simu_1_mean': subset['err_Simu_1'].mean(),
        'err_Simu_1_std': subset['err_Simu_1'].std(),
        'max_err': subset['err_Simu_1'].max(),
        'success_rate_1e-3': (subset['err_Simu_1'] <= 0.001).sum() / len(subset)
    }

all_means = [metrics_data[m]['err_Simu_1_mean'] for m in models]
all_stds = [metrics_data[m]['err_Simu_1_std'] for m in models]
all_maxs = [metrics_data[m]['max_err'] for m in models]
all_success = [metrics_data[m]['success_rate_1e-3'] for m in models]

angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist()
angles += angles[:1]
metrics_names = ['Mean Error', 'Stability\n(1/Std)', 'Max Error', 'Excellence Rate']

colors_radar = [SCI_COLORS['RF'], SCI_COLORS['ET']]

for model, color in zip(models, colors_radar):
    values = [
        metrics_data[model]['err_Simu_1_mean'] / max(all_means) if max(all_means) > 0 else 0,
        1 - (metrics_data[model]['err_Simu_1_std'] / max(all_stds)) if max(all_stds) > 0 else 1,
        metrics_data[model]['max_err'] / max(all_maxs) if max(all_maxs) > 0 else 0,
        metrics_data[model]['success_rate_1e-3'] / max(all_success) if max(all_success) > 0 else 0
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model, markersize=6)
    ax.fill(angles, values, alpha=0.15, color=color)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics_names, fontsize=11, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('Model Comprehensive Performance Radar Chart\n(Larger Area Means Better Performance)', 
             fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=11)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('7_radar_chart.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# ============================================================
# 打印统计摘要
# ============================================================
print("=" * 80)
print("THREE-VALUE COMPARISON REPORT (Target vs Prediction vs Simulation)")
print("=" * 80)

print("\n[1. Bias Statistics]")
print(f"  Prediction Mean Bias: {df_renamed['pred_bias'].mean():.8f}")
print(f"  Prediction Bias Std: {df_renamed['pred_bias'].std():.8f}")
print(f"  Simulation Mean Bias: {df_renamed['sim_bias'].mean():.8f}")
print(f"  Simulation Bias Std: {df_renamed['sim_bias'].std():.8f}")

print("\n[2. Best Combination (Closest Simulation to Target)]")
best_sim = df_renamed.loc[df_renamed['err_Simu_1'].idxmin()]
print(f"  Combination: {best_sim['model']} + {best_sim['method']} (ID: {best_sim['ID']})")
print(f"  Prediction: {best_sim['predicted_y']:.9f} (Bias: {best_sim['pred_bias']:.9f})")
print(f"  Simulation: {best_sim['Simulation']:.9f} (Bias: {best_sim['sim_bias']:.9f})")
print(f"  Parameters: Period={best_sim['Period (μm)']:.3f}μm, f={best_sim['f']:.3f}, h={best_sim['h (μm)']:.3f}μm")

print("\n[3. Worst Combination (Farthest Simulation from Target)]")
worst_sim = df_renamed.loc[df_renamed['err_Simu_1'].idxmax()]
print(f"  Combination: {worst_sim['model']} + {worst_sim['method']} (ID: {worst_sim['ID']})")
print(f"  Prediction: {worst_sim['predicted_y']:.9f} (Bias: {worst_sim['pred_bias']:.9f})")
print(f"  Simulation: {worst_sim['Simulation']:.9f} (Bias: {worst_sim['sim_bias']:.9f})")

print("\n[4. Prediction-Simulation Consistency]")
correlation = np.corrcoef(df_renamed['predicted_y'], df_renamed['Simulation'])[0,1]
print(f"  Correlation Coefficient: {correlation:.6f}")
print(f"  Mean Pred-Sim Deviation: {df_renamed['err_pre_simu'].mean():.6f}")

print("\n[5. Model Ranking by Average Simulation Error]")
model_ranking = df_renamed.groupby('model')['err_Simu_1'].mean().sort_values()
for i, (model, err) in enumerate(model_ranking.items(), 1):
    medal = "🏆" if i == 1 else ("🥈" if i == 2 else ("🥉" if i == 3 else ""))
    print(f"  {i}. {model}: {err:.6f} {medal}")

print("\n[6. Parameters Range]")
print(f"  Period (μm): {df_renamed['Period (μm)'].min():.3f} - {df_renamed['Period (μm)'].max():.3f}")
print(f"  f: {df_renamed['f'].min():.3f} - {df_renamed['f'].max():.3f}")
print(f"  h (μm): {df_renamed['h (μm)'].min():.3f} - {df_renamed['h (μm)'].max():.3f}")

print("\n" + "=" * 80)
print("All figures have been saved successfully!")
print("=" * 80)