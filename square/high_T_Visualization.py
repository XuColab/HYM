import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ======================
# 全局设置
# ======================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.labelweight'] = 'bold'   # 坐标轴标签加粗
plt.rcParams['axes.titleweight'] = 'bold'   # 标题加粗

# SCI论文风格配色
SCI_COLORS = {
    'target': '#4D4D4D',      # 深灰
    'prediction': '#E69F00',   # 橙
    'simulation': '#56B4E9',   # 天蓝
    'XGB': '#009E73',          # 墨绿
    'RF': '#D55E00',           # 红棕
    'ET': '#0072B2',           # 深蓝
    'error1': '#CC79A7',       # 粉紫
    'error2': '#F0E442',       # 金黄
    'error3': '#999999'        # 灰
}

# ======================
# 1. 新数据
# ======================
data = {
    'ID': [1,2,3,4,5,6,7,8,9,10,11,12],
    'model': ['RF','RF','RF','RF','RF','RF','ET','ET','ET','ET','ET','ET'],
    'method': ['PSO','PSO','PSO','PSO','PSO','GA','PSO','PSO','PSO','PSO','GA','PSO'],
    'pitch': [1.073,1.074,1.072,1.062,1.096,1.356,0.949,0.950,0.949,0.950,0.839,1.462],
    'a': [0.700,0.702,0.694,0.701,0.700,0.725,0.700,0.700,0.700,0.700,0.699,0.700],
    'h': [1.271,1.273,1.254,1.253,1.252,1.251,1.253,1.253,1.253,1.253,1.254,1.251],
    'predicted_y': [0.998754794,0.998754794,0.998754794,0.998754794,0.998754794,
                    0.998495182,0.994548176,0.994548176,0.994548176,0.994548176,
                    0.994384634,0.99409917],
    'error_pre_1': [0.001,0.001,0.001,0.001,0.001,0.002,0.005,0.005,0.005,0.005,0.006,0.006],
    'Simulation': [0.999878802,0.999813142,0.99999023,0.999575142,0.999726561,
                   0.996641403,0.999376505,0.999389336,0.999376505,0.999389336,
                   0.999216588,0.999901536],
    'err_pre_simu': [0.001,0.001,0.001,0.001,0.001,0.002,0.005,0.005,0.005,0.005,0.005,0.006],
    'err_Simu_1': [0.00012,0.00019,0.00001,0.00042,0.00027,0.00336,
                   0.00062,0.00061,0.00062,0.00061,0.00078,0.00010]
}
df = pd.DataFrame(data)

# 重命名自变量列
df.rename(columns={'pitch':'Period (μm)', 'a':'f', 'h':'h (μm)'}, inplace=True)

# 组合名称
df['Combination'] = df['model'] + '-' + df['method']
# 偏差
df['pred_bias'] = df['predicted_y'] - 1.0
df['sim_bias'] = df['Simulation'] - 1.0

# ======================
# 2. 原始数据（用于平行坐标对比，即之前41个高T点）
# ======================
data_original = """pitch	a	h	T
0.5	0.65	1.15	0.995611285
0.5	0.65	1.2	0.998410879
0.5	0.65	1.25	0.996354967
0.5	0.7	1.25	0.998000483
0.5	0.7	1.3	0.999049901
0.5	0.7	1.35	0.99599319
0.6	0.7	1.3	0.999128308
0.7	0.7	1.3	0.99918447
0.8	0.7	1.3	0.999190889
0.9	0.7	1.3	0.999196154
1	0.65	1.15	0.995564417
1	0.65	1.2	0.996667755
1	0.7	1.2	0.995578758
1	0.7	1.25	0.999435607
1	0.7	1.3	0.99910933
1.1	0.7	1.2	0.99603167
1.1	0.7	1.3	0.99894698
1.2	0.7	1.2	0.996660673
1.2	0.7	1.3	0.998709423
1.3	0.7	1.2	0.997197211
1.3	0.7	1.3	0.998316186
1.4	0.7	1.2	0.99761326
1.4	0.7	1.3	0.997863267
1.5	0.7	1.2	0.998006627
1.5	0.7	1.25	0.999843839
1.5	0.7	1.3	0.997229625
1.5	0.75	1.35	0.995205033
1.6	0.7	1.2	0.998270734
1.6	0.7	1.3	0.996305505
1.7	0.7	1.2	0.998361735
1.7	0.7	1.3	0.995122534
1.8	0.7	1.2	0.998283832
1.9	0.7	1.2	0.99800027
2	0.7	1.2	0.997376305
2	0.7	1.25	0.995878255
2	0.7	1.3	0.995928754
2	0.75	1.25	0.995016845
2	0.75	1.3	0.998440285
2	0.75	1.35	0.997826786
2.5	0.75	1.25	0.998980735
2.5	0.75	1.3	0.998781695"""
from io import StringIO
df_orig = pd.read_csv(StringIO(data_original), sep='\t')
df_orig.rename(columns={'pitch':'Period (μm)', 'a':'f', 'h':'h (μm)'}, inplace=True)

# ======================
# 3. 整合图（条形图+折线图）
# ======================
fig, ax = plt.subplots(figsize=(14, 7))
df_sorted = df.sort_values('err_Simu_1').reset_index(drop=True)
x_pos = np.arange(len(df_sorted))
width = 0.25

target_vals = [1.0] * len(df_sorted)
pred_vals = df_sorted['predicted_y'].values
sim_vals = df_sorted['Simulation'].values

# 条形图
ax.bar(x_pos - width, target_vals, width, label='Target Value (1.0)',
       color=SCI_COLORS['target'], edgecolor='black', alpha=0.6)
ax.bar(x_pos, pred_vals, width, label='ML Predicted Value',
       color=SCI_COLORS['prediction'], edgecolor='black', alpha=0.7)
ax.bar(x_pos + width, sim_vals, width, label='Simulation Value',
       color=SCI_COLORS['simulation'], edgecolor='black', alpha=0.7)

# 折线图叠加
ax.plot(x_pos, pred_vals, 'o-', color=SCI_COLORS['prediction'], linewidth=1.5, markersize=6, alpha=0.8)
ax.plot(x_pos, sim_vals, 's-', color=SCI_COLORS['simulation'], linewidth=1.5, markersize=6, alpha=0.8)

# 误差箭头
for i, (p, s) in enumerate(zip(pred_vals, sim_vals)):
    if abs(p-s) > 0.001:
        ax.annotate('', xy=(i+width/2, s), xytext=(i+width/2, p),
                    arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8, alpha=0.5))
        ax.text(i+width/2, (p+s)/2, f'{abs(p-s):.4f}', ha='center', va='center', fontsize=7)

ax.set_xticks(x_pos)
ax.set_xticklabels(df_sorted['Combination'] + f"\n(ID:{df_sorted['ID'].astype(int)})",
                   rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Value', fontsize=12, fontweight='bold')
ax.set_xlabel('Model-Optimization Combination', fontsize=12, fontweight='bold')
ax.set_title('Target vs Prediction vs Simulation (Sorted by Simulation Error)', fontsize=14, fontweight='bold')
ax.set_ylim(0.99, 1.005)
ax.legend(fontsize=10, loc='upper right', frameon=True)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.axhline(y=1.0, color=SCI_COLORS['target'], linestyle='-', linewidth=1.5, alpha=0.8)
ax.axhspan(0.998, 1.002, alpha=0.1, color=SCI_COLORS['target'])

plt.tight_layout()
plt.savefig('0_integrated_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ======================
# 4. 偏差箱线图
# ======================
fig, ax = plt.subplots(figsize=(8, 6))
bp = ax.boxplot([df['pred_bias'], df['sim_bias']],
                labels=['Prediction Bias\n(Pred - Target)', 'Simulation Bias\n(Sim - Target)'],
                patch_artist=True, showmeans=True, meanline=True,
                medianprops=dict(linewidth=2, color='black'),
                meanprops=dict(linewidth=2, color='red', linestyle='--'))
for patch, color in zip(bp['boxes'], [SCI_COLORS['prediction'], SCI_COLORS['simulation']]):
    patch.set_facecolor(color); patch.set_alpha(0.7); patch.set_edgecolor('black')
ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.8)
ax.set_ylabel('Bias Value', fontsize=12, fontweight='bold')
ax.set_title('Bias Distribution: Prediction vs Simulation', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

pred_mean, pred_std = df['pred_bias'].mean(), df['pred_bias'].std()
sim_mean, sim_std = df['sim_bias'].mean(), df['sim_bias'].std()
ax.text(0.65, 0.0015, f'Mean={pred_mean:.6f}\nStd={pred_std:.6f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
ax.text(1.65, 0.0015, f'Mean={sim_mean:.6f}\nStd={sim_std:.6f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
plt.tight_layout()
plt.savefig('1_bias_boxplot.png', dpi=300)
plt.show()

# ======================
# 5. 仿真误差柱状图
# ======================
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df.sort_values('err_Simu_1').reset_index(drop=True)
x_pos = np.arange(len(df_sorted))
colors_bar = [SCI_COLORS['RF'] if m=='RF' else SCI_COLORS['ET'] for m in df_sorted['model']]
bars = ax.bar(x_pos, df_sorted['err_Simu_1'], color=colors_bar, edgecolor='black', alpha=0.8)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{row['Combination']}\n(ID:{row['ID']})" for _, row in df_sorted.iterrows()],
                   rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Simulation Error', fontsize=12, fontweight='bold')
ax.set_xlabel('Model-Optimization Combination', fontsize=12, fontweight='bold')
ax.set_title('Simulation Error (Lower is Better)', fontsize=14, fontweight='bold')
for bar, val in zip(bars, df_sorted['err_Simu_1']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.00005, f'{val:.5f}', ha='center', fontsize=8, rotation=90)
ax.axhline(y=0.001, color='green', linestyle='--', linewidth=1.5, label='Excellent (0.001)')
ax.axhline(y=0.002, color='orange', linestyle='--', linewidth=1.5, label='Good (0.002)')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('2_simulation_error.png', dpi=300)
plt.show()

# ======================
# 6. 三种误差对比
# ======================
fig, ax = plt.subplots(figsize=(12, 6))
comb = df.groupby('Combination')[['error_pre_1','err_pre_simu','err_Simu_1']].mean().sort_values('err_Simu_1')
x = np.arange(len(comb)); width = 0.25
ax.bar(x-width, comb['error_pre_1'], width, label='Prediction Error (vs Target)', color=SCI_COLORS['error1'], edgecolor='black')
ax.bar(x, comb['err_pre_simu'], width, label='Pred-Sim Deviation', color=SCI_COLORS['error2'], edgecolor='black')
ax.bar(x+width, comb['err_Simu_1'], width, label='Simulation Error (vs Target)', color=SCI_COLORS['error3'], edgecolor='black')
ax.set_xticks(x); ax.set_xticklabels(comb.index, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Error Value', fontsize=12, fontweight='bold')
ax.set_title('Three Types of Errors Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('3_three_errors_comparison.png', dpi=300)
plt.show()

# ======================
# 7. 仿真值散点图
# ======================
fig, ax = plt.subplots(figsize=(10, 6))
markers = {'RF':'s', 'ET':'o'}
colors_model = {'RF':SCI_COLORS['RF'], 'ET':SCI_COLORS['ET']}
for model in df['model'].unique():
    sub = df[df['model']==model]
    ax.scatter(sub['Simulation'], [1]*len(sub), s=150, marker=markers[model],
               c=colors_model[model], edgecolors='black', linewidth=1.5, alpha=0.8, label=model, zorder=5)
    for _, row in sub.iterrows():
        ax.annotate(row['method'], (row['Simulation'], 1), xytext=(0,12), textcoords='offset points', ha='center', fontsize=8)
ax.axvline(x=1.0, color='black', linestyle='-', linewidth=2, alpha=0.8, label='Target')
ax.axvspan(0.999, 1.001, alpha=0.2, color='green', label='Excellent (±0.1%)')
ax.axvspan(0.998, 0.999, alpha=0.2, color='yellow', label='Good (0.1-0.2%)')
ax.set_xlabel('Simulation Value', fontsize=12, fontweight='bold')
ax.set_title('Simulation Distribution (Closer to 1 is Better)', fontsize=14, fontweight='bold')
ax.set_xlim(0.995, 1.001); ax.set_ylim(0.998,1.002)
ax.set_yticks([1]); ax.set_yticklabels(['Target = 1'])
ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('4_simulation_scatter.png', dpi=300)
plt.show()

# ======================
# 8. 按模型箱线图
# ======================
fig, ax = plt.subplots(figsize=(8, 6))
models = ['RF','ET']
data_by_model = [df[df['model']==m]['err_Simu_1'].values for m in models]
bp = ax.boxplot(data_by_model, labels=models, patch_artist=True, showmeans=True, meanline=True)
for patch, color in zip(bp['boxes'], [SCI_COLORS['RF'], SCI_COLORS['ET']]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_ylabel('Simulation Error', fontsize=12, fontweight='bold')
ax.set_title('Error Distribution by Model', fontsize=14, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
for i, m in enumerate(models,1):
    mean_val = df[df['model']==m]['err_Simu_1'].mean()
    median_val = df[df['model']==m]['err_Simu_1'].median()
    ax.text(i, 0.002, f'Mean={mean_val:.4f}\nMedian={median_val:.4f}', ha='center', fontsize=9, bbox=dict(boxstyle='round', facecolor='white'))
plt.tight_layout()
plt.savefig('5_model_boxplot.png', dpi=300)
plt.show()

# ======================
# 9. Top10排名
# ======================
fig, ax = plt.subplots(figsize=(10, 6))
top10 = df.nsmallest(10, 'err_Simu_1')[['Combination','err_Simu_1','model','method']]
colors_top = [SCI_COLORS['RF'] if m=='RF' else SCI_COLORS['ET'] for m in top10['model']]
bars = ax.barh(range(len(top10)), top10['err_Simu_1'], color=colors_top, edgecolor='black')
ax.set_yticks(range(len(top10)))
ax.set_yticklabels([f"{row['Combination']} ({row['method']})" for _, row in top10.iterrows()], fontsize=10)
ax.set_xlabel('Simulation Error', fontsize=12, fontweight='bold')
ax.set_title('Top 10 Best Solutions', fontsize=14, fontweight='bold')
for bar, val in zip(bars, top10['err_Simu_1']):
    ax.text(val+0.00005, bar.get_y()+bar.get_height()/2, f'{val:.5f}', va='center', fontsize=9)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('6_top10_ranking.png', dpi=300)
plt.show()

# ======================
# 10. 参数vs误差散点图（新名称）
# ======================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
params = ['Period (μm)', 'f', 'h (μm)']
for i, param in enumerate(params):
    for model in df['model'].unique():
        sub = df[df['model']==model]
        axes[i].scatter(sub[param], sub['err_Simu_1'], s=80,
                        marker='s' if model=='RF' else 'o',
                        c=SCI_COLORS['RF'] if model=='RF' else SCI_COLORS['ET'],
                        edgecolors='black', linewidth=1, alpha=0.7, label=model)
    axes[i].set_xlabel(param, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('Simulation Error', fontsize=12, fontweight='bold')
    axes[i].set_title(f'{param} vs Error', fontsize=12, fontweight='bold')
    axes[i].grid(True, alpha=0.3)
    axes[i].legend(fontsize=9)
plt.tight_layout()
plt.savefig('7_params_vs_error.png', dpi=300)
plt.show()

# ======================
# 11. 雷达图
# ======================
fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(projection='polar'))
metrics_data = {}
for model in models:
    sub = df[df['model']==model]
    metrics_data[model] = {
        'mean_err': sub['err_Simu_1'].mean(),
        'std_err': sub['err_Simu_1'].std(),
        'max_err': sub['err_Simu_1'].max(),
        'excel_rate': (sub['err_Simu_1']<=0.001).sum()/len(sub)
    }
all_means = [metrics_data[m]['mean_err'] for m in models]
all_stds  = [metrics_data[m]['std_err'] for m in models]
all_maxs  = [metrics_data[m]['max_err'] for m in models]
all_rates = [metrics_data[m]['excel_rate'] for m in models]
angles = np.linspace(0, 2*np.pi, 4, endpoint=False).tolist() + [0]
metrics_names = ['Mean Error', 'Stability\n(1/Std)', 'Max Error', 'Excellence Rate']
for model, color in zip(models, [SCI_COLORS['RF'], SCI_COLORS['ET']]):
    values = [
        metrics_data[model]['mean_err']/max(all_means) if max(all_means)>0 else 0,
        1 - metrics_data[model]['std_err']/max(all_stds) if max(all_stds)>0 else 1,
        metrics_data[model]['max_err']/max(all_maxs) if max(all_maxs)>0 else 0,
        metrics_data[model]['excel_rate']/max(all_rates) if max(all_rates)>0 else 0
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
    ax.fill(angles, values, alpha=0.15, color=color)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics_names, fontsize=11)
ax.set_ylim(0,1); ax.set_title('Model Performance Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.2,1.0)); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('8_radar_chart.png', dpi=300)
plt.show()

# ======================
# 12. 平行坐标图（左右对比：原始数据 vs 预测数据）
# ======================
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 准备原始数据（41个高T点）
orig_plot = df_orig[['Period (μm)','f','h (μm)','T']].copy()
# 准备预测数据（12个点，用Simulation作为T值）
pred_plot = df[['Period (μm)','f','h (μm)','Simulation']].copy()
pred_plot.rename(columns={'Simulation':'T'}, inplace=True)
pred_plot['Method'] = df['Combination']

# 创建左右子图
fig_par = make_subplots(rows=1, cols=2,
                        subplot_titles=('Original Data (41 points, Rainbow)', 'ML Predictions (12 points, Black)'),
                        shared_yaxes=False)

# 左图：原始数据
fig_par.add_trace(go.Parcoords(
    line=dict(color=orig_plot['T'], colorscale='Rainbow', showscale=True,
              colorbar=dict(title="Temperature (T)", x=0.45)),
    dimensions=[
        dict(label='Period (μm)', values=orig_plot['Period (μm)'],
             range=[orig_plot['Period (μm)'].min(), orig_plot['Period (μm)'].max()]),
        dict(label='f', values=orig_plot['f'],
             range=[orig_plot['f'].min(), orig_plot['f'].max()]),
        dict(label='h (μm)', values=orig_plot['h (μm)'],
             range=[orig_plot['h (μm)'].min(), orig_plot['h (μm)'].max()]),
        dict(label='T', values=orig_plot['T'],
             range=[orig_plot['T'].min(), orig_plot['T'].max()])
    ],
    name='Original Data',
    visible=True
), row=1, col=1)

# 右图：预测数据（黑色线条）
fig_par.add_trace(go.Parcoords(
    line=dict(color='black', showscale=False),
    dimensions=[
        dict(label='Period (μm)', values=pred_plot['Period (μm)'],
             range=[orig_plot['Period (μm)'].min(), orig_plot['Period (μm)'].max()]),
        dict(label='f', values=pred_plot['f'],
             range=[orig_plot['f'].min(), orig_plot['f'].max()]),
        dict(label='h (μm)', values=pred_plot['h (μm)'],
             range=[orig_plot['h (μm)'].min(), orig_plot['h (μm)'].max()]),
        dict(label='T', values=pred_plot['T'],
             range=[orig_plot['T'].min(), orig_plot['T'].max()])
    ],
    name='Predictions',
    visible=True
), row=1, col=2)

fig_par.update_layout(
    title='Parallel Coordinates Comparison: Original Data vs ML Predictions',
    height=600, width=1400,
    font=dict(family='Times New Roman', size=12),
    plot_bgcolor='white', paper_bgcolor='white'
)
fig_par.write_html("parallel_coordinates_comparison.html")
print("✓ 生成: parallel_coordinates_comparison.html")

# ======================
# 13. 统计报告
# ======================
print("="*80)
print("MODIFIED VISUALIZATION REPORT (New Data)")
print("="*80)
print(f"\n[1] Bias Statistics")
print(f"    Prediction Mean Bias: {df['pred_bias'].mean():.8f}  Std: {df['pred_bias'].std():.8f}")
print(f"    Simulation Mean Bias: {df['sim_bias'].mean():.8f}  Std: {df['sim_bias'].std():.8f}")
print(f"\n[2] Best Combination (lowest simulation error):")
best = df.loc[df['err_Simu_1'].idxmin()]
print(f"    {best['model']}+{best['method']} (ID:{best['ID']})  Error={best['err_Simu_1']:.6f}")
print(f"\n[3] Worst Combination:")
worst = df.loc[df['err_Simu_1'].idxmax()]
print(f"    {worst['model']}+{worst['method']} (ID:{worst['ID']})  Error={worst['err_Simu_1']:.6f}")
print(f"\n[4] Prediction-Simulation Correlation: {np.corrcoef(df['predicted_y'], df['Simulation'])[0,1]:.6f}")
print(f"\n[5] Model Ranking by Avg Error:")
for i, (m, e) in enumerate(df.groupby('model')['err_Simu_1'].mean().sort_values().items(),1):
    print(f"    {i}. {m}: {e:.6f}")
print("\nAll figures generated successfully!")