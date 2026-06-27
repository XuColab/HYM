import numpy as np
import pandas as pd
from scipy.interpolate import Rbf, griddata, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

# 设置matplotlib字体为Times New Roman
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12
rcParams['mathtext.fontset'] = 'stix'

# ======================
# 1. 数据读取
# ======================
df = pd.read_csv("total.csv")
df.columns = ["pitch", "a", "h", "T"]
df = df.astype(float).dropna()

# 去重 + 平均
df_clean = df.groupby(["pitch", "a", "h"], as_index=False).mean()

x = df_clean["pitch"].values
y = df_clean["a"].values
z = df_clean["h"].values
v = df_clean["T"].values

# ======================
# 2. 网格（关键：ij索引）
# ======================
pitch = np.arange(0, 6.01, 0.05)
a = np.arange(0.1, 1.01, 0.1)
h = np.arange(0, 3.01, 0.5)

grid_x, grid_y, grid_z = np.meshgrid(
    pitch, a, h,
    indexing='ij'   # ⭐ 必须
)

# ======================
# 3. 插值方法
# ======================

# ---------- 3.1 RBF（降采样） ----------
sample_size = min(1000, len(x))
idx = np.random.choice(len(x), sample_size, replace=False)

rbf_model = Rbf(
    x[idx], y[idx], z[idx], v[idx],
    function='multiquadric',
    smooth=0.01
)

grid_rbf = rbf_model(grid_x, grid_y, grid_z)

# ---------- 3.2 Linear ----------
grid_linear = griddata(
    (x, y, z), v,
    (grid_x, grid_y, grid_z),
    method='linear'
)

# ---------- 3.3 Nearest ----------
grid_nearest = griddata(
    (x, y, z), v,
    (grid_x, grid_y, grid_z),
    method='nearest'
)

# ---------- 3.4 RGI（核心修复） ----------
grid_linear_full = griddata(
    (x, y, z), v,
    (grid_x, grid_y, grid_z),
    method='linear'
)

grid_linear_full = np.nan_to_num(grid_linear_full)

rgi = RegularGridInterpolator(
    (pitch, a, h),
    grid_linear_full,
    method='linear',
    bounds_error=False,
    fill_value=0
)

points = np.stack([
    grid_x.flatten(),
    grid_y.flatten(),
    grid_z.flatten()
], axis=-1)

grid_rgi = rgi(points).reshape(grid_x.shape)

# ======================
# 4. 平滑 + 约束
# ======================
def process(data):
    data = np.nan_to_num(data)
    data = np.clip(data, 0, 1)
    return np.round(data, 5)

def smooth(data, sigma=1):
    return gaussian_filter(data, sigma=sigma)

grid_rbf = process(smooth(process(grid_rbf)))
grid_linear = process(smooth(process(grid_linear)))
grid_nearest = process(smooth(process(grid_nearest)))
grid_rgi = process(smooth(process(grid_rgi)))

# ======================
# 5. 贝叶斯优化
# ======================

class BayesianOptimizer:
    def __init__(self, X_train, y_train, n_iter=50, n_initial=10):
        """
        X_train: 训练点坐标 (n_samples, 3)
        y_train: 目标值 (n_samples,)
        """
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.n_iter = n_iter
        self.n_initial = n_initial
        
        # 定义高斯过程核函数
        kernel = Matern(nu=2.5, length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            alpha=1e-6,
            normalize_y=True
        )
        
        # 边界（基于训练数据）
        self.bounds = np.array([
            [self.X_train[:, 0].min(), self.X_train[:, 0].max()],  # pitch
            [self.X_train[:, 1].min(), self.X_train[:, 1].max()],  # a
            [self.X_train[:, 2].min(), self.X_train[:, 2].max()]   # h
        ])
        
        # 存储优化历史
        self.X_history = self.X_train.copy()
        self.y_history = self.y_train.copy()
        self.best_x = None
        self.best_y = -np.inf
        
    def expected_improvement(self, X, xi=0.01):
        """计算期望改进（EI）"""
        mu, sigma = self.gp.predict(X, return_std=True)
        mu = mu.reshape(-1)
        sigma = sigma.reshape(-1)
        
        # 当前最佳值
        y_best = np.max(self.y_history)
        
        with np.errstate(divide='warn'):
            imp = mu - y_best - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0
        
        return ei
    
    def propose_location(self, n_samples=10000):
        """提出下一个采样点"""
        # 随机采样候选点
        X_candidates = np.random.uniform(
            self.bounds[:, 0], self.bounds[:, 1],
            size=(n_samples, 3)
        )
        
        # 计算EI
        ei = self.expected_improvement(X_candidates)
        
        # 选择EI最大的点
        best_idx = np.argmax(ei)
        return X_candidates[best_idx:best_idx+1]
    
    def optimize(self):
        """执行贝叶斯优化"""
        print("Starting Bayesian optimization...")
        
        # 初始拟合GP
        self.gp.fit(self.X_history, self.y_history)
        self.best_y = np.max(self.y_history)
        best_idx = np.argmax(self.y_history)
        self.best_x = self.X_history[best_idx]
        
        print(f"Initial best value: {self.best_y:.4f} at {self.best_x}")
        
        # 迭代优化
        for i in range(self.n_iter):
            # 提出新点
            X_next = self.propose_location()
            
            # 模拟实际评估（使用RGI插值）
            y_next = self.evaluate_point(X_next)
            
            # 添加到历史
            self.X_history = np.vstack([self.X_history, X_next])
            self.y_history = np.append(self.y_history, y_next)
            
            # 更新GP
            self.gp.fit(self.X_history, self.y_history)
            
            # 更新最佳值
            current_best = np.max(self.y_history)
            if current_best > self.best_y:
                self.best_y = current_best
                best_idx = np.argmax(self.y_history)
                self.best_x = self.X_history[best_idx]
                print(f"Iteration {i+1}/{self.n_iter}: New best value {self.best_y:.4f} at {self.best_x}")
            else:
                if (i+1) % 10 == 0:
                    print(f"Iteration {i+1}/{self.n_iter}: Current best {self.best_y:.4f}")
        
        print(f"\nOptimization completed! Final best value: {self.best_y:.4f}")
        print(f"Best parameters: pitch={self.best_x[0]:.3f}, a={self.best_x[1]:.3f}, h={self.best_x[2]:.3f}")
        
        return self.best_x, self.best_y
    
    def evaluate_point(self, X):
        """评估点（使用RGI插值）"""
        return rgi(X)[0]

# ======================
# 6. 准备贝叶斯优化数据（均匀采样点）
# ======================

def sample_uniform_points(n_points=200):
    """在长方体内均匀采样点"""
    pitch_range = [0, 6]
    a_range = [0.1, 1]
    h_range = [0, 3]
    
    pitch_samples = np.random.uniform(pitch_range[0], pitch_range[1], n_points)
    a_samples = np.random.uniform(a_range[0], a_range[1], n_points)
    h_samples = np.random.uniform(h_range[0], h_range[1], n_points)
    
    X_samples = np.column_stack([pitch_samples, a_samples, h_samples])
    
    # 使用RGI插值计算T值
    y_samples = rgi(X_samples)
    
    return X_samples, y_samples

# 生成均匀采样的初始点
print("Generating uniform sampling points...")
X_uniform, y_uniform = sample_uniform_points(n_points=500)

# ======================
# 7. 执行贝叶斯优化
# ======================
optimizer = BayesianOptimizer(X_uniform, y_uniform, n_iter=100, n_initial=50)
best_x, best_y = optimizer.optimize()

# ======================
# 8. 生成优化后的完整网格
# ======================
# 创建更密集的网格用于最终结果
pitch_final = np.arange(0, 6.01, 0.05)
a_final = np.arange(0.1, 1.01, 0.1)
h_final = np.arange(0, 3.01, 0.5)

grid_x_final, grid_y_final, grid_z_final = np.meshgrid(
    pitch_final, a_final, h_final,
    indexing='ij'
)

points_final = np.stack([
    grid_x_final.flatten(),
    grid_y_final.flatten(),
    grid_z_final.flatten()
], axis=-1)

# 优化后的插值
grid_optimized = rgi(points_final).reshape(grid_x_final.shape)
grid_optimized = process(smooth(process(grid_optimized)))

# ======================
# 9. 查找全局T值在0.995-1之间的点（修复数组连接错误）
# ======================

def find_global_points_in_range(data_3d, x_coords, y_coords, z_coords, lower=0.995, upper=1.0):
    """
    查找所有T值在指定范围内的点（全局）
    返回: 坐标列表和对应的T值，以及掩码
    """
    mask = (data_3d >= lower) & (data_3d <= upper)
    indices = np.where(mask)
    
    points_in_range = []
    values_in_range = []
    
    for i, j, k in zip(*indices):
        points_in_range.append([x_coords[i], y_coords[j], z_coords[k]])
        values_in_range.append(data_3d[i, j, k])
    
    if len(points_in_range) > 0:
        return np.array(points_in_range), np.array(values_in_range), mask
    else:
        return np.empty((0, 3)), np.array([]), mask

# 查找原始数据中T值在0.995-1之间的点
original_mask = (v >= 0.995) & (v <= 1.0)
if np.any(original_mask):
    original_points_in_range = np.column_stack([x[original_mask], y[original_mask], z[original_mask]])
    original_values_in_range = v[original_mask]
else:
    original_points_in_range = np.empty((0, 3))
    original_values_in_range = np.array([])

# 查找插值数据中T值在0.995-1之间的点
interp_points, interp_values, interp_mask = find_global_points_in_range(
    grid_rgi, pitch, a, h, lower=0.995, upper=1.0
)

print(f"\n{'='*60}")
print(f"GLOBAL POINTS IN TARGET RANGE [0.995, 1.000]:")
print(f"{'='*60}")
print(f"Original data points: {len(original_points_in_range)}")
print(f"Interpolated grid points: {len(interp_points)}")

# 安全地合并所有符合条件的点
all_points_list = []
all_values_list = []

if len(original_points_in_range) > 0:
    all_points_list.append(original_points_in_range)
    all_values_list.append(original_values_in_range)

if len(interp_points) > 0:
    all_points_list.append(interp_points)
    all_values_list.append(interp_values)

if len(all_points_list) > 0:
    all_points = np.vstack(all_points_list)
    all_values = np.concatenate(all_values_list)
    
    # 去重
    unique_points_dict = {}
    for pt, val in zip(all_points, all_values):
        key = tuple(np.round(pt, 3))
        if key not in unique_points_dict:
            unique_points_dict[key] = (pt, val)
    
    unique_points = np.array([v[0] for v in unique_points_dict.values()])
    unique_values = np.array([v[1] for v in unique_points_dict.values()])
    
    print(f"Unique points after deduplication: {len(unique_points)}")
else:
    unique_points = np.empty((0, 3))
    unique_values = np.array([])
    print("No points found in target range")

# ======================
# 10. 对全局高T区域进行切片分析（修复空数组处理）
# ======================

def analyze_high_t_region(grid_before, grid_after, pitch_coords, a_coords, h_coords,
                          original_points, interp_points, all_points, all_values,
                          output_dir="high_t_region_analysis"):
    """
    对T值在0.995-1之间的区域进行详细的切片分析
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if len(all_points) == 0:
        print("\n⚠ No points found in range [0.995, 1.000]")
        return False
    
    print(f"\n{'='*60}")
    print(f"ANALYZING HIGH T REGION (T ∈ [0.995, 1.000])")
    print(f"{'='*60}")
    print(f"Total points to analyze: {len(all_points)}")
    
    # 统计信息
    mean_t = np.mean(all_values)
    std_t = np.std(all_values)
    max_t = np.max(all_values)
    min_t = np.min(all_values)
    
    print(f"\nStatistical summary of high T region:")
    print(f"  Mean T: {mean_t:.6f}")
    print(f"  Std T: {std_t:.6f}")
    print(f"  Max T: {max_t:.6f}")
    print(f"  Min T: {min_t:.6f}")
    
    # 空间分布范围
    pitch_range = [np.min(all_points[:, 0]), np.max(all_points[:, 0])]
    a_range = [np.min(all_points[:, 1]), np.max(all_points[:, 1])]
    h_range = [np.min(all_points[:, 2]), np.max(all_points[:, 2])]
    
    print(f"\nSpatial distribution of high T region:")
    print(f"  Pitch range: [{pitch_range[0]:.4f}, {pitch_range[1]:.4f}]")
    print(f"  a range: [{a_range[0]:.4f}, {a_range[1]:.4f}]")
    print(f"  h range: [{h_range[0]:.4f}, {h_range[1]:.4f}]")
    
    # 创建colormap
    cmap = plt.cm.RdYlGn_r
    
    # 1. 3D散点图
    fig_scatter = go.Figure()
    
    if len(original_points) > 0:
        fig_scatter.add_trace(go.Scatter3d(
            x=original_points[:, 0], y=original_points[:, 1], z=original_points[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', symbol='circle'),
            name='Original data points'
        ))
    
    if len(interp_points) > 0:
        n_sample = min(500, len(interp_points))
        sample_idx = np.random.choice(len(interp_points), n_sample, replace=False)
        interp_sample = interp_points[sample_idx]
        
        fig_scatter.add_trace(go.Scatter3d(
            x=interp_sample[:, 0], y=interp_sample[:, 1], z=interp_sample[:, 2],
            mode='markers',
            marker=dict(size=3, color='blue', symbol='diamond', opacity=0.5),
            name='Interpolated grid points'
        ))
    
    fig_scatter.update_layout(
        title=f'Points with T in [0.995, 1.000] (Total: {len(all_points)})',
        scene=dict(
            xaxis_title='pitch',
            yaxis_title='a',
            zaxis_title='h'
        ),
        height=600,
        width=800
    )
    fig_scatter.write_html(f"{output_dir}/high_t_region_3d_points.html")
    print(f"\n✓ Saved: {output_dir}/high_t_region_3d_points.html")
    
    # 2. 创建多个切片
    slice_positions = {
        'pitch': [pitch_range[0], (pitch_range[0] + pitch_range[1])/2, pitch_range[1]],
        'a': [a_range[0], (a_range[0] + a_range[1])/2, a_range[1]],
        'h': [h_range[0], (h_range[0] + h_range[1])/2, h_range[1]]
    }
    
    for dim, positions in slice_positions.items():
        for idx, pos in enumerate(positions):
            if dim == 'pitch':
                # 找到最近的pitch索引
                pitch_idx = np.argmin(np.abs(pitch_coords - pos))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                # 优化前的切片
                im1 = ax1.contourf(a_coords, h_coords, grid_before[pitch_idx, :, :].T,
                                  levels=50, cmap=cmap, vmin=0.99, vmax=1.0)
                ax1.set_xlabel('a')
                ax1.set_ylabel('h')
                ax1.set_title(f'Before: pitch={pos:.4f}')
                ax1.grid(True, alpha=0.3)
                
                # 标记该切片上的高T点
                for pt in all_points:
                    if abs(pt[0] - pos) < (pitch_coords[1] - pitch_coords[0]):
                        ax1.scatter(pt[1], pt[2], color='red', s=30, marker='*', alpha=0.7)
                
                # 优化后的切片
                im2 = ax2.contourf(a_coords, h_coords, grid_after[pitch_idx, :, :].T,
                                  levels=50, cmap=cmap, vmin=0.99, vmax=1.0)
                ax2.set_xlabel('a')
                ax2.set_ylabel('h')
                ax2.set_title(f'After: pitch={pos:.4f}')
                ax2.grid(True, alpha=0.3)
                
                for pt in all_points:
                    if abs(pt[0] - pos) < (pitch_coords[1] - pitch_coords[0]):
                        ax2.scatter(pt[1], pt[2], color='red', s=30, marker='*', alpha=0.7)
                
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label('Temperature (T)')
                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label('Temperature (T)')
                
                plt.suptitle(f'Slice at Pitch = {pos:.4f} (T ∈ [0.995, 1.000])')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/slice_pitch_{idx}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: {output_dir}/slice_pitch_{idx}.png")
            
            elif dim == 'a':
                a_idx = np.argmin(np.abs(a_coords - pos))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                im1 = ax1.contourf(pitch_coords, h_coords, grid_before[:, a_idx, :].T,
                                  levels=50, cmap=cmap, vmin=0.99, vmax=1.0)
                ax1.set_xlabel('pitch')
                ax1.set_ylabel('h')
                ax1.set_title(f'Before: a={pos:.4f}')
                ax1.grid(True, alpha=0.3)
                
                for pt in all_points:
                    if abs(pt[1] - pos) < (a_coords[1] - a_coords[0]):
                        ax1.scatter(pt[0], pt[2], color='red', s=30, marker='*', alpha=0.7)
                
                im2 = ax2.contourf(pitch_coords, h_coords, grid_after[:, a_idx, :].T,
                                  levels=50, cmap=cmap, vmin=0.99, vmax=1.0)
                ax2.set_xlabel('pitch')
                ax2.set_ylabel('h')
                ax2.set_title(f'After: a={pos:.4f}')
                ax2.grid(True, alpha=0.3)
                
                for pt in all_points:
                    if abs(pt[1] - pos) < (a_coords[1] - a_coords[0]):
                        ax2.scatter(pt[0], pt[2], color='red', s=30, marker='*', alpha=0.7)
                
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label('Temperature (T)')
                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label('Temperature (T)')
                
                plt.suptitle(f'Slice at a = {pos:.4f} (T ∈ [0.995, 1.000])')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/slice_a_{idx}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: {output_dir}/slice_a_{idx}.png")
            
            elif dim == 'h':
                h_idx = np.argmin(np.abs(h_coords - pos))
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                im1 = ax1.contourf(pitch_coords, a_coords, grid_before[:, :, h_idx].T,
                                  levels=50, cmap=cmap, vmin=0.99, vmax=1.0)
                ax1.set_xlabel('pitch')
                ax1.set_ylabel('a')
                ax1.set_title(f'Before: h={pos:.4f}')
                ax1.grid(True, alpha=0.3)
                
                for pt in all_points:
                    if abs(pt[2] - pos) < (h_coords[1] - h_coords[0]):
                        ax1.scatter(pt[0], pt[1], color='red', s=30, marker='*', alpha=0.7)
                
                im2 = ax2.contourf(pitch_coords, a_coords, grid_after[:, :, h_idx].T,
                                  levels=50, cmap=cmap, vmin=0.99, vmax=1.0)
                ax2.set_xlabel('pitch')
                ax2.set_ylabel('a')
                ax2.set_title(f'After: h={pos:.4f}')
                ax2.grid(True, alpha=0.3)
                
                for pt in all_points:
                    if abs(pt[2] - pos) < (h_coords[1] - h_coords[0]):
                        ax2.scatter(pt[0], pt[1], color='red', s=30, marker='*', alpha=0.7)
                
                cbar1 = plt.colorbar(im1, ax=ax1)
                cbar1.set_label('Temperature (T)')
                cbar2 = plt.colorbar(im2, ax=ax2)
                cbar2.set_label('Temperature (T)')
                
                plt.suptitle(f'Slice at h = {pos:.4f} (T ∈ [0.995, 1.000])')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/slice_h_{idx}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Saved: {output_dir}/slice_h_{idx}.png")
    
    # 3. 温度分布直方图
    fig_hist, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.hist(all_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(x=0.9975, color='red', linestyle='--', label='Target range center')
    ax1.set_xlabel('Temperature (T)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of T in High Region')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    global_values = grid_rgi.flatten()
    ax2.hist(global_values, bins=50, alpha=0.5, color='gray', label='Global', edgecolor='black')
    ax2.hist(all_values, bins=30, alpha=0.7, color='red', label='High T Region', edgecolor='black')
    ax2.set_xlabel('Temperature (T)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('High T Region vs Global Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Temperature Distribution Analysis')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/temperature_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_dir}/temperature_distribution.png")
    
    # 4. 导出CSV
    # 创建source标签
    sources = []
    n_original = len(original_points)
    n_interp = len(interp_points)
    sources.extend(['original'] * n_original)
    sources.extend(['interpolated'] * n_interp)
    
    df_high_t = pd.DataFrame({
        'pitch': all_points[:, 0],
        'a': all_points[:, 1],
        'h': all_points[:, 2],
        'T': all_values,
        'source': sources
    })
    df_high_t.to_csv(f'{output_dir}/high_t_points.csv', index=False)
    print(f"✓ Saved: {output_dir}/high_t_points.csv")
    
    return True

# 执行高T区域分析
if len(unique_points) > 0:
    analysis_performed = analyze_high_t_region(
        grid_rgi, grid_optimized, pitch_final, a_final, h_final,
        original_points_in_range, interp_points, unique_points, unique_values,
        output_dir="high_t_region_analysis"
    )
else:
    analysis_performed = False
    print("\n⚠ No points found in range [0.995, 1.000]. Skipping high T region analysis.")

# ======================
# 11. 保存切片图片（原有功能：T值在0.995和1之间时才生成）
# ======================

def save_slice_images(grid_before, grid_after, pitch_coords, a_coords, h_coords, 
                      best_point, best_value, output_dir="slices"):
    """
    保存三个方向的二维切片图片
    只有当最佳T值在0.995和1之间时才生成
    """
    import os
    
    # 检查T值是否在0.995和1之间
    if 0.995 <= best_value <= 1:
        print(f"\n✓ Best T value ({best_value:.6f}) is between 0.995 and 1. Generating slice images...")
        os.makedirs(output_dir, exist_ok=True)
        
        # 找到最佳点附近的索引
        pitch_idx = np.argmin(np.abs(pitch_coords - best_point[0]))
        a_idx = np.argmin(np.abs(a_coords - best_point[1]))
        h_idx = np.argmin(np.abs(h_coords - best_point[2]))
        
        # 创建colormap
        cmap = plt.cm.RdYlGn_r
        
        # 1. Pitch固定切片 (a-h平面)
        fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.contourf(a_coords, h_coords, grid_before[pitch_idx, :, :].T, 
                           levels=50, cmap=cmap, vmin=0, vmax=1)
        ax1.set_xlabel('a', fontsize=12)
        ax1.set_ylabel('h', fontsize=12)
        ax1.set_title(f'Before Optimization\npitch = {best_point[0]:.4f}', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.scatter(best_point[1], best_point[2], color='blue', s=100, 
                   marker='*', edgecolors='white', linewidth=2, 
                   label=f'Optimal point (T={best_value:.6f})')
        ax1.legend(loc='upper right', fontsize=10)
        
        im2 = ax2.contourf(a_coords, h_coords, grid_after[pitch_idx, :, :].T, 
                           levels=50, cmap=cmap, vmin=0, vmax=1)
        ax2.set_xlabel('a', fontsize=12)
        ax2.set_ylabel('h', fontsize=12)
        ax2.set_title(f'After Optimization\npitch = {best_point[0]:.4f}', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.scatter(best_point[1], best_point[2], color='blue', s=100, 
                   marker='*', edgecolors='white', linewidth=2,
                   label=f'Optimal point (T={best_value:.6f})')
        ax2.legend(loc='upper right', fontsize=10)
        
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Temperature (T)', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Temperature (T)', fontsize=11)
        
        plt.suptitle(f'Slice at Fixed Pitch (p = {best_point[0]:.4f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/slice_fixed_pitch.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir}/slice_fixed_pitch.png")
        
        # 2. a固定切片 (pitch-h平面)
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.contourf(pitch_coords, h_coords, grid_before[:, a_idx, :].T, 
                           levels=50, cmap=cmap, vmin=0, vmax=1)
        ax1.set_xlabel('pitch', fontsize=12)
        ax1.set_ylabel('h', fontsize=12)
        ax1.set_title(f'Before Optimization\na = {best_point[1]:.4f}', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.scatter(best_point[0], best_point[2], color='blue', s=100, 
                   marker='*', edgecolors='white', linewidth=2,
                   label=f'Optimal point (T={best_value:.6f})')
        ax1.legend(loc='upper right', fontsize=10)
        
        im2 = ax2.contourf(pitch_coords, h_coords, grid_after[:, a_idx, :].T, 
                           levels=50, cmap=cmap, vmin=0, vmax=1)
        ax2.set_xlabel('pitch', fontsize=12)
        ax2.set_ylabel('h', fontsize=12)
        ax2.set_title(f'After Optimization\na = {best_point[1]:.4f}', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.scatter(best_point[0], best_point[2], color='blue', s=100, 
                   marker='*', edgecolors='white', linewidth=2,
                   label=f'Optimal point (T={best_value:.6f})')
        ax2.legend(loc='upper right', fontsize=10)
        
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Temperature (T)', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Temperature (T)', fontsize=11)
        
        plt.suptitle(f'Slice at Fixed a (a = {best_point[1]:.4f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/slice_fixed_a.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir}/slice_fixed_a.png")
        
        # 3. h固定切片 (pitch-a平面)
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        im1 = ax1.contourf(pitch_coords, a_coords, grid_before[:, :, h_idx].T, 
                           levels=50, cmap=cmap, vmin=0, vmax=1)
        ax1.set_xlabel('pitch', fontsize=12)
        ax1.set_ylabel('a', fontsize=12)
        ax1.set_title(f'Before Optimization\nh = {best_point[2]:.4f}', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.scatter(best_point[0], best_point[1], color='blue', s=100, 
                   marker='*', edgecolors='white', linewidth=2,
                   label=f'Optimal point (T={best_value:.6f})')
        ax1.legend(loc='upper right', fontsize=10)
        
        im2 = ax2.contourf(pitch_coords, a_coords, grid_after[:, :, h_idx].T, 
                           levels=50, cmap=cmap, vmin=0, vmax=1)
        ax2.set_xlabel('pitch', fontsize=12)
        ax2.set_ylabel('a', fontsize=12)
        ax2.set_title(f'After Optimization\nh = {best_point[2]:.4f}', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.scatter(best_point[0], best_point[1], color='blue', s=100, 
                   marker='*', edgecolors='white', linewidth=2,
                   label=f'Optimal point (T={best_value:.6f})')
        ax2.legend(loc='upper right', fontsize=10)
        
        cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('Temperature (T)', fontsize=11)
        cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label('Temperature (T)', fontsize=11)
        
        plt.suptitle(f'Slice at Fixed h (h = {best_point[2]:.4f})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/slice_fixed_h.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir}/slice_fixed_h.png")
        
        # 汇总图
        fig4, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        im1 = axes[0].contourf(a_coords, h_coords, grid_before[pitch_idx, :, :].T, 
                               levels=50, cmap=cmap, vmin=0, vmax=1)
        axes[0].set_xlabel('a')
        axes[0].set_ylabel('h')
        axes[0].set_title(f'Before: pitch={best_point[0]:.4f}')
        axes[0].grid(True, alpha=0.3)
        axes[0].scatter(best_point[1], best_point[2], color='blue', s=80, marker='*')
        
        im2 = axes[1].contourf(pitch_coords, h_coords, grid_before[:, a_idx, :].T, 
                               levels=50, cmap=cmap, vmin=0, vmax=1)
        axes[1].set_xlabel('pitch')
        axes[1].set_ylabel('h')
        axes[1].set_title(f'Before: a={best_point[1]:.4f}')
        axes[1].grid(True, alpha=0.3)
        axes[1].scatter(best_point[0], best_point[2], color='blue', s=80, marker='*')
        
        im3 = axes[2].contourf(pitch_coords, a_coords, grid_before[:, :, h_idx].T, 
                               levels=50, cmap=cmap, vmin=0, vmax=1)
        axes[2].set_xlabel('pitch')
        axes[2].set_ylabel('a')
        axes[2].set_title(f'Before: h={best_point[2]:.4f}')
        axes[2].grid(True, alpha=0.3)
        axes[2].scatter(best_point[0], best_point[1], color='blue', s=80, marker='*')
        
        im4 = axes[3].contourf(a_coords, h_coords, grid_after[pitch_idx, :, :].T, 
                               levels=50, cmap=cmap, vmin=0, vmax=1)
        axes[3].set_xlabel('a')
        axes[3].set_ylabel('h')
        axes[3].set_title(f'After: pitch={best_point[0]:.4f}')
        axes[3].grid(True, alpha=0.3)
        axes[3].scatter(best_point[1], best_point[2], color='blue', s=80, marker='*')
        
        im5 = axes[4].contourf(pitch_coords, h_coords, grid_after[:, a_idx, :].T, 
                               levels=50, cmap=cmap, vmin=0, vmax=1)
        axes[4].set_xlabel('pitch')
        axes[4].set_ylabel('h')
        axes[4].set_title(f'After: a={best_point[1]:.4f}')
        axes[4].grid(True, alpha=0.3)
        axes[4].scatter(best_point[0], best_point[2], color='blue', s=80, marker='*')
        
        im6 = axes[5].contourf(pitch_coords, a_coords, grid_after[:, :, h_idx].T, 
                               levels=50, cmap=cmap, vmin=0, vmax=1)
        axes[5].set_xlabel('pitch')
        axes[5].set_ylabel('a')
        axes[5].set_title(f'After: h={best_point[2]:.4f}')
        axes[5].grid(True, alpha=0.3)
        axes[5].scatter(best_point[0], best_point[1], color='blue', s=80, marker='*')
        
        cbar_ax = fig4.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig4.colorbar(im6, cax=cbar_ax)
        cbar.set_label('Temperature (T)', fontsize=12)
        
        plt.suptitle(f'2D Slices at Optimal Point (T = {best_value:.6f})', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        plt.savefig(f'{output_dir}/all_slices_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: {output_dir}/all_slices_summary.png")
        
        return True
    else:
        print(f"\n⚠ Best T value ({best_value:.6f}) is not in [0.995, 1.000]. No slice images generated.")
        return False

# 调用原有切片生成功能
images_generated = save_slice_images(grid_rgi, grid_optimized, pitch_final, a_final, h_final, 
                                      best_x, best_y, output_dir="optimization_slices")

# ======================
# 12. 可视化（均匀采样点）
# ======================
n_vis_points = 1000
vis_indices = np.random.choice(len(X_uniform), min(n_vis_points, len(X_uniform)), replace=False)
X_vis = X_uniform[vis_indices]
y_vis = y_uniform[vis_indices]

# ======================
# 13. 创建对比可视化
# ======================

terrain_colorscale = [
    [0.0, "darkblue"],
    [0.2, "blue"],
    [0.4, "green"],
    [0.6, "yellow"],
    [0.8, "orange"],
    [1.0, "red"]
]

fig_compare = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'scene'}, {'type': 'scene'}]],
    subplot_titles=('Before Optimization (RGI)', 'After Bayesian Optimization'),
    horizontal_spacing=0.1
)

# 优化前
fig_compare.add_trace(
    go.Volume(
        x=grid_x_final.flatten(),
        y=grid_y_final.flatten(),
        z=grid_z_final.flatten(),
        value=grid_rgi.flatten(),
        opacity=0.5,
        surface_count=10,
        colorscale=terrain_colorscale,
        colorbar=dict(title="T", x=0.45),
        name="Before"
    ),
    row=1, col=1
)

# 优化后
fig_compare.add_trace(
    go.Volume(
        x=grid_x_final.flatten(),
        y=grid_y_final.flatten(),
        z=grid_z_final.flatten(),
        value=grid_optimized.flatten(),
        opacity=0.5,
        surface_count=10,
        colorscale=terrain_colorscale,
        colorbar=dict(title="T", x=0.95),
        name="After"
    ),
    row=1, col=2
)

# 添加采样点（优化前）
fig_compare.add_trace(
    go.Scatter3d(
        x=X_vis[:, 0], y=X_vis[:, 1], z=X_vis[:, 2],
        mode='markers',
        marker=dict(size=3, color=y_vis, colorscale=terrain_colorscale, showscale=False),
        name="Sampling Points",
        opacity=0.6
    ),
    row=1, col=1
)

# 添加高T区域的点（新增加）
if len(unique_points) > 0:
    fig_compare.add_trace(
        go.Scatter3d(
            x=unique_points[:, 0], y=unique_points[:, 1], z=unique_points[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', symbol='diamond', opacity=0.8),
            name=f'High T Points (0.995-1.000)'
        ),
        row=1, col=2
    )

# 添加优化找到的最佳点
fig_compare.add_trace(
    go.Scatter3d(
        x=[best_x[0]], y=[best_x[1]], z=[best_x[2]],
        mode='markers',
        marker=dict(size=8, color='gold', symbol='star', line=dict(width=2, color='black')),
        name=f"Optimal Point (T={best_y:.6f})"
    ),
    row=1, col=2
)

# 更新布局
fig_compare.update_layout(
    title="Comparison Before and After Bayesian Optimization",
    height=600,
    width=1200
)

# 更新各子图的scene
fig_compare.update_scenes(
    dict(
        xaxis_title="pitch",
        yaxis_title="a",
        zaxis_title="h",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    row=1, col=1
)

fig_compare.update_scenes(
    dict(
        xaxis_title="pitch",
        yaxis_title="a",
        zaxis_title="h",
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    row=1, col=2
)

fig_compare.write_html("bayesian_optimization_comparison.html")

# ======================
# 14. 优化过程可视化
# ======================
fig_optimization = go.Figure()

iterations = np.arange(len(optimizer.y_history))
best_so_far = np.maximum.accumulate(optimizer.y_history)

fig_optimization.add_trace(go.Scatter(
    x=iterations,
    y=optimizer.y_history,
    mode='markers',
    name='All sampled points',
    marker=dict(size=4, opacity=0.6)
))

fig_optimization.add_trace(go.Scatter(
    x=iterations,
    y=best_so_far,
    mode='lines',
    name='Best value so far',
    line=dict(color='red', width=2)
))

fig_optimization.add_hline(y=0.995, line_dash="dash", line_color="green", 
                           annotation_text="Target lower bound (0.995)")
fig_optimization.add_hline(y=1.0, line_dash="dash", line_color="blue",
                           annotation_text="Target upper bound (1.0)")

fig_optimization.update_layout(
    title="Bayesian Optimization Convergence",
    xaxis_title="Iteration",
    yaxis_title="Temperature (T)",
    hovermode='closest'
)

fig_optimization.write_html("optimization_convergence.html")

# ======================
# 15. 导出结果
# ======================

# 导出优化后的完整网格
df_optimized = pd.DataFrame({
    "pitch": grid_x_final.flatten(),
    "a": grid_y_final.flatten(),
    "h": grid_z_final.flatten(),
    "T_optimized": grid_optimized.flatten(),
    "T_original": grid_rgi.flatten()
})
df_optimized.to_csv("optimized_interpolation.csv", index=False)

# 导出优化历史
df_history = pd.DataFrame({
    "iteration": np.arange(len(optimizer.y_history)),
    "pitch": optimizer.X_history[:, 0],
    "a": optimizer.X_history[:, 1],
    "h": optimizer.X_history[:, 2],
    "T": optimizer.y_history,
    "best_so_far": np.maximum.accumulate(optimizer.y_history)
})
df_history.to_csv("optimization_history.csv", index=False)

# 导出最佳点信息
df_best = pd.DataFrame({
    "pitch": [best_x[0]],
    "a": [best_x[1]],
    "h": [best_x[2]],
    "T": [best_y]
})
df_best.to_csv("best_point.csv", index=False)

# ======================
# 16. 最终输出
# ======================
print("\n" + "="*60)
print("✅ COMPLETED! Generated files:")
print("="*60)
print("1. bayesian_optimization_comparison.html - 3D comparison before/after")
print("2. optimization_convergence.html - Optimization convergence curve")
print("3. optimized_interpolation.csv - Optimized interpolation data")
print("4. optimization_history.csv - Full optimization history")
print("5. best_point.csv - Best point information")

if images_generated:
    print("\n📁 OPTIMIZATION SLICES (Best point in [0.995, 1.000]):")
    print("   - optimization_slices/slice_fixed_pitch.png")
    print("   - optimization_slices/slice_fixed_a.png")
    print("   - optimization_slices/slice_fixed_h.png")
    print("   - optimization_slices/all_slices_summary.png")

if analysis_performed:
    print("\n📁 HIGH T REGION ANALYSIS (Global points with T ∈ [0.995, 1.000]):")
    print("   - high_t_region_analysis/high_t_region_3d_points.html")
    print("   - high_t_region_analysis/slice_pitch_0.png, slice_pitch_1.png, slice_pitch_2.png")
    print("   - high_t_region_analysis/slice_a_0.png, slice_a_1.png, slice_a_2.png")
    print("   - high_t_region_analysis/slice_h_0.png, slice_h_1.png, slice_h_2.png")
    print("   - high_t_region_analysis/temperature_distribution.png")
    print("   - high_t_region_analysis/high_t_points.csv")

print("="*60)
print(f"\n🎯 OPTIMAL PARAMETERS:")
print(f"   pitch = {best_x[0]:.6f}")
print(f"   a = {best_x[1]:.6f}")
print(f"   h = {best_x[2]:.6f}")
print(f"   T = {best_y:.6f}")
print(f"\n📊 Distance from target (T=1): {abs(1-best_y):.6f}")
print(f"📊 Best T in range [0.995, 1.000]: {0.995 <= best_y <= 1}")
if len(unique_points) > 0:
    print(f"📊 Global points with T in [0.995, 1.000]: {len(unique_points)}")
else:
    print(f"📊 Global points with T in [0.995, 1.000]: 0")