import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ======================
# 1. 数据读取和清洗
# ======================

def clean_data(df):
    """
    数据清洗函数
    1. 剔除任何有缺陷（缺失值）的数据行
    2. 如果前三项数据完全相同，则对应的T取平均值并合并为一条
    """
    print("\n" + "="*60)
    print("数据清洗过程")
    print("="*60)
    
    original_count = len(df)
    print(f"原始数据行数: {original_count}")
    
    # 1. 剔除有缺失值的行
    df_clean = df.dropna()
    removed_na = original_count - len(df_clean)
    print(f"剔除缺失值: {removed_na} 行")
    
    # 2. 对前三列相同的行进行合并，T取平均值
    # 按前三列分组，计算T的平均值
    df_grouped = df_clean.groupby(['pitch', 'a', 'h'], as_index=False)['T'].mean()
    grouped_count = len(df_grouped)
    removed_duplicates = len(df_clean) - grouped_count
    print(f"合并重复项（前三列相同）: {removed_duplicates} 行")
    
    # 重置索引
    df_grouped = df_grouped.reset_index(drop=True)
    
    print(f"清洗后数据行数: {len(df_grouped)}")
    print(f"有效数据保留率: {len(df_grouped)/original_count*100:.2f}%")
    
    return df_grouped

# 读取CSV文件
CSV_FILE_PATH = "totle.csv"  # 请替换为您的文件路径

try:
    # 读取原始数据
    df_raw = pd.read_csv(CSV_FILE_PATH)
    
    # 检查列数
    if df_raw.shape[1] != 4:
        print(f"警告: 检测到{df_raw.shape[1]}列，期望4列")
        print("将使用前四列")
        df_raw = df_raw.iloc[:, :4]
        df_raw.columns = ['pitch', 'a', 'h', 'T']
    else:
        # 尝试识别列名
        expected_cols = ['pitch', 'a', 'h', 'T']
        if all(col in df_raw.columns for col in expected_cols):
            pass  # 列名正确
        else:
            # 使用默认列名
            df_raw.columns = expected_cols
    
    # 数据清洗
    df = clean_data(df_raw)
    
    print("\n清洗后数据预览:")
    print(df.head())
    print("\n清洗后数据统计:")
    print(df.describe())
    
except FileNotFoundError:
    print(f"错误: 找不到文件 {CSV_FILE_PATH}")
    print("请修改 CSV_FILE_PATH 变量为您的文件路径")
    print("\n使用示例数据进行演示...")
    np.random.seed(42)
    n_points = 500
    pitch = np.random.uniform(0.4, 3.0, n_points)
    a = np.random.uniform(0.1, 0.9, n_points)
    h = np.random.uniform(0.4, 6.0, n_points)
    # 生成更真实的T值
    T = 0.7 + 0.2 * np.sin(pitch) * np.cos(a*5) * np.sin(h*2) + np.random.normal(0, 0.05, n_points)
    T = np.clip(T, 0.08, 1.0)
    df = pd.DataFrame({'pitch': pitch, 'a': a, 'h': h, 'T': T})
    print(f"生成了 {len(df)} 个示例数据点")

# ======================
# 3. 提取数据
# ======================
pitch = df['pitch'].values
a = df['a'].values
h = df['h'].values
T = df['T'].values

# 数据范围
pitch_min, pitch_max = pitch.min(), pitch.max()
a_min, a_max = a.min(), a.max()
h_min, h_max = h.min(), h.max()

print(f"\n数据范围:")
print(f"  Pitch: [{pitch_min:.3f}, {pitch_max:.3f}]")
print(f"  a: [{a_min:.3f}, {a_max:.3f}]")
print(f"  h: [{h_min:.3f}, {h_max:.3f}]")
print(f"  T: [{T.min():.6f}, {T.max():.6f}]")

# ======================
# 4. 创建插值网格
# ======================
# 定义网格分辨率（可根据需要调整）
n_pitch = 50  # pitch方向网格数
n_a = 30      # a方向网格数
n_h = 40      # h方向网格数

pitch_grid = np.linspace(pitch_min, pitch_max, n_pitch)
a_grid = np.linspace(a_min, a_max, n_a)
h_grid = np.linspace(h_min, h_max, n_h)

print(f"\n网格分辨率: {n_pitch} × {n_a} × {n_h} = {n_pitch * n_a * n_h} 个点")

# 创建3D网格
X, Y, Z = np.meshgrid(pitch_grid, a_grid, h_grid, indexing='ij')

# 使用griddata进行插值
points = np.column_stack([pitch, a, h])
values = T

print("正在进行线性插值...")
T_interp = griddata(points, values, (X, Y, Z), method='linear', fill_value=np.nan)

# 处理NaN值（用最近邻插值填充）
if np.any(np.isnan(T_interp)):
    print("检测到NaN值，使用最近邻插值填充...")
    T_nearest = griddata(points, values, (X, Y, Z), method='nearest')
    nan_mask = np.isnan(T_interp)
    T_interp[nan_mask] = T_nearest[nan_mask]

# 可选：平滑处理
smooth_sigma = 0.5
T_interp_smooth = gaussian_filter(T_interp, sigma=smooth_sigma)
print(f"已完成插值和平滑 (sigma={smooth_sigma})")

# ======================
# 5. 创建动画切片可视化
# ======================

def create_animation_slices(data_3d, x_coords, y_coords, z_coords, 
                            slice_dimension='pitch',  # 'pitch', 'a', 或 'h'
                            n_slices=15,
                            title="4D Data Visualization - Animated Slices"):
    """
    创建沿指定维度的动画切片
    """
    
    # 选择切片维度
    if slice_dimension == 'pitch':
        values_1d = x_coords
        slice_label = 'Pitch'
        x_label = y_coords
        y_label = z_coords
        x_title = 'a'
        y_title = 'h'
    elif slice_dimension == 'a':
        values_1d = y_coords
        slice_label = 'a'
        x_label = x_coords
        y_label = z_coords
        x_title = 'pitch'
        y_title = 'h'
    elif slice_dimension == 'h':
        values_1d = z_coords
        slice_label = 'h'
        x_label = x_coords
        y_label = y_coords
        x_title = 'pitch'
        y_title = 'a'
    else:
        raise ValueError("slice_dimension must be 'pitch', 'a', or 'h'")
    
    # 均匀选择切片索引
    slice_indices = np.linspace(0, len(values_1d) - 1, n_slices, dtype=int)
    
    # 颜色范围
    t_min, t_max = T.min(), T.max()
    if t_max - t_min < 0.01:
        t_min = max(0.08, t_min - 0.005)
        t_max = min(1.0, t_max + 0.005)
    
    # 创建帧
    frames = []
    for idx in slice_indices:
        if slice_dimension == 'pitch':
            slice_data = data_3d[idx, :, :].T
        elif slice_dimension == 'a':
            slice_data = data_3d[:, idx, :].T
        else:  # h
            slice_data = data_3d[:, :, idx].T
        
        slice_value = values_1d[idx]
        
        frame = go.Frame(
            data=[go.Surface(
                z=slice_data,
                x=x_label,
                y=y_label,
                colorscale='RdYlGn_r',
                showscale=True,
                colorbar=dict(title="T", x=1.02),
                cmin=t_min,  # 使用 cmin 而不是 zmin
                cmax=t_max   # 使用 cmax 而不是 zmax
            )],
            name=f'{slice_label}={slice_value:.3f}',
            layout=dict(
                title=f'{title}<br>{slice_label} = {slice_value:.3f}'
            )
        )
        frames.append(frame)
    
    # 创建初始切片
    first_idx = slice_indices[0]
    if slice_dimension == 'pitch':
        first_slice = data_3d[first_idx, :, :].T
    elif slice_dimension == 'a':
        first_slice = data_3d[:, first_idx, :].T
    else:
        first_slice = data_3d[:, :, first_idx].T
    
    # 创建图形
    fig = go.Figure(
        data=[go.Surface(
            z=first_slice,
            x=x_label,
            y=y_label,
            colorscale='RdYlGn_r',
            showscale=True,
            colorbar=dict(title="T", x=1.02),
            cmin=t_min,  # 使用 cmin 而不是 zmin
            cmax=t_max   # 使用 cmax 而不是 zmax
        )],
        frames=frames
    )
    
    # 更新布局
    fig.update_layout(
        title=dict(text=f'{title}<br>Use slider to navigate through slices', x=0.5),
        scene=dict(
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title='Temperature (T)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            zaxis=dict(range=[t_min, t_max])
        ),
        height=700,
        width=1000,
        updatemenus=[
            dict(
                type='buttons',
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, {'frame': {'duration': 300, 'redraw': True},
                                     'fromcurrent': True, 'mode': 'immediate'}]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                       'mode': 'immediate'}])
                ],
                direction='left',
                pad=dict(r=10, t=85),
                showactive=False,
                x=0.1,
                y=0,
                xanchor='right',
                yanchor='top'
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method='animate',
                        args=[[f'{slice_label}={values_1d[idx]:.3f}'],
                              {'mode': 'immediate',
                               'frame': {'duration': 300, 'redraw': True},
                               'transition': {'duration': 300}}],
                        label=f'{values_1d[idx]:.2f}'
                    ) for idx in slice_indices
                ],
                transition=dict(duration=300),
                x=0.1,
                len=0.9,
                currentvalue=dict(
                    font=dict(size=12),
                    prefix=f'{slice_label}: ',
                    visible=True,
                    xanchor='center'
                )
            )
        ]
    )
    
    return fig

# 生成三个方向的动画
print("\n" + "="*60)
print("生成动画可视化...")
print("="*60)

# 沿pitch方向的动画
fig_pitch = create_animation_slices(
    T_interp_smooth, pitch_grid, a_grid, h_grid,
    slice_dimension='pitch',
    n_slices=min(20, len(pitch_grid)),
    title="Temperature Field - Slices along Pitch Direction"
)
fig_pitch.write_html("4d_animation_pitch.html")
print("✓ 生成: 4d_animation_pitch.html")

# 沿a方向的动画
fig_a = create_animation_slices(
    T_interp_smooth, pitch_grid, a_grid, h_grid,
    slice_dimension='a',
    n_slices=min(20, len(a_grid)),
    title="Temperature Field - Slices along a Direction"
)
fig_a.write_html("4d_animation_a.html")
print("✓ 生成: 4d_animation_a.html")

# 沿h方向的动画
fig_h = create_animation_slices(
    T_interp_smooth, pitch_grid, a_grid, h_grid,
    slice_dimension='h',
    n_slices=min(20, len(h_grid)),
    title="Temperature Field - Slices along h Direction"
)
fig_h.write_html("4d_animation_h.html")
print("✓ 生成: 4d_animation_h.html")

# ======================
# 6. 3D散点图（原始数据）
# ======================

fig_scatter = go.Figure()

# 原始数据点
fig_scatter.add_trace(go.Scatter3d(
    x=pitch, y=a, z=h,
    mode='markers',
    marker=dict(
        size=3,
        color=T,
        colorscale='RdYlGn_r',
        showscale=True,
        colorbar=dict(title="T", x=0.85),
        cmin=T.min(),
        cmax=T.max(),
        line=dict(width=0.3, color='black')
    ),
    text=[f'pitch={p:.3f}<br>a={a_val:.3f}<br>h={h_val:.3f}<br>T={t_val:.6f}' 
          for p, a_val, h_val, t_val in zip(pitch, a, h, T)],
    hoverinfo='text',
    name='Data Points'
))

fig_scatter.update_layout(
    title='3D Scatter Plot of Original Data (Cleaned)',
    scene=dict(
        xaxis_title='Pitch',
        yaxis_title='a',
        zaxis_title='h',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    height=700,
    width=1000
)

fig_scatter.write_html("4d_scatter_3d.html")
print("✓ 生成: 4d_scatter_3d.html")

# ======================
# 7. 体积渲染
# ======================

fig_volume = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=T_interp_smooth.flatten(),
    isomin=T_interp_smooth.min(),
    isomax=T_interp_smooth.max(),
    opacity=0.3,
    surface_count=15,
    colorscale='RdYlGn_r',
    colorbar=dict(title="T")
))

fig_volume.update_layout(
    title='3D Volume Rendering of Temperature Field',
    scene=dict(
        xaxis_title='Pitch',
        yaxis_title='a',
        zaxis_title='h',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
    ),
    height=700,
    width=1000
)

fig_volume.write_html("4d_volume_rendering.html")
print("✓ 生成: 4d_volume_rendering.html")

# ======================
# 8. 固定切片视图（三个典型位置）
# ======================

# 选择三个pitch位置（25%, 50%, 75%）
pitch_positions = [
    pitch_grid[len(pitch_grid)//4],
    pitch_grid[len(pitch_grid)//2],
    pitch_grid[3*len(pitch_grid)//4]
]

fig_slices = make_subplots(
    rows=1, cols=3,
    specs=[[{'type': 'surface'}, {'type': 'surface'}, {'type': 'surface'}]],
    subplot_titles=(f'Pitch = {pitch_positions[0]:.2f}', 
                    f'Pitch = {pitch_positions[1]:.2f}', 
                    f'Pitch = {pitch_positions[2]:.2f}'),
    horizontal_spacing=0.05
)

for i, pitch_val in enumerate(pitch_positions):
    idx = np.argmin(np.abs(pitch_grid - pitch_val))
    slice_data = T_interp_smooth[idx, :, :].T
    
    fig_slices.add_trace(
        go.Surface(
            z=slice_data, 
            x=a_grid, 
            y=h_grid, 
            colorscale='RdYlGn_r',
            showscale=(i==2),
            colorbar=dict(title="T", x=1.05),
            cmin=T.min(),
            cmax=T.max()
        ),
        row=1, col=i+1
    )

fig_slices.update_layout(
    title='Fixed Slices at Different Pitch Values',
    height=600,
    width=1400
)

fig_slices.write_html("4d_fixed_slices.html")
print("✓ 生成: 4d_fixed_slices.html")

# ======================
# 9. 等高线图（2D投影）
# ======================

# 创建三个方向的等高线图
fig_contour = make_subplots(
    rows=2, cols=2,
    subplot_titles=('a = median (Contour in pitch-h plane)', 
                    'pitch = median (Contour in a-h plane)',
                    'h = median (Contour in pitch-a plane)',
                    'Temperature Distribution'),
    vertical_spacing=0.15,
    horizontal_spacing=0.15
)

# 子图1: a = median
a_median = np.median(a_grid)
a_idx = np.argmin(np.abs(a_grid - a_median))
contour1 = T_interp_smooth[:, a_idx, :].T
fig_contour.add_trace(
    go.Contour(z=contour1, x=pitch_grid, y=h_grid, 
               colorscale='RdYlGn_r', contours=dict(showlabels=True),
               colorbar=dict(title="T")),
    row=1, col=1
)

# 子图2: pitch = median
pitch_median = np.median(pitch_grid)
pitch_idx = np.argmin(np.abs(pitch_grid - pitch_median))
contour2 = T_interp_smooth[pitch_idx, :, :].T
fig_contour.add_trace(
    go.Contour(z=contour2, x=a_grid, y=h_grid, 
               colorscale='RdYlGn_r', contours=dict(showlabels=True)),
    row=1, col=2
)

# 子图3: h = median
h_median = np.median(h_grid)
h_idx = np.argmin(np.abs(h_grid - h_median))
contour3 = T_interp_smooth[:, :, h_idx].T
fig_contour.add_trace(
    go.Contour(z=contour3, x=pitch_grid, y=a_grid, 
               colorscale='RdYlGn_r', contours=dict(showlabels=True)),
    row=2, col=1
)

# 子图4: 温度直方图
fig_contour.add_trace(
    go.Histogram(x=T, nbinsx=30, name='T Distribution', 
                 marker_color='blue', opacity=0.7),
    row=2, col=2
)

fig_contour.update_layout(
    title='2D Contour Plots and Distribution',
    height=800,
    width=1200,
    showlegend=False
)

fig_contour.write_html("4d_contour_plots.html")
print("✓ 生成: 4d_contour_plots.html")

# ======================
# 10. 平行坐标图
# ======================

fig_parallel = go.Figure(data=go.Parcoords(
    line=dict(color=T, colorscale='RdYlGn_r', showscale=True,
              colorbar=dict(title="T", x=0.85), cmin=T.min(), cmax=T.max()),
    dimensions=list([
        dict(label='Pitch', values=pitch, range=[pitch_min, pitch_max]),
        dict(label='a', values=a, range=[a_min, a_max]),
        dict(label='h', values=h, range=[h_min, h_max]),
        dict(label='T', values=T, range=[T.min(), T.max()])
    ])
))

fig_parallel.update_layout(
    title='Parallel Coordinates Plot - Multidimensional Visualization',
    height=600,
    width=1000
)

fig_parallel.write_html("4d_parallel_coordinates.html")
print("✓ 生成: 4d_parallel_coordinates.html")

# ======================
# 11. 导出清洗后的数据和插值结果
# ======================

print("\n导出数据...")

# 导出清洗后的原始数据
df.to_csv("cleaned_data.csv", index=False)
print("✓ 导出清洗后的数据: cleaned_data.csv")

# 导出插值网格数据（采样以避免文件过大）
grid_points = []
sample_step = 5  # 采样步长
for i in range(0, len(pitch_grid), sample_step):
    for j in range(0, len(a_grid), sample_step):
        for k in range(0, len(h_grid), sample_step):
            grid_points.append({
                'pitch': pitch_grid[i],
                'a': a_grid[j],
                'h': h_grid[k],
                'T_interpolated': T_interp_smooth[i, j, k]
            })

df_grid = pd.DataFrame(grid_points)
df_grid.to_csv("interpolated_grid_data.csv", index=False)
print(f"✓ 导出插值网格数据: interpolated_grid_data.csv ({len(df_grid)} 个点)")

# 导出数据统计
stats = {
    'Metric': ['Original Rows', 'Cleaned Rows', 'Pitch Range', 'a Range', 'h Range', 'T Range'],
    'Value': [len(df_raw), len(df), 
              f'[{pitch_min:.3f}, {pitch_max:.3f}]',
              f'[{a_min:.3f}, {a_max:.3f}]',
              f'[{h_min:.3f}, {h_max:.3f}]',
              f'[{T.min():.6f}, {T.max():.6f}]']
}
df_stats = pd.DataFrame(stats)
df_stats.to_csv("data_statistics.csv", index=False)
print("✓ 导出数据统计: data_statistics.csv")

# ======================
# 12. 最终报告
# ======================

print("\n" + "="*60)
print("✅ 4D VISUALIZATION COMPLETED!")
print("="*60)
print("\n📁 Generated Files:")
print("  📄 4d_animation_pitch.html - Animated slices along pitch direction")
print("  📄 4d_animation_a.html - Animated slices along a direction")
print("  📄 4d_animation_h.html - Animated slices along h direction")
print("  📄 4d_scatter_3d.html - Interactive 3D scatter plot")
print("  📄 4d_volume_rendering.html - 3D volume rendering")
print("  📄 4d_fixed_slices.html - Fixed slices at different positions")
print("  📄 4d_contour_plots.html - 2D contour plots")
print("  📄 4d_parallel_coordinates.html - Parallel coordinates plot")
print("  📄 cleaned_data.csv - Cleaned original data")
print("  📄 interpolated_grid_data.csv - Interpolated grid data")
print("  📄 data_statistics.csv - Statistical summary")
print("\n" + "="*60)
print("\n💡 Data Cleaning Summary:")
print(f"  • Removed rows with missing values")
print(f"  • Merged duplicate points (same pitch, a, h) by averaging T")
print(f"  • Final dataset: {len(df)} unique points")
print("\n💡 Usage Tips:")
print("  • Use the play/pause buttons and slider to control the animation")
print("  • Rotate/zoom the 3D plots for different perspectives")
print("  • Hover over points/surfaces to see detailed values")
print("  • The color scale shows temperature values (red=high, green=low)")
print("="*60)