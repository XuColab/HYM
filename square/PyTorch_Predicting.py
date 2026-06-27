# ===================== PyTorch深度神经网络训练 + 原生反演 =====================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# from pyswarm import pso
import pygad

# =====================
# 1. 数据加载与 PyTorch 预处理
# =====================
df = pd.read_csv(r'D:\Concurrent\HYM\square\total.csv')
df.columns = df.columns.str.strip().str.lower()
target_col = df.columns[-1]

df = df.groupby(['period', 'f', 'h'], as_index=False).mean()

X = df[['period', 'f', 'h']].values
y = df[target_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# PyTorch 极其依赖特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 转换为 Tensor
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)

# 构建 DataLoader (小批量训练)
train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# =====================
# 2. 定义纯正的 PyTorch 网络结构
# =====================
class DeepRegressionNet(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super(DeepRegressionNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

# =====================
# 3. 模型初始化与标准训练循环
# =====================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"正在使用的计算设备: {device}")

model = DeepRegressionNet(input_dim=3, hidden_dim=64).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

epochs = 5 # 本机调试，跑5次，正式跑改回 500。原本要让数据集在网络中前向/反向传播 500 次，改成 5 次瞬间就能跑完，足够验证 train_loader、loss.backward() 等底层逻辑
print("\n开始 PyTorch 模型训练...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.6f}")

# =====================
# 4. 模型评估与保存
# =====================
model.eval()  # 切换到推理模式
with torch.no_grad():
    y_pred_t = model(X_test_t.to(device))
    y_pred = y_pred_t.cpu().numpy().flatten()

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "="*50)
print(" PyTorch 模型评估结果")
print(f"   R²  ：{r2:.4f}")
print(f"   MSE ：{mse:.6f}")
print(f"   RMSE：{rmse:.4f}")
print("="*50)

# PyTorch 标准保存方式（保存参数字典）
torch.save(model.state_dict(), 'PyTorch_model.pth')
# 必须额外保存 StandardScaler，否则推理时无法归一化新数据
import joblib
joblib.dump(scaler, 'PyTorch_scaler.pkl')
print("已保存模型参数 (PyTorch_model.pth) 和归一化器 (PyTorch_scaler.pkl)")

# =====================
# 5. 针对 PyTorch 的反演推理函数
# =====================
def predict_pytorch(x_input):
    # 1. 使用保存的 scaler 对单个样本进行归一化
    x_scaled = scaler.transform([x_input])
    # 2. 转换为 Tensor
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
    # 3. 推理并限制在 [0,1]
    model.eval()
    with torch.no_grad():
        val = model(x_tensor).cpu().numpy()[0][0]
    return float(np.clip(val, 0.0, 1.0))

# =====================
# 自定义 PSO（避免第三方库版本问题）
# =====================
def my_pso(objective, lb, ub, swarmsize=30, maxiter=100):
    dim = len(lb)
    pos = np.random.uniform(lb, ub, (swarmsize, dim))
    vel = np.random.uniform(-1, 1, (swarmsize, dim)) * (np.array(ub) - np.array(lb)) * 0.1
    pbest_pos = pos.copy()
    pbest_val = np.array([objective(p) for p in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    w, c1, c2 = 0.7, 1.5, 1.5
    for _ in range(maxiter):
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = pos + vel
        pos = np.clip(pos, lb, ub)
        for i in range(swarmsize):
            val = objective(pos[i])
            if val < pbest_val[i]:
                pbest_val[i] = val
                pbest_pos[i] = pos[i].copy()
                if val < gbest_val:
                    gbest_val = val
                    gbest_pos = pos[i].copy()
    return gbest_pos, gbest_val

# =====================
# 6. PSO + GA 反演
# =====================
target_y = float(input("\n输入目标y(建议0~1): "))
lb, ub = [0, 0.1, 0], [6, 1, 3]
all_results = []

print("\n开始 PyTorch PSO 反演...")
for i in range(1): # 本机调试，跑 1 遍证明逻辑通顺，正式跑改回 5
    def pso_objective(x):
        return abs(predict_pytorch(x) - target_y)
    
    best_x, _ = my_pso(pso_objective, lb, ub, swarmsize=30, maxiter=100)
    pred_y = predict_pytorch(best_x)
    error = abs(pred_y - target_y)
    all_results.append(['PyTorch', 'PSO', best_x[0], best_x[1], best_x[2], pred_y, error])

print("开始 PyTorch GA 反演...")
def fitness_func(ga_instance, solution, solution_idx):
    return -abs(predict_pytorch(solution) - target_y)

# 本机调试，遗传算法进化代数从 100 代直接砍到 2 代
ga_instance = pygad.GA(
    num_generations=2, num_parents_mating=10, sol_per_pop=20, num_genes=3,
    init_range_low=[lb[0], lb[1], lb[2]], init_range_high=[ub[0], ub[1], ub[2]],
    fitness_func=fitness_func, mutation_percent_genes=20
)
ga_instance.run()
solution, _, _ = ga_instance.best_solution()
pred_y = predict_pytorch(solution)
error = abs(pred_y - target_y)
all_results.append(['PyTorch', 'GA', solution[0], solution[1], solution[2], pred_y, error])

result_df = pd.DataFrame(all_results, columns=['model','method','period','f','h','predicted_y','error'])
result_df = result_df.sort_values('error')
result_df.to_csv('inverse_PyTorch.csv', index=False)
print("\n反演完成，已保存 inverse_PyTorch.csv")