# ===================== ANN 独立系统：AutoML + 多层感知机 + 反演（PSO + GA） =====================

import numpy as np
import pandas as pd
import optuna
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
# from pyswarm import pso
import pygad

# =====================
# 1. 数据加载与预处理
# =====================
df = pd.read_csv(r'D:\Concurrent\HYM\square\total.csv')
df.columns = df.columns.str.strip().str.lower()
target_col = df.columns[-1]

df = df.groupby(['period', 'f', 'h'], as_index=False).mean()
print(f"数据去重并取平均后样本数: {len(df)}")

X = df[['period', 'f', 'h']].values
y = df[target_col].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# 2. Optuna 调参 (针对 ANN)
# =====================
def objective(trial):
    params = {
        'hidden_layer_sizes': trial.suggest_categorical('hidden_layer_sizes', [(64, 64), (128, 64, 32), (100,)]),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
    }
    # 必须标准化
    model = make_pipeline(
        StandardScaler(), 
        MLPRegressor(**params, max_iter=50, early_stopping=True, random_state=42) # 本机调试max_iter=50，正式跑改回500
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    return -score.mean()

print("\n开始 Optuna 寻优 ANN 超参数...")
study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=50)  # 神经网络较慢，建议 50 次
study.optimize(objective, n_trials=2)  # 本机调试 2 次，正式跑改回 50 次
best_params = study.best_params

# =====================
# 3. 训练最优模型与评估
# =====================
best_ann_model = make_pipeline(
    StandardScaler(),
    MLPRegressor(
        hidden_layer_sizes=best_params['hidden_layer_sizes'],
        activation=best_params['activation'],
        alpha=best_params['alpha'],
        learning_rate_init=best_params['learning_rate_init'],
        # max_iter=800, 本机调试，使用50，正式跑改回800
        max_iter=50,
        random_state=42
    )
)

best_ann_model.fit(X_train, y_train)
y_pred = best_ann_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n" + "="*50)
print("ANN 模型评估结果")
print(f"   R²  ：{r2:.4f}")
print(f"   MSE ：{mse:.6f}")
print(f"   RMSE：{rmse:.4f}")
print(f"   最优参数：{best_params}")
print("="*50)

# 保存模型
joblib.dump(best_ann_model, 'ANN_model.pkl')
print("已保存模型: ANN_model.pkl")

# =====================
# 4. 预测约束与反演模块
# =====================
def predict_clipped(model, x):
    y_pred = model.predict([x])[0]
    return float(np.clip(y_pred, 0.0, 1.0))

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


target_y = float(input("\n输入目标y(建议0~1): "))
lb, ub = [0, 0.1, 0], [6, 1, 3]
all_results = []

# --- PSO 反演 ---
print("\n开始 PSO 反演...")
for i in range(1): # 本机调试，跑 1 遍证明逻辑通顺，正式跑改回5
    def pso_objective(x):
        return abs(predict_clipped(best_ann_model, x) - target_y)
    
    best_x, _ = my_pso(pso_objective, lb, ub, swarmsize=30, maxiter=100)
    pred_y = predict_clipped(best_ann_model, best_x)
    error = abs(pred_y - target_y)
    all_results.append(['ANN', 'PSO', best_x[0], best_x[1], best_x[2], pred_y, error])

# --- GA 反演 ---
print("开始 GA 反演...")
def fitness_func(ga_instance, solution, solution_idx):
    return -abs(predict_clipped(best_ann_model, solution) - target_y)

# 本机调试，遗传算法进化代数从 100 代直接砍到 2 代
ga_instance = pygad.GA(
    num_generations=2, num_parents_mating=10, sol_per_pop=20, num_genes=3,
    init_range_low=[lb[0], lb[1], lb[2]], init_range_high=[ub[0], ub[1], ub[2]],
    fitness_func=fitness_func, mutation_percent_genes=20
)
ga_instance.run()
solution, _, _ = ga_instance.best_solution()
pred_y = predict_clipped(best_ann_model, solution)
error = abs(pred_y - target_y)
all_results.append(['ANN', 'GA', solution[0], solution[1], solution[2], pred_y, error])

# 保存结果
result_df = pd.DataFrame(all_results, columns=['model','method','period','f','h','predicted_y','error'])
result_df = result_df.sort_values('error')
result_df.to_csv('inverse_ANN.csv', index=False)
print("\n 反演完成，已保存 inverse_ANN.csv")