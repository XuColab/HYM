# ===================== 完整系统：AutoML + 多模型反演（PSO + GA）+ 结果约束[0,1] + Web =====================

import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from pyswarm import pso
import joblib

# GA 使用 pygad（更易用）
import pygad

# =====================
# 1. 数据加载 + 预处理（重复自变量取平均）
# =====================
df = pd.read_csv('total.csv')
df.columns = df.columns.str.strip().str.lower()
# 假设最后一列为目标（如有固定列名请改为：target_col='y' 等）
target_col = df.columns[-1]

# 分组取平均
df = df.groupby(['period', 'f', 'h'], as_index=False).mean()

print("数据去重并取平均后样本数:", len(df))

X = df[['period', 'f', 'h']].values
y = df[target_col].values

# =====================
# 2. 数据划分
# =====================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =====================
# 3. Optuna 调参（统一）
# =====================

def tune_model(model_name):
    def objective(trial):
        if model_name == 'RF':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 70, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 30)
            }
            model = RandomForestRegressor(**params)

        elif model_name == 'ET':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 70, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 30)
            }
            model = ExtraTreesRegressor(**params)

        elif model_name == 'XGB':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 70, 500),
                'max_depth': trial.suggest_int('max_depth', 1, 30),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2)
            }
            model = XGBRegressor(**params)

        score = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        return -score.mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    best_params = study.best_params

    if model_name == 'RF':
        return RandomForestRegressor(**best_params)
    elif model_name == 'ET':
        return ExtraTreesRegressor(**best_params)
    else:
        return XGBRegressor(**best_params)

# =====================
# 4. 训练模型 + 评估
# =====================
models = {}
results = []

for name in ['RF', 'ET', 'XGB']:
    model = tune_model(name)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    models[name] = model
    results.append((name, rmse))

    print(f"{name} RMSE: {rmse:.4f}")

# =====================
# 5. 保存模型
# =====================
for name, model in models.items():
    joblib.dump(model, f'{name}_model.pkl')

# =====================
# 新增：输出最优模型参数 + R² / MSE / RMSE 并保存到 CSV
# =====================
from sklearn.metrics import r2_score, mean_squared_error

print("\n" + "="*70)
print("           最优模型参数 + 回归指标（R² / MSE / RMSE）")
print("="*70)

params_list = []

for name, model in models.items():
    # 计算测试集指标
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    # 获取最优参数
    best_params = model.get_params()
    
    # 控制台输出
    print(f"\n 模型: {name}")
    print(f"   R²  ：{r2:.4f}")
    print(f"   MSE ：{mse:.6f}")
    print(f"   RMSE：{rmse:.4f}")
    print(f"   最优参数：{best_params}")
    
    # 保存到字典（用于导出CSV）
    params_list.append({
        "model": name,
        "R2": round(r2, 4),
        "MSE": round(mse, 6),
        "RMSE": round(rmse, 4),
        **best_params  # 把所有参数展开存入同一行
    })

# 保存到 CSV
params_df = pd.DataFrame(params_list)
params_df.to_csv("best_model_params_with_metrics.csv", index=False, encoding="utf-8-sig")

print("\n" + "="*70)
print("已保存：best_model_params_with_metrics.csv")
print("="*70 + "\n")
# =====================
# 6. 统一约束函数（预测结果限制在[0,1]）
# =====================

def predict_clipped(model, x):
    y_pred = model.predict([x])[0]
    # 强制限制在[0,1]
    return float(np.clip(y_pred, 0.0, 1.0))

# =====================
# 7. PSO 反向预测（每模型多解）
# =====================

target_y = float(input("输入目标y(建议0~1): "))

lb = [0, 0.1, 0]
ub = [6, 1, 3]

all_results = []

for name, model in models.items():
    print(f"\nPSO 反演模型: {name}")

    for i in range(5):
        def objective(x):
            y_hat = predict_clipped(model, x)
            return abs(y_hat - target_y)

        best_x, _ = pso(objective, lb, ub)
        pred_y = predict_clipped(model, best_x)
        error = abs(pred_y - target_y)

        all_results.append([name, 'PSO', best_x[0], best_x[1], best_x[2], pred_y, error])

# =====================
# 8. GA 反向预测（对比模块）
# =====================

for name, model in models.items():
    print(f"\nGA 反演模型: {name}")

    def fitness_func(ga_instance, solution, solution_idx):
        y_hat = predict_clipped(model, solution)
        # pygad 是最大化问题 -> 取负误差
        return -abs(y_hat - target_y)

    ga_instance = pygad.GA(
        num_generations=50*2,
        num_parents_mating=10,
        sol_per_pop=20,
        num_genes=3,
        init_range_low=[lb[0], lb[1], lb[2]],
        init_range_high=[ub[0], ub[1], ub[2]],
        fitness_func=fitness_func,
        mutation_percent_genes=20
    )

    ga_instance.run()

    solution, fitness, _ = ga_instance.best_solution()
    pred_y = predict_clipped(model, solution)
    error = abs(pred_y - target_y)

    all_results.append([name, 'GA', solution[0], solution[1], solution[2], pred_y, error])

# =====================
# 9. 输出结果 CSV
# =====================

result_df = pd.DataFrame(all_results, columns=['model','method','period','f','h','predicted_y','error'])
result_df = result_df.sort_values('error')
result_df.to_csv('inverse_all_models_with_GA.csv', index=False)

print("PSO + GA 反演完成，已保存 inverse_all_models_with_GA.csv")

# =====================
# 10. Flask Web
# =====================

from flask import Flask, request, jsonify

app = Flask(__name__)

rf_model = joblib.load('RF_model.pkl')
et_model = joblib.load('ET_model.pkl')
xgb_model = joblib.load('XGB_model.pkl')

models_web = {'RF': rf_model, 'ET': et_model, 'XGB': xgb_model}

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    target_y = float(data['target_y'])
    model_name = data['model']
    method = data.get('method', 'PSO')

    model = models_web[model_name]

    if method == 'PSO':
        def objective(x):
            return abs(predict_clipped(model, x) - target_y)

        best_x, _ = pso(objective, lb, ub)
        pred_y = predict_clipped(model, best_x)

    else:  # GA
        def fitness_func(ga_instance, solution, solution_idx):
            return -abs(predict_clipped(model, solution) - target_y)

        ga_instance = pygad.GA(
            num_generations=30*2,
            num_parents_mating=8,
            sol_per_pop=16,
            num_genes=3,
            init_range_low=lb,
            init_range_high=ub,
            fitness_func=fitness_func
        )
        ga_instance.run()
        best_x, _, _ = ga_instance.best_solution()
        pred_y = predict_clipped(model, best_x)

    return jsonify({
        'period': float(best_x[0]),
        'f': float(best_x[1]),
        'h': float(best_x[2]),
        'predicted_y': float(pred_y)
    })

if __name__ == '__main__':
    app.run(debug=True)

# =====================
# 11. 前端 HTML（index.html）
# =====================

"""
<!DOCTYPE html>
<html>
<head><title>反向设计系统</title></head>
<body>
<h2>输入目标 y (0~1)</h2>
<input id="y" type="number" step="0.01">

<select id="model">
<option value="RF">RF</option>
<option value="ET">ET</option>
<option value="XGB">XGB</option>
</select>

<select id="method">
<option value="PSO">PSO</option>
<option value="GA">GA</option>
</select>

<button onclick="predict()">计算</button>
<p id="result"></p>

<script>
function predict(){
fetch('http://127.0.0.1:5000/predict',{
method:'POST',
headers:{'Content-Type':'application/json'},
body:JSON.stringify({
target_y:parseFloat(document.getElementById('y').value),
model:document.getElementById('model').value,
method:document.getElementById('method').value
})
})
.then(res=>res.json())
.then(data=>{
document.getElementById('result').innerHTML =
`period=${data.period}, f=${data.f}, h=${data.h}, y=${data.predicted_y}`;
});
}
</script>
</body>
</html>
"""
