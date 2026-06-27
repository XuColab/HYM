import numpy as np
import pandas as pd
import joblib
import pygad
import warnings
warnings.filterwarnings('ignore')

# =====================
# 加载训练好的模型
# =====================
models = {
    'RF': joblib.load('RF_model.pkl'),
    'ET': joblib.load('ET_model.pkl'),
    'XGB': joblib.load('XGB_model.pkl')
}

# =====================
# 预测函数（限制输出 [0,1]）
# =====================
def predict_clipped(model, x):
    x = np.array(x).reshape(1, -1)
    return float(np.clip(model.predict(x)[0], 0.0, 1.0))

# =====================
# 自定义 PSO（避免第三方库版本问题）
# =====================
def my_pso(objective, lb, ub, swarmsize=30, maxiter=100):
    """
    简单的粒子群优化算法
    :param objective: 目标函数，接受一个位置向量，返回标量
    :param lb: 下界列表
    :param ub: 上界列表
    :param swarmsize: 粒子数
    :param maxiter: 最大迭代次数
    :return: (best_position, best_value)
    """
    dim = len(lb)
    # 初始化粒子位置和速度
    pos = np.random.uniform(lb, ub, (swarmsize, dim))
    vel = np.random.uniform(-1, 1, (swarmsize, dim)) * (np.array(ub) - np.array(lb)) * 0.1
    pbest_pos = pos.copy()
    pbest_val = np.array([objective(p) for p in pos])
    gbest_idx = np.argmin(pbest_val)
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_val = pbest_val[gbest_idx]

    w = 0.7  # 惯性权重
    c1 = 1.5
    c2 = 1.5

    for _ in range(maxiter):
        r1, r2 = np.random.rand(dim), np.random.rand(dim)
        vel = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (gbest_pos - pos)
        pos = pos + vel
        # 边界处理
        pos = np.clip(pos, lb, ub)
        # 评估新位置
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
# 边界定义（根据要求修改）
# =====================
# 顺序: class, alpha, period, f, h
# class 为离散值 {1,2}，在优化时连续取值然后取整
# alpha 假设范围 [0.1, 1.0]
# period 假设范围 [0, 3]
# f 最大值不超过 0.78
# h 假设范围 [0, 5]
lb_cont = [1.0, 0.1, 0.0, 0.0, 0.0]   # 连续下界
ub_cont = [2.0, 1.0, 3.0, 0.78, 5.0] # 连续上界

# 用户输入
target_y = float(input("输入目标 y (0~1): "))

all_results = []

# =====================
# PSO 反演（每个模型5次，class取整）
# =====================
for name, model in models.items():
    print(f"\nPSO 反演模型: {name}")
    for i in range(5):
        def objective(x):
            # x: [class_cont, alpha, period, f, h]
            x_int = x.copy()
            x_int[0] = np.round(x[0])  # class 取整
            x_int[0] = np.clip(x_int[0], 1, 2)  # 确保在1或2
            return abs(predict_clipped(model, x_int) - target_y)

        best_x_cont, best_val = my_pso(objective, lb_cont, ub_cont, swarmsize=30, maxiter=100)
        # 最终解（class取整）
        best_x = best_x_cont.copy()
        best_x[0] = int(np.round(best_x_cont[0]))
        best_x[0] = np.clip(best_x[0], 1, 2)
        pred_y = predict_clipped(model, best_x)
        error = abs(pred_y - target_y)

        all_results.append([name, 'PSO', 
                            best_x[0], best_x[1], best_x[2], best_x[3], best_x[4],
                            pred_y, error])

# =====================
# GA 反演（class 直接作为离散基因）
# =====================
for name, model in models.items():
    print(f"\nGA 反演模型: {name}")

    def fitness_func(ga_instance, solution, solution_idx):
        # solution: [class, alpha, period, f, h]
        # class 已经是整数（1或2）
        y_hat = predict_clipped(model, solution)
        return -abs(y_hat - target_y)  # 最大化负误差

    # GA 配置：
    # - class: 离散选择 [1,2]
    # - alpha: 连续 [0.1, 1.0]
    # - period: 连续 [0, 3]
    # - f: 连续 [0, 0.78]
    # - h: 连续 [0, 5]
    ga_instance = pygad.GA(
        num_generations=100,
        num_parents_mating=10,
        sol_per_pop=30,
        num_genes=5,
        gene_space=[
            [1, 2],                 # class: 离散整数 1 或 2
            {'low': 0.1, 'high': 1.0},
            {'low': 0.0, 'high': 3.0},
            {'low': 0.0, 'high': 0.78},
            {'low': 0.0, 'high': 5.0}
        ],
        fitness_func=fitness_func,
        mutation_percent_genes=20,
        mutation_type="random",
        crossover_type="single_point",
        parent_selection_type="sss",
        keep_elitism=2,
        random_seed=42
    )
    ga_instance.run()
    solution, fitness, _ = ga_instance.best_solution()
    # 确保 class 是整数（GA 可能返回浮点，但空间定义为整数一般会遵守）
    solution[0] = int(np.round(solution[0]))
    pred_y = predict_clipped(model, solution)
    error = abs(pred_y - target_y)

    all_results.append([name, 'GA',
                        solution[0], solution[1], solution[2], solution[3], solution[4],
                        pred_y, error])

# 保存结果
columns = ['model', 'method', 'class', 'alpha', 'period', 'f', 'h', 'predicted_y', 'error']
df_result = pd.DataFrame(all_results, columns=columns)
df_result = df_result.sort_values('error')
df_result.to_csv('inverse_results.csv', index=False, encoding='utf-8-sig')
print("\n反演完成，结果保存为 inverse_results.csv")