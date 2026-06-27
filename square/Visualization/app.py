from flask import Flask, jsonify, render_template
import pandas as pd
import numpy as np
from scipy.interpolate import Rbf

app = Flask(__name__)

# ======================
# 1. 数据处理
# ======================
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(base_dir, "data.csv")

df = pd.read_csv(csv_path)

df.columns = ["pitch", "a", "h", "T"]

df_clean = df.groupby(["pitch", "a", "h"], as_index=False).mean()

x = df_clean["pitch"].values
y = df_clean["a"].values
z = df_clean["h"].values
v = df_clean["T"].values

# ======================
# 2. 插值（RBF）
# ======================
grid_x, grid_y, grid_z = np.mgrid[
    x.min():x.max():20j,
    y.min():y.max():20j,
    z.min():z.max():20j
]

rbf = Rbf(x, y, z, v)
grid_val = rbf(grid_x, grid_y, grid_z)

# ======================
# 3. API接口
# ======================
@app.route("/data")
def get_data():
    return jsonify({
        "points": [
            {"x": float(x[i]), "y": float(y[i]), "z": float(z[i]), "v": float(v[i])}
            for i in range(len(x))
        ],
        "grid": [
            {"x": float(grid_x.flatten()[i]),
             "y": float(grid_y.flatten()[i]),
             "z": float(grid_z.flatten()[i]),
             "v": float(grid_val.flatten()[i])}
            for i in range(len(grid_val.flatten()))
        ]
    })

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)