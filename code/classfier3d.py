from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================
# Task 2: 3D Decision Boundary (Binary Classification)
# ============================

# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# ----------------------------
# 只选择两类花（Versicolor 和 Virginica）
# 原始标签：
# 0 - Setosa
# 1 - Versicolor
# 2 - Virginica
# ----------------------------
binary_mask = (y == 1) | (y == 2)
X = X[binary_mask]
y = y[binary_mask]

# 将标签转换为 0 和 1（便于二分类）
y = (y == 2).astype(int)

# ----------------------------
# 选择三个特征进行 3D 可视化
# 0: sepal length
# 2: petal length
# 3: petal width
# ----------------------------
X_3d = X[:, [0, 2, 3]]

# ----------------------------
# 训练逻辑回归模型
# ----------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_3d, y)

# ----------------------------
# 创建 3D 网格点，用于预测分类结果
# ----------------------------
x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
z_min, z_max = X_3d[:, 2].min() - 0.5, X_3d[:, 2].max() + 0.5

xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, 20),
    np.linspace(y_min, y_max, 20),
    np.linspace(z_min, z_max, 20)
)

# 将网格点组合成模型输入
grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# 对网格点进行预测
pred = model.predict(grid_points)

# ----------------------------
# 3D 可视化
# ----------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
scatter = ax.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=y,
    cmap='coolwarm',
    edgecolors='k',
    s=50
)

# 绘制预测分类结果（表示决策边界区域）
ax.scatter(
    grid_points[:, 0],
    grid_points[:, 1],
    grid_points[:, 2],
    c=pred,
    cmap='coolwarm',
    alpha=0.1
)

# 坐标轴标签
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Petal Length')
ax.set_zlabel('Petal Width')

ax.set_title('3D Decision Boundary (Binary Classification)')

plt.show()
