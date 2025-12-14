from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================
# Task 3: 3D Probability Map (Binary Classification)
# ============================

# 加载 Iris 数据集
iris = load_iris()
X = iris.data
y = iris.target

# 只选择两类（Versicolor 和 Virginica）
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]

# 转换为二分类标签（0 / 1）
y = (y == 2).astype(int)

# 选择三个特征
X_3d = X[:, [0, 2, 3]]

# 训练逻辑回归模型
model = LogisticRegression(max_iter=200)
model.fit(X_3d, y)

# 构造 3D 网格
x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
z_min, z_max = X_3d[:, 2].min() - 0.5, X_3d[:, 2].max() + 0.5

xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, 20),
    np.linspace(y_min, y_max, 20),
    np.linspace(z_min, z_max, 20)
)

grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# 预测属于“Virginica”的概率
probs = model.predict_proba(grid_points)[:, 1]

# ============================
# 3D 概率可视化
# ============================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 原始数据点
ax.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=y,
    cmap='coolwarm',
    edgecolors='k',
    s=50
)

# 概率分布（颜色深浅表示概率大小）
sc = ax.scatter(
    grid_points[:, 0],
    grid_points[:, 1],
    grid_points[:, 2],
    c=probs,
    cmap='viridis',
    alpha=0.15
)

fig.colorbar(sc, ax=ax, label='Probability of Virginica')

ax.set_xlabel('Sepal Length')
ax.set_ylabel('Petal Length')
ax.set_zlabel('Petal Width')
ax.set_title('3D Probability Map (Binary Classification)')

plt.show()
