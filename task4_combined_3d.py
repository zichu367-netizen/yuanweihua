# task4_combined_3d.py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from matplotlib import cm
from matplotlib.colors import ListedColormap

# ============================
# 加载并准备数据
# ============================
iris = load_iris()
X = iris.data
y = iris.target

# 只选择两类（Versicolor 和 Virginica）
mask = (y == 1) | (y == 2)
X = X[mask]
y = y[mask]
y = (y == 2).astype(int)  # 转换为二分类标签

# 选择三个特征
X_3d = X[:, [0, 2, 3]]  # 萼片长度, 花瓣长度, 花瓣宽度

# ============================
# 训练模型
# ============================
model = LogisticRegression(max_iter=200)
model.fit(X_3d, y)

# ============================
# 创建网格
# ============================
x_min, x_max = X_3d[:, 0].min() - 0.5, X_3d[:, 0].max() + 0.5
y_min, y_max = X_3d[:, 1].min() - 0.5, X_3d[:, 1].max() + 0.5
z_min, z_max = X_3d[:, 2].min() - 0.5, X_3d[:, 2].max() + 0.5

xx, yy, zz = np.meshgrid(
    np.linspace(x_min, x_max, 25),
    np.linspace(y_min, y_max, 25),
    np.linspace(z_min, z_max, 25)
)

grid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# ============================
# 预测
# ============================
probs = model.predict_proba(grid_points)[:, 1]  # 属于Virginica的概率
predictions = model.predict(grid_points)

# ============================
# 创新可视化：3D边界 + 概率图
# ============================
fig = plt.figure(figsize=(18, 8))

# 子图1：3D决策边界（带透明度）
ax1 = fig.add_subplot(121, projection='3d')

# 用颜色表示预测类别，用透明度表示概率置信度
alpha_values = np.abs(probs - 0.5) * 2  # 越接近0.5越透明，越接近0或1越不透明
alpha_values = np.clip(alpha_values, 0.1, 0.8)  # 限制透明度范围

# 绘制预测网格点（带透明度）
sc1 = ax1.scatter(
    grid_points[:, 0],
    grid_points[:, 1],
    grid_points[:, 2],
    c=predictions,
    cmap='coolwarm',
    alpha=alpha_values.reshape(-1) * 0.4,  # 整体降低透明度
    s=10
)

# 绘制原始数据点
ax1.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=y,
    cmap='coolwarm',
    edgecolors='k',
    s=80,
    depthshade=True,
    label='Actual Data'
)

# 突出显示决策边界附近的点（概率接近0.5）
boundary_mask = (probs > 0.45) & (probs < 0.55)
if boundary_mask.any():
    ax1.scatter(
        grid_points[boundary_mask, 0],
        grid_points[boundary_mask, 1],
        grid_points[boundary_mask, 2],
        c='yellow',
        alpha=0.6,
        s=20,
        label='Decision Boundary'
    )

ax1.set_xlabel('Sepal Length (cm)')
ax1.set_ylabel('Petal Length (cm)')
ax1.set_zlabel('Petal Width (cm)')
ax1.set_title('3D Decision Boundary with Confidence')
ax1.legend()

# 子图2：3D概率图（带等值面效果）
ax2 = fig.add_subplot(122, projection='3d')

# 创建自定义颜色映射（从蓝色到红色）
colors = np.zeros((len(probs), 4))
colors[:, 0] = probs  # 红色通道：概率值
colors[:, 2] = 1 - probs  # 蓝色通道：1-概率值
colors[:, 3] = 0.3  # 固定透明度

sc2 = ax2.scatter(
    grid_points[:, 0],
    grid_points[:, 1],
    grid_points[:, 2],
    c=probs,
    cmap='RdBu_r',
    alpha=0.2,
    s=20
)

# 绘制原始数据点
ax2.scatter(
    X_3d[:, 0],
    X_3d[:, 1],
    X_3d[:, 2],
    c=y,
    cmap='coolwarm',
    edgecolors='k',
    s=80,
    depthshade=True
)

# 添加概率为0.5的等值面（决策边界）
# 由于matplotlib 3D限制，我们添加一个概率为0.5的平面示意
xx_surface, yy_surface = np.meshgrid(
    np.linspace(x_min, x_max, 10),
    np.linspace(y_min, y_max, 10)
)

# 为简单起见，我们创建一个位于平均z值的平面
z_surface = np.ones_like(xx_surface) * ((z_min + z_max) / 2)

ax2.plot_surface(xx_surface, yy_surface, z_surface, 
                alpha=0.3, color='yellow', label='p=0.5 Boundary')

ax2.set_xlabel('Sepal Length (cm)')
ax2.set_ylabel('Petal Length (cm)')
ax2.set_zlabel('Petal Width (cm)')
ax2.set_title('3D Probability Map with Decision Plane')

plt.suptitle('Task 4: Combined 3D Visualization\nDecision Boundary + Probability Map', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('task4_combined_3d.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================
# 额外：创建2D截面图
# ============================
fig2, axes = plt.subplots(2, 2, figsize=(12, 10))

# 固定每个维度，查看截面
z_fixed = np.median(X_3d[:, 2])  # 固定花瓣宽度

# 创建2D网格
xx_2d, yy_2d = np.meshgrid(
    np.linspace(x_min, x_max, 100),
    np.linspace(y_min, y_max, 100)
)

# 预测2D截面上的概率
z_values = np.ones_like(xx_2d.ravel()) * z_fixed
grid_2d = np.c_[xx_2d.ravel(), yy_2d.ravel(), z_values]
probs_2d = model.predict_proba(grid_2d)[:, 1].reshape(xx_2d.shape)

# 子图1：2D概率热图
im1 = axes[0, 0].imshow(
    probs_2d,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    cmap='RdBu_r',
    aspect='auto',
    alpha=0.7
)
plt.colorbar(im1, ax=axes[0, 0])
axes[0, 0].scatter(X_3d[:, 0], X_3d[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
axes[0, 0].set_xlabel('Sepal Length (cm)')
axes[0, 0].set_ylabel('Petal Length (cm)')
axes[0, 0].set_title(f'2D Slice at Petal Width = {z_fixed:.2f} cm')

# 子图2：决策边界等高线
axes[0, 1].contour(xx_2d, yy_2d, probs_2d, levels=[0.5], colors='yellow', linewidths=2)
axes[0, 1].contourf(xx_2d, yy_2d, probs_2d, levels=20, cmap='RdBu_r', alpha=0.5)
axes[0, 1].scatter(X_3d[:, 0], X_3d[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=50)
axes[0, 1].set_xlabel('Sepal Length (cm)')
axes[0, 1].set_ylabel('Petal Length (cm)')
axes[0, 1].set_title('Decision Boundary Contour')

# 子图3：概率分布直方图
axes[1, 0].hist(probs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 0].axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
axes[1, 0].set_xlabel('Predicted Probability (Virginica)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Distribution of Predicted Probabilities')
axes[1, 0].legend()

# 子图4：模型性能指标
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
y_pred = model.predict(X_3d)
accuracy = accuracy_score(y, y_pred)
conf_matrix = confusion_matrix(y, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, 
                              display_labels=['Versicolor', 'Virginica'])
disp.plot(ax=axes[1, 1], cmap='Blues')
axes[1, 1].set_title(f'Confusion Matrix (Accuracy: {accuracy:.3f})')

plt.suptitle('Additional 2D Analysis and Model Evaluation', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('task4_additional_analysis.png', dpi=300, bbox_inches='tight')
plt.show()