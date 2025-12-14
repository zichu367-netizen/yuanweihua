from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================
# 加载 Iris 数据集
# ============================
iris = load_iris()

# 选择后两个特征（花瓣长度、花瓣宽度）进行二维可视化
X = iris.data[:, 2:]
y = iris.target

# ============================
# 划分训练集和测试集
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================
# 训练逻辑回归模型
# ============================
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# ============================
# 构造二维网格，用于绘制决策边界
# ============================
xx, yy = np.meshgrid(
    np.arange(X[:, 0].min() - 1, X[:, 0].max() + 1, 0.1),
    np.arange(X[:, 1].min() - 1, X[:, 1].max() + 1, 0.1)
)

# ============================
# 计算每个网格点的预测概率
# ============================
probs = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
probs = probs.reshape(xx.shape[0], xx.shape[1], 3)

# 每个类别对应一种固定颜色
class_colors = ['yellow', 'green', 'blue']

# ============================
# 创建画布（1 个决策边界 + 3 个概率图）
# ============================
fig, axs = plt.subplots(1, 4, figsize=(20, 5))

# ----------------------------
# 1. 整体决策边界图
# ----------------------------
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axs[0].imshow(
    Z,
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    origin='lower',
    cmap=plt.cm.colors.ListedColormap(class_colors),
    alpha=0.6
)

axs[0].scatter(
    X[:, 0], X[:, 1],
    c=y,
    edgecolors='k',
    marker='o',
    s=50,
    cmap=mcolors.ListedColormap(class_colors)
)

axs[0].set_title('Overall Decision Boundaries')
axs[0].set_xlabel('Petal Length')   # 修正：原代码这里写错
axs[0].set_ylabel('Petal Width')    # 修正：原代码这里写错

# ----------------------------
# 2~4. 各类别的概率分布图
# ----------------------------
for i, class_prob in enumerate(probs.transpose(2, 0, 1)):
    ax = axs[i + 1]

    # 为每个类别创建渐变色图
    cmap = mcolors.LinearSegmentedColormap.from_list(
        f'class_{i}_colormap',
        ['white', class_colors[i]],
        N=256
    )

    contour = ax.contourf(xx, yy, class_prob, alpha=0.7, cmap=cmap)

    ax.scatter(
        X[:, 0], X[:, 1],
        c=y,
        edgecolors='k',
        marker='o',
        s=50,
        cmap=mcolors.ListedColormap(class_colors)
    )

    fig.colorbar(contour, ax=ax)

    ax.set_title(f'Class {i} Probability')
    ax.set_xlabel('Petal Length')
    ax.set_ylabel('Petal Width')

plt.tight_layout()
plt.show()
