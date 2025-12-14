import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# ============================
# 加载 Iris 数据集
# ============================
# 使用 seaborn 自带的 iris 数据集
df = sns.load_dataset('iris')

# 打印部分数据，观察数据结构
print(df[50:100])

# ============================
# 数据预处理
# ============================
# 删除缺失值（iris 数据集实际上没有缺失值）
df = df.dropna()

# 将类别型数据 species 转换为数值（0,1,2）
df['species'] = df['species'].astype('category').cat.codes

# ============================
# 使用 Matplotlib + Seaborn 绘制箱线图
# ============================
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Sepal Length 的箱线图
sns.boxplot(x='species', y='sepal_length', data=df, ax=axes[0, 0])
axes[0, 0].set_title('Sepal Length by Species')

# Sepal Width 的箱线图
sns.boxplot(x='species', y='sepal_width', data=df, ax=axes[0, 1])
axes[0, 1].set_title('Sepal Width by Species')

# Petal Length 的箱线图
sns.boxplot(x='species', y='petal_length', data=df, ax=axes[1, 0])
axes[1, 0].set_title('Petal Length by Species')

# Petal Width 的箱线图
sns.boxplot(x='species', y='petal_width', data=df, ax=axes[1, 1])
axes[1, 1].set_title('Petal Width by Species')

# 调整子图布局
plt.tight_layout()
plt.show()

# ============================
# 使用 Plotly 绘制交互式散点图
# ============================

fig1 = px.scatter(df, x='sepal_length', y='sepal_width',
                  color='species', title="Sepal Length vs Sepal Width")
fig2 = px.scatter(df, x='sepal_length', y='petal_length',
                  color='species', title="Sepal Length vs Petal Length")
fig3 = px.scatter(df, x='sepal_length', y='petal_width',
                  color='species', title="Sepal Length vs Petal Width")
fig4 = px.scatter(df, x='sepal_width', y='petal_length',
                  color='species', title="Sepal Width vs Petal Length")
fig5 = px.scatter(df, x='sepal_width', y='petal_width',
                  color='species', title="Sepal Width vs Petal Width")
fig6 = px.scatter(df, x='petal_length', y='petal_width',
                  color='species', title="Petal Length vs Petal Width")

# 显示交互式图表
fig1.show()
fig2.show()
fig3.show()
fig4.show()
fig5.show()
fig6.show()
