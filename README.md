鸢尾花分类项目
markdown
# 🌸 Iris Flower Classification Project
鸢尾花数据分类与可视化项目

## 📁 项目结构
yuanweihua/

├── code/ # 所有Python代码

│ ├── data_preview.py # 数据探索与可视化

│ ├── classifier2d.py # 任务一：2D分类边界

│ ├── classfier3d.py # 任务二：3D决策边界

│ ├── classifier3d_probability.py # 任务三：3D概率图

│ └── task4_combined_3d.py # 任务四：创新组合可视化

├── figures/ # 所有生成的图片

│ ├── boxplots.png # 数据探索：箱线图

│ ├── original_2d_boundary.png # 任务一：2D决策边界

│ ├── 3d_decision_boundary.png # 任务二：3D决策边界

│ ├── 3d_probability_map.png # 任务三：3D概率图

│ └── task4_combined.png # 任务四：组合可视化

├── project3.pptx # 项目演示PPT

└── README.md # 项目说明文件

text

## 🚀 如何运行
### 1. 安装依赖
```bash
pip install scikit-learn matplotlib numpy seaborn
2. 运行代码
按顺序运行以下文件：

bash
# 数据探索
python code/data_preview.py

# 任务一：2D分类
python code/classifier2d.py

# 任务二：3D决策边界
python code/classfier3d.py

# 任务三：3D概率图
python code/classifier3d_probability.py

# 任务四：创新组合
python code/task4_combined_3d.py
📊 项目任务
✅ 任务一：2D分类边界可视化
使用花瓣长度和花瓣宽度两个特征

对三种鸢尾花（Setosa, Versicolor, Virginica）进行分类

显示决策边界和每个类别的概率分布

https://figures/original_2d_boundary.png

✅ 任务二：3D决策边界
选择萼片长度、花瓣长度、花瓣宽度三个特征

只分类Versicolor和Virginica两种花

在3D空间显示决策边界

https://figures/3d_decision_boundary.png

✅ 任务三：3D概率图
使用和任务二相同的特征

显示每个网格点属于Virginica的概率

颜色深浅表示概率大小

https://figures/3d_probability_map.png

✅ 任务四：创新组合可视化
将3D决策边界和概率图结合起来

显示更丰富的分类信息

突出显示决策边界区域

https://figures/task4_combined.png

🛠️ 技术栈
编程语言: Python 3.x

机器学习库: scikit-learn

数据可视化: matplotlib, seaborn

数据处理: numpy

📚 数据集信息
数据集: Iris鸢尾花数据集

样本数: 150个

特征数: 4个（萼片长度、萼片宽度、花瓣长度、花瓣宽度）

类别: 3种（Setosa, Versicolor, Virginica）

📈 运行结果
所有运行结果图片保存在 figures/ 文件夹中，包括：

boxplots.png - 四个特征的分布情况

original_2d_boundary.png - 2D决策边界

3d_decision_boundary.png - 3D决策边界

3d_probability_map.png - 3D概率图

task4_combined.png - 组合可视化

📝 注意事项
确保安装了所有依赖包

代码按顺序运行以获得最佳结果

所有图片会自动保存到figures文件夹

👨‍🎓 作者信息
姓名: [gaozichun]
github地址
https://github.com/zichu367-netizen/yuanweihua/edit/main

最后更新: 2025年12月14日
