import numpy as np
import matplotlib.pyplot as plt

# 生成两个5000个数的随机数组
np.random.seed(42)  # 设置随机种子保证可复现
array1 = np.random.normal(loc=0, scale=1, size=5000)  # 正态分布数据
array2 = np.random.exponential(scale=1, size=5000)     # 指数分布数据

# 创建直方图对比
plt.figure(figsize=(12, 6))

# 绘制第一个数组的直方图
plt.hist(array1, bins=50, alpha=0.5, label='Array1 (Normal)', color='blue')

# 绘制第二个数组的直方图
plt.hist(array2, bins=50, alpha=0.5, label='Array2 (Exponential)', color='red')

# 添加图表元素
plt.title('Comparison of Two 5000-Element Arrays')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.legend()

plt.show()