import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'Noto Sans CJK SC']

plt.rcParams['axes.unicode_minus'] = False   # To display minus sign correctly

# Data from the image
data = {
    "模型": ["ERNIE-4.5-8K-Preview", "ERNIE-4.0-Turbo-8K", "ERNIE-3.5", "ERNIE-Speed-Pro", "ERNIE-X1-32K-Preview"],
    "注册会计师": [81.16, 74.75, 65.29, 50.19, 80.17],
    "银行从业资格": [89.07, 86.09, 81.59, 66.16, 88.18],
    "证券从业资格": [82.99, 79.51, 70.92, 58.5, 81.04],
    "基金从业资格": [86.35, 84.4, 74.2, 63.76, 85.67],
    "保险从业资格": [79.89, 78.59, 70.98, 60.49, 77.3],
    "经济师": [94.42, 90.38, 83.65, 77.12, 91.54],
    "税务师": [74.39, 68.61, 58.61, 46.11, 81.97],
    "期货从业资格": [83.87, 80.66, 74.24, 66.1, 88.61],
    "退休规划师": [85.08, 80.68, 74.24, 66.1, 83.05],
    "精算师": [47.73, 37.5, 35.23, 23.86, None]
}

# Convert to DataFrame
df = pd.DataFrame(data)
df.set_index("模型", inplace=True)

# Transpose for plotting
df_t = df.transpose()

# Plot bar chart
ax = df_t.plot(kind='bar', figsize=(16, 8), rot=45)
plt.title('ERNIE模型在各类金融考试中的表现')
plt.ylabel('得分')
plt.xlabel('考试类型')
plt.legend(title='模型', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

plt.show()
