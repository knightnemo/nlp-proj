import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 数据准备
data = {
    'Model': ['TinyLlama (1.1B)', 'Mistral 7B', 'Llama 11B', 'Qwen 32B', 'Llama3 70b', 'Llama3 405b', 'GPT3.5', 'GPT4o'],
    'Total': [0, 0, 0, 21.4, 37.5, 40.1, 44.75, 58.2],
    'Unchanged': [0, 0, 0, 33.3, 36.1, 44.4, 58.9, 63.9],
    'Changed': [0, 0, 0, 0, 40, 30, 21.2, 50]
}

# 创建 DataFrame
df = pd.DataFrame(data)

# 设置图形大小
plt.figure(figsize=(12, 7))

# 绘制条形图
bar_width = 0.25
x = np.arange(len(df))

plt.bar(x, df['Total'], width=bar_width, label='Total', color='lightblue', align='center')
plt.bar(x + bar_width, df['Unchanged'], width=bar_width, label='Unchanged', color='lightgreen', align='center')
plt.bar(x + bar_width * 2, df['Changed'], width=bar_width, label='Changed', color='salmon', align='center')

# 拟合曲线
for i, column in enumerate(['Total', 'Unchanged', 'Changed']):
    # 使用高阶多项式进行拟合
    z = np.polyfit(x, df[column], 3)  # 使用3次多项式拟合
    p = np.poly1d(z)

    # 生成拟合曲线的 x 值
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = p(x_fit)  # 使用拟合的多项式
    # 绘制拟合曲线
    plt.plot(x_fit, y_fit, label=f'{column} Fit', linestyle='--')

# 添加标签和标题
plt.xlabel('Model')
plt.ylabel('Percentage (%)')
plt.title('Emergent Abilities of Different Models')
plt.xticks(x + bar_width, df['Model'], rotation=45, ha='right')
plt.legend()

# 保存图形到文件
plt.tight_layout()
plt.savefig('emergent_abilities_with_fit.png')
plt.close()