import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
# matplotlib.use('Qt5Agg')  # 或 'TkAgg'
from matplotlib.lines import Line2D

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
def get_data():
    TYPE = [(0, 0.4), (0, 0.6), (0, 0.8),
            (0.2, 0.4), (0.2, 0.6), (0.2, 0.8),
            (0.4, 0.4), (0.4, 0.6), (0.4, 0.8),
            (0.6, 0.4), (0.6, 0.6), (0.6, 0.8),
            (0.8, 0.4), (0.8, 0.6), (0.8, 0.8)]
    best_x_data = pd.read_excel('../Result/Analysis/best_x_result.xlsx')
    IW_data = pd.DataFrame([])
    S_data = pd.DataFrame([])

    for index, type in enumerate(TYPE):
        start_index = index * 21
        end_index = start_index + 16
        data = best_x_data.loc[start_index:end_index - 1]
        iw = data[3].values.reshape(-1, 1)
        s = data[[0, 1, 2]].values
        s = pd.DataFrame(np.sum(s, axis=1))
        iw = pd.DataFrame(iw)
        IW_data = pd.concat((IW_data, iw), axis=1)
        S_data = pd.concat((S_data, s), axis=1)
    IW_data.columns = TYPE
    S_data.columns = TYPE
    return IW_data,S_data
def plot_box(data,type):

    # 绘制箱线图
    # 设置更宽的图形尺寸
    plt.figure(figsize=(max(10, len(data.columns) * 1), 6))  # 动态调整宽度

    box = data.boxplot(
        patch_artist=True,
        boxprops=dict(facecolor='lightgreen', color='green',alpha = 0.5),
        medianprops=dict(color='red'),
        showfliers=False,  # 不显示异常值
    )
    min_mid_num = 0xffff
    max_mid_num = -0xffff
    # 添加散点
    for idx, column in enumerate(data.columns):

        # 计算四分位数和IQR
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        # 确定异常值范围
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 剔除异常值
        # filtered_data = data[column][(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        filtered_data = data[column]
        mid = np.median(filtered_data)
        if mid < min_mid_num:
            min_mid_num = mid
        elif mid > max_mid_num:
            max_mid_num = mid
        # 使用索引作为x
        x = np.full(len(filtered_data), idx + 1)
        plt.scatter(x, filtered_data, color='black', alpha=0.5)
    print(min_mid_num,max_mid_num)
    # 添加标题和标签
    plt.title('不同组合的 {} 箱线图'.format(type))
    plt.ylabel('值')
    plt.xlabel('组合')
    plt.xticks(ticks=np.arange(1, len(data.columns) + 1), labels=[str(col) for col in data.columns],
               rotation=45)  # 设置x轴标签
    plt.tight_layout()

    folder_path = '../Result/Analysis'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig('{}/不同组合的 {} 箱线图.png'.format(folder_path,type))
    plt.show()
def plot_3D_best(g1_true,g2_true,g3_true):
    data = pd.read_excel('../Result/Analysis/best_y_result.xlsx')
    data = data[[data.columns[1],data.columns[2],data.columns[3]]].values
    data[:,2] = data[:,2] **2 # 调整
    G1,G2,G3 = data[:,0],data[:,1],data[:,2]
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111,projection = '3d')
    # 指定16种不同颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'cyan', 'magenta', 'yellow', 'black', 'pink',
              'brown', 'gray', 'lime', 'teal', 'navy', 'gold']

    # 绘制数据点并创建标签
    labels = []
    for index1, i in enumerate([0, 0.2, 0.4, 0.6, 0.8]):
        for index2, j in enumerate([0.4, 0.6, 0.8]):
            index = index1 * 3 + index2
            type_label = '({},{})'.format(i, j)
            g1, g2, g3 = G1[index], G2[index], G3[index]
            ax.scatter(g1, g2, g3, c=colors[index])
            labels.append((colors[index], type_label))

    # 绘制现状点
    ax.scatter(g1_true, g2_true, g3_true, c=colors[15])
    labels.append((colors[15], '现状'))

    # 创建自定义图例
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=color, markersize=10)
                       for color, label in labels]

    # 添加图例到图形
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title('Compare Plot Best')
    ax.set_xlabel('G1 axis')
    ax.set_ylabel('G2 axis')
    ax.set_zlabel('G3 axis')
    folder_path = '../Result/Analysis'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig('{}/Compare Plot Best.png'.format(folder_path))
    plt.show()


def plot_3D_all(g1_true,g2_true,g3_true):
    plt.ion()  # 开启交互模式
    # 指定16种不同颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple',
              'cyan', 'magenta', 'yellow', 'black', 'pink',
              'brown', 'gray', 'lime', 'teal', 'navy', 'gold']
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # 绘制数据点并创建标签
    labels = []
    for index1, i in enumerate([0, 0.2, 0.4, 0.6, 0.8]):
        for index2, j in enumerate([0.4, 0.6, 0.8]):
            index = index1 * 3 + index2
            type_label = '({},{})'.format(i, j)
            filename = '../Result/{}-{}-y_result.xlsx'.format(i, j)
            data = pd.read_excel(filename)
            data = data.drop([data.columns[0]],axis=1).values
            G1,G2,G3 = data[:,0],data[:,1],data[:,2]
            ax.scatter(G1, G2, G3, c=colors[index])
            labels.append((colors[index], type_label))
    # 绘制现状点
    ax.scatter(g1_true, g2_true, g3_true, c=colors[15])
    labels.append((colors[15], '现状'))

    # 创建自定义图例
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                              markerfacecolor=color, markersize=10)
                       for color, label in labels]

    # 添加图例到图形
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title('Compare Plot All')
    ax.set_xlabel('G1 axis')
    ax.set_ylabel('G2 axis')
    ax.set_zlabel('G3 axis')
    folder_path = '../Result/Analysis'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig('{}/Compare Plot All.png'.format(folder_path))
    plt.show()
g1_true,g2_true,g3_true = 19670136623.873913,4.81265426132003,4.694839349506769
IW_data,S_data = get_data()
# 绘制箱线图图表
print('IW:',end='')
plot_box(IW_data,'IW')
print('S:',end='')
plot_box(S_data,'S')
plt.tight_layout()
plt.show(block=True)

# 绘制3D图:best
plot_3D_best(g1_true,g2_true,g3_true)

# 绘制3D图:all
plot_3D_all(g1_true,g2_true,g3_true)
