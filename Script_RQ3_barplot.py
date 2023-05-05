import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

path_to_datasets = 'datasets/'
path_to_saved_csvs = 'output_csvs/dataset_complexity/'

data_normalizations = ["standard", "min-max"]
for normalization_type in data_normalizations:
    dcm_df = pd.read_csv(path_to_saved_csvs + "{}_norm_dc_measures.csv".format(normalization_type))
    dcm_df = dcm_df.loc[:, []]
    print()


# 创建一个Pandas数据框
methods = ['Raw', 'Norm', 'Norm+FS', 'Norm+FS+RS']
performance = ['F1', 'N2', 'C1', 'BI3']

means = [[0.778, 0.622, 0.359, 0.246],
         [0.778, 0.672, 0.359, 0.238],
         [0.709, 0.478, 0.000, 0.000],
         [0.732, 0.470, 0.000, 0.000]
         ]

stds = [[0.140, 0.171, 0.261, 0.201],
        [0.140, 0.151, 0.261, 0.200],
        [0.122, 0.010, 0.000, 0.000],
        [0.119, 0.015, 0.000, 0.000]
        ]

df = pd.DataFrame(means, index=methods, columns=performance)
df_std = pd.DataFrame(stds, index=methods, columns=performance)

# 格式化数据
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=df.columns)
df_std_melt = pd.melt(df_std.reset_index(), id_vars=['index'], value_vars=df_std.columns)

# 绘制条形图
plt.figure(figsize=(15, 5), dpi=500)
sns.set(style="darkgrid", palette="muted")
ax = sns.barplot(x="variable", y="value", hue="index", data=df_melt)

# 添加误差线
for i, artist in enumerate(ax.containers):

    # 在每个条形上方添加文本
    for j, bar in enumerate(artist):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01, '{:.3f}'.format(height),
                ha='center', va='bottom', fontsize=8, color='black')

        # 添加误差线
        x = bar.get_x() + bar.get_width() / 2.
        y = height
        std_val = df_std_melt.iloc[i * 4 + j]['value']  # 获取标准差值
        ax.errorbar(x, y, yerr=std_val, color='black', capsize=3)

# # 添加均值标注
# for i, artist in enumerate(ax.containers):
#     for j in range(i * 4, i * 4 + 4):
#         mean_val = df_melt.iloc[j]['value']
        # plt.text(j, mean_val + 0.1, f"{mean_val:.2f}", ha='center', va='bottom', fontsize=8)

# 设置图例位置
plt.legend()

# 添加标签和标题
# plt.xlabel('Performance Metrics')
plt.ylabel('Mean')
# plt.title('Method Performance')

# 显示图形
plt.show()