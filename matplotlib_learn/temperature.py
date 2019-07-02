# 绘制温度趋势图
# 对于可以绘制哪些图像可以在matplotlib查看
from matplotlib import pyplot as plt
import random
import matplotlib
from matplotlib import font_manager

# 显示中文
# window和linux设置显示字体
# font = {'family': 'monospace',
#         'weight': 'bold',
#         'size': 'larger'}
# matplotlib.rc("font", **font)
# 另外一种设置字体大小大方式
my_font = font_manager.FontProperties(fname="字体路径")

x = range(0, 120)
y_1 = [random.randint(20, 35) for i in range(120)]
y_2 = [random.randint(20, 35) for i in range(120)]


# 设置图像大小，figsize设置宽和高，dpi设置每一刻度的像素
plt.figure(figsize=(20, 8), dpi=80)

# 一张图像中显示两条折线,alpha设置透明度
plt.plot(x, y_1, lable="y_1",
         color='r',
         linestyle='--',
         linewidth=5,
         alpha=0.5)
plt.plot(x, y_2, lable="y_2")

# 调整x轴刻度,_x设置为数组
_x = list(x)
# _x = list(x)[::10]
# _xtick_lables = ["hello,{}".format(i) for i in _x]
# 中文显示时间
_xtick_lables = ["10点{}分".format(i) for i in range(60)]
_xtick_lables += ["11点{}分".format(i - 60) for i in range(60, 120)]
# list才可以使用[]
# 前后一一对应，rotation表示旋转度数，仅仅是字体旋转
plt.xticks(_x[::3], _xtick_lables[::3], rotation=90, fontproperties=my_font)
plt.yticks(list(y))

# 添加描述信息
plt.xlabel("时间", fontproperties=my_font)
plt.ylabel("温度 单位(℃)", fontproperties=my_font)
plt.title("10点到12点每分钟的气温变化情况", fontproperties=my_font)

# 绘制网格,根据xy轴稀疏度变化
plt.grid()

# 添加图例，多看源码
plt.legend(prop=my_font, loc='upper left')

plt.show()
