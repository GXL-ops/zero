#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 22:25
# @Author  : GXL
# @Site    : 
# @File    : numpy_5.py
# @Software: PyCharm

import numpy as np
from matplotlib import pyplot as plt

# 绘制两种评论的直方图
# 加载国家数据
_data_1 = "数据信息来源"
_data_2 = "数据信息来源"
_data_1 = np.loadtxt(_data_1, delimiter=",", dtype=int)
_data_2 = np.loadtxt(_data_2, delimiter=",", dtype=int)
# 取评论的数据（最后一列）
_t_comments_1 = _data_1[:, -1]
# 只选择大多数集中的数据（这里只选择<5000的数据）
_t_comments_1 = _t_comments_1[_t_comments_1 <= 5000]
# 绘制直方图
# 打印一下最大值和最小值，方便寻找组距
print(_t_comments_1.max(), _t_comments_1.min())
# 设置组距
d = 500
# 设置组数
bin_nums = (_t_comments_1.max() - _t_comments_1.min()) // d
plt.figure(figsize=(20, 8), dpi=80)
plt.hist(_t_comments_1, bin_nums)
plt.show()

# 了解视频中评论数和喜欢数之间的关系，并绘制图像，可以先绘制散点图查看
# 获取喜欢数<=500000的所有数据
_data_1 = _data_1[_data_1[:, 1] <= 500000]
# 列出评论数、喜欢数
_t_comments_1 = _data_1[:, -1]
_t_like_1 = _data_1[:, 1]

plt.figure(figsize=(20, 8), dpi=80)
plt.scatter(_t_like_1, _t_comments_1)
plt.show()

