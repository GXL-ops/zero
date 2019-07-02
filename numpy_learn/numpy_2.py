#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 10:09
# @Author  : GXL
# @Site    :
# @File    : numpy_3.py
# @Software: PyCharm

# 读取本地数据和索引
import numpy as np

_file_path = "csv文件目录"

# delimiter：按照“，”分割, unpack=True表示转置。查看源码了解所有参数
np.loadtxt(_file_path, delimiter=",", dtype="int", unpack=True)

print("*" * 100)

# 转置
t1 = np.array(range(24)).reshape(4, 6)
t2 = t1.transpose()
t3 = t1.T
t4 = t1.swapaxes(1, 0)  # ------------交换轴

# 取行t1[0]表示第一行
print(t1[2])

# 取连续多行
print(t1[2:])

# 取不连续多行，注意多层[]
print(t1[[2, 4, 10]])

# 取列t1[:,0]表示第一列
print(t1[1, :])
print(t1[:, 0])

# 取多行多列，第3行到第5行，第2列到第4列的结果
t5 = t1[2:5, 1:4]

# 取(0, 0)和(1, 3)点
t6 = t1[[0, 1], [0, 3]]

# 将第三列和第四列置位0
t7 = t1[:, 2:4] = 0

# 转换为bool数值,当数值小于4的时候设置为true
t8 = t1 < 4

# 当数值小于4的时候设置为1
t9 = t1[t1 < 4] = 1

# 只获取小于4的数值
t10 = t1[t1 < 4]

# 小于4的数值替换为0，大于4的数值替换为10
t11 = np.where(t1 < 4, 0, 10)

# clip裁剪操作，将小于3的替换为3，大于10的替换为10
t1.clip(3, 10)

