#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 10:09
# @Author  : GXL
# @Site    : 
# @File    : numpy_3.py
# @Software: PyCharm

# numpy中的nan和常用方法
import numpy as np

t1 = np.array(range(12))
t2 = np.array(range(12, 24))

# 竖直拼接
t3 = np.vstack((t1, t2))

# 水平拼接
t4 = np.hstack((t1, t2))

t5 = t4.reshape(4, 6)
# 行交换(1, 2)交换
t5[[1, 2], :] = t5[[2, 1], :]
print("行交换..............\n", t5)
# 列交换(0, 2)交换
t5[:, [0, 2]] = t5[:, [2, 0]]
print("列交换..............\n", t5)

# ***********************************************************
# 案例：希望把两个国家的数据放在一起研究，同时保留国家信息
# ***********************************************************
# 加载国家数据
_data_1 = "数据信息来源"
_data_2 = "数据信息来源"

_data_1 = np.loadtxt(_data_1, delimiter=",", dtype=int)
_data_2 = np.loadtxt(_data_2, delimiter=",", dtype=int)


# 添加国家信息
#   构造全部为0的数据，使用dtype控制类型
_zero_data = np.zeros((_data_1.shape[0], 1), dtype=int)
#   构造全部为1的数据，使用dtype控制类型
_one_data = np.ones((_data_2.shape[0], 1), dtype=int)
#   分别添加一列全部为0或1的数据
_data_1 = np.hstack((_data_1, _zero_data))
_data_2 = np.hstack((_data_2, _one_data))

# 拼接两组信息
_data_all = np.vstack((_data_1, _data_1))

# ************************************************************
# 对角线
np.eye(5)
