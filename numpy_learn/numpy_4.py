#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/2 21:29
# @Author  : GXL
# @Site    : 
# @File    : numpy_4.py
# @Software: PyCharm

import numpy as np

# seed随机数种子，下面生成的随机数不变化
np.random.seed(10)
t1 = np.random.randint(0, 10, (3, 4))
print(t1)

# copy 和 view
# 视图操作互相影响
a = t1
# 视图的切片，互相影响
b = t1[:]
# copy操作互不影响
c = t1.copy()

# nan = not a number，两个nan不相等，
# count_nonzero计算非零个数
np.count_nonzero(t1)
# 判断数组中nan的个数
np.count_nonzero(t1 != t1)
# 判断一个数字是否为nan，将nan替换为0
t1[np.isnan(t1)] = 0

# sum求和
np.sum(t1)
# 每列求和
np.sum(t1, axis=0)
# 每行求和
np.sum(t1, axis=1)
t1.sum()
t1.sum(axis=0)

# 均值mean
t1.mean(axis=0)
np.mean(t1, axis=0)

# 中值
np.median(t1, axis=0)

# 最大(小)值
t1.max()
t1.min()
np.max(t1)
np.min(t1)

# 极值
t1.ptp(axis=0)
np.ptp(t1)

# 标准差
t1.std(axis=0)
np.std(t1)


# ************************************************************
# 将数组中的nan替换为均值
def fill_nan_array(t):
    for i in range(t.shape[1]):
        # 当前的一列
        temp_col = t[:, i]
        nan_num = np.count_nonzero(temp_col != temp_col)
        if nan_num != 0:
            # 获取当前不为nan的数组
            temp_not_nan_col = temp_col[temp_col == temp_col]
            # 获取当前列的均值
            temp_not_nan_col.mean()
            # 将均值放入nan的地方
            temp_col[np.isnan(temp_col)] = temp_not_nan_col.mean()
    return t


# 调用上面方法
if __name__ == '__main__':
    t2 = np.arange(14).reshape((3, 4)).astype("float")
    t2[1, 2:] = np.nan
    print(t2)
    t2 = fill_nan_array(t2)
    print(t2)














