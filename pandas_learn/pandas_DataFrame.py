#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/6 15:13
# @Author  : GXL
# @Site    : 
# @File    : pandas_DataFrame.py
# @Software: PyCharm

import pandas as pd
import numpy as np

df = pd.read_csv("local")

# DataFrame中排序方法，降序
df.sort_values(by="name", ascending=False)
print(df.tail())
# 前20行，name列
print(df[:20]["name"])
# 一列是Series类型，两列是DateFrame类型
# df.loc 通过标签索引行数据；df.iloc通过位置索引获取行数据
t3 = pd.DataFrame(np.arange(12).reshape(3, 4), index=list("abc"), columns=list("WXYZ"))