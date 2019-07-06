#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/7/3 14:48
# @Author  : GXL
# @Site    : 
# @File    : pandas_series.py
# @Software: PyCharm

# pandas常用数据类型，series一维数组，DataFrame二维数组(series容器)
# pandas读取外部文件
import pandas as pd

df = pd.read_csv("文件目录")
# pandas读取mysql数据，pd.read_mysql(sql_sentence, connection)


# pandas读取mongodb数据
from pymongo import MongoClient

client = MongoClient()
collection = client["douban"]["tv1"]
data = list(collection.find())
print(data)

# Series
t1 = data[0]
t1 = pd.Series(t1)
print(t1)

# 筛选数据
data_list = []
for i in data:
    temp = {"info": i["info"], "rating_count": i["rating"]["count"]}
    data_list.append(temp)

df = pd.DataFrame(data_list)
# 显示头1行,
print(df.head(1))
print(df.tail(1))
# 展示df的概览
print(df.info())
# 帮助快速进行统计
print(df.describe())
# 数据切割,转换为列表
print(df["info"].str.split("/").tolist())
