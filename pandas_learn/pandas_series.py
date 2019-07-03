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
import pandas as pd

client = MongoClient()
collection = client["douban"]["tv1"]
data = list(collection.find())
print(data)
