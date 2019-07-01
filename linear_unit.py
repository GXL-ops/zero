#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Project   : zero
# File      : linear_unit.py
# Author    : GXL
# Date      : 2019/6/26

# 线性单元和梯度下降
# 感知器有一个问题，当面对的数据集不是线性可分的时候，『感知器规则』可能无法收敛，这意味着我们永远也无法完
# 成一个感知器的训练。为了解决这个问题，我们使用一个可导的线性函数来替代感知器的阶跃函数，这种感知器就叫做
# 线性单元。线性单元在面对线性不可分的数据集时，会收敛到一个最佳的近似上。

# 随机梯度下降算法(Stochastic Gradient Descent, SGD)
# 每次更新w的迭代，要遍历训练数据中所有的样本进行计算，我们称这种算法叫做批梯度下降(Batch Gradient Descent)
# SGD不仅仅效率高，而且随机性有时候反而是好事。

from perceptron import Perceptron


# 定义激活函数f
f = lambda x: x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        """初始化线性单元，设置输入参数的个数"""
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    """
    捏造5个人的收入数据
    """
    # 构建训练数据
    # 输入向量列表，每一项是工作年限
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # 期望的输出列表，月薪，注意要与输入一一对应
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    """
    使用数据训练线性单元
    """
    # 创建感知器，输入参数的特征数为1（工作年限）
    lu = LinearUnit(1)
    # 训练，迭代10轮, 学习速率为0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # 返回训练好的线性单元【权重集合】
    return lu


def plot(linear_unit):
    """
    绘制图像
    :param linear_unit: 训练好的线性单元【权重集合】
    :return:
    """
    import matplotlib.pyplot as plt
    input_vecs, labels = get_training_dataset()
    map1 = list(map(lambda x: x[0], input_vecs))
    plt.scatter(map1, labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = list(range(0, 12, 1))
    y = list(map(lambda x: weights[0] * x + bias, x))
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    '''训练线性单元'''
    linear_unit = train_linear_unit()
    # 打印训练获得的权重
    print(linear_unit)
    # 测试
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)
