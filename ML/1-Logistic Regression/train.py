# -*- coding: utf-8 -*-
# @Time    : 2019-1-8 18:27
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Logsitic_Regression.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np


# 定义 Sigmoid 函数
def Sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 定义损失函数
def error_rate(h, label):
    '''计算当前的损失函数值
    input:  h(mat):预测值
            label(mat):实际值
    output: err/m(float):错误率
    '''
    m = np.shape(h)[0]

    sum_err = 0.0
    for i in range(m):
        if h[i, 0] > 0 and (1 - h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1 - label[i, 0]) * np.log(1 - h[i, 0]))
        else:
            sum_err -= 0
    return sum_err / m


# 定义 BGD 梯度下降法
def lr_train_bgd(feature, label, maxCycle, alpha):
    n = np.shape(feature)[1]    # 特征的个数
    w = np.mat(np.ones((n, 1)))   # 初始化权重
    i = 0
    while i <= maxCycle:
        i += 1
        h = Sigmoid(feature * w)
        error = label - h
        if i % 100 == 0:
            print('\t 迭代次数为=：' + str(i) + ',训练误差率为：' + str(error_rate(h, label)))
        w = w + alpha * feature.T * error
    return w


def load_data(file_name):
    '''导入训练数据
    input:  file_name(string)训练数据的位置
    output: feature_data(mat)特征
            label_data(mat)标签
    '''
    f = open(file_name)  # 打开文件
    feature_data = []
    label_data = []
    for line in f.readlines():
        feature_tmp = []
        lable_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # 偏置项
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        lable_tmp.append(float(lines[-1]))

        feature_data.append(feature_tmp)
        label_data.append(lable_tmp)
    f.close()  # 关闭文件
    return np.mat(feature_data), np.mat(label_data)


# 保持最终的模型
def save_model(filename, w):
    m = np.shape(w)[0]
    f_w = open(filename, 'w')
    w_array = []
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f_w.write('\t'.join(w_array))
    f_w.close()


def main():
    print('***************** 导入模型 *****************')
    features, labels = load_data('data.txt')
    print('***************** 训练模型 *****************')
    w = lr_train_bgd(features, labels, 1000, 0.01)
    print('***************** 保存模型 *****************')
    save_model('weights', w)


if __name__ == '__main__':
    main()

