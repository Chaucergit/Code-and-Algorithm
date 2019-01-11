# -*- coding: utf-8 -*-
# @Time    : 2019-1-11 13:59
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Train.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np
import time


# 导入准备训练的数据集
def load_data(filename):
    data = open(filename)
    feature = []
    label = []
    for line in data.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        for x in range(len(lines)-1):
            feature_tmp.append(float(lines[x]))
        label.append(int(lines[-1])*2-1)
        feature.append(feature_tmp)
    data.close()
    return feature, label


# 初始化权重 w 和交叉项权重 v
def initialize_w_v(n, k):
    w = np.ones((n, 1))
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = np.random.normal(0, 0.2)    # 把 v 中的值变为服从 N(0, 0.2) 正态分布的数值
    return w, v


# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 定义误差损失函数 loss(y', y) = ∑-ln[sigmoid(y'* y)]
def get_cost(predict, classLabels):
    m = np.shape(predict)[0]
    # m = len(predict)
    cost = []
    error = 0.0
    for i in range(m):
        error -= np.log(sigmoid(predict[i] * classLabels[i]))
        cost.append(error)
    return error


# 用梯度下降法求解模型参数，训练 FM 模型。
def stocGradient(dataMatrix, classLabels, k, max_iter, alpha):
    """
    :param dataMatrix: 输入的数据集特征
    :param classLabels: 特征对应的标签
    :param k: 交叉项矩阵的维度
    :param max_iter: 最大迭代次数
    :param alpha: 学习率
    :return: 
    """
    m, n = np.shape(dataMatrix)
    w0 = 0
    w, v = initialize_w_v(n, k)     # 初始化参数
    for it in range(max_iter):
        # print('第 %d 次迭代' % it)
        for x in range(m):
            v_1 = dataMatrix[x] * v    # dataMatrix[x]的 shape 为(1,n),v的 shape 为(n,k)--->v_1的 shape为(1, k)
            v_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
            interaction = 0.5 * np.sum(np.multiply(v_1, v_1) - v_2)
            p = w0 + dataMatrix[x] * w + interaction
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            w0 = w0 - alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] * alpha * loss * classLabels[x] * (dataMatrix[x, i] * v_1[0, j] - v[i, j] *
                                                                             dataMatrix[x, i]*dataMatrix[x, i])
        if it % 1000 == 0:
            print("\t迭代次数:" + str(it) + ",误差:" + str(get_cost(prediction(np.mat(dataMatrix), w0, w, v), classLabels)))
    return w0, w, v


# 定义预测结果的函数
def prediction(dataMatrix, w0, w, v):
    m = np.shape(dataMatrix)[0]
    result = []
    for x in range(m):
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
        interaction = 0.5 * np.sum(np.multiply(inter_1, inter_1) - inter_2)
        p = w0 + dataMatrix[x] * w + interaction
        pre = sigmoid(p[0, 0])
        result.append(pre)
    return result


# 计算准确度
def getaccuracy(predict, classLabels):
    m = np.shape(predict)[0]
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error)/allItem


# 保存模型的参数
def save_model(filename, w0, w, v):
    f = open(filename, 'w')
    f.write(str(w0)+'\n')
    w_array = []
    m = np.shape(w)[0]
    for i in range(m):
        w_array.append(str(w[i, 0]))
    f.write('\t'.join(w_array)+'\n')
    m1, n1 = np.shape(v)
    for i in range(m1):
        v_tmp = []
        for j in range(n1):
            v_tmp.append(str(v[i, j]))
        f.write('\t'.join(v_tmp)+'\n')
    f.close()


# 主函数
def main():
    # 第一步：导入数据
    feature, label = load_data('train_data.txt')
    # print(feature, label)
    # 第二步：利用梯度下降训练模型
    w0, w, v = stocGradient(np.mat(feature), label, 4, 20001, 0.02)
    predict_result = prediction(np.mat(feature), w0, w, v)
    print('训练精度为：%f' % (1-getaccuracy(predict_result, label)))
    # 第三步保存模型
    save_model('weights_FM', w0, w, v)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('训练模型用时为：%s' % str(end-start))
