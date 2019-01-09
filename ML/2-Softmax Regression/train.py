# -*- coding: utf-8 -*-
# @Time    : 2019-1-9 13:47
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Softmax_Regression_Train.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np


# 梯度更新函数 GradientAscent
def GradientAscent(feature_data, label_data, k, maxCycle, alpha):
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        error = np.exp(feature_data * weights)
        # print('**********************************************')
        # print(error, '*****2828****')
        # print('**********************************************')

        if i % 1000 == 0:
            print("\t 迭代次数 Iter 为: " + str(i) + ", 误差Cost为: " + str(Cost(error, label_data)))
        rowsum = -error.sum(axis=1)    # axis=1 表示将一个矩阵的每一行向量相加
        '''
         (1)numpy.repeat(a,repeats,axis=None); 
          (2)object(ndarray).repeat(repeats,axis=None):理解了第一种方法就可以轻松知道第二种方法了。
          参数的意义：axis=None，时候就会flatten当前矩阵，实际上就是变成了一个行向量
               axis=0,沿着y轴复制，实际上增加了行数
               axis=1,沿着x轴复制，实际上增加列数
        '''
        rowsum = rowsum.repeat(k, axis=1)
        # print('************************* rowsum ===>'+'\n', rowsum)
        error = error / rowsum     # 得到整体事件中每个事件发生的概率
        print('**********************************************')
        print(error, '*****2828****')
        print('**********************************************')
        for x in range(m):
            error[x, label_data[x, 0]] += 1     # 更具有普适性
            # print('**********************************************')
            # print(error, '*****2828****')
            # print('**********************************************')
            # error[x, label_data[x]] += 1 # label 为一维数据时这样也可以
        weights = weights + (alpha / m) * feature_data.T * error
        i += 1
    return weights


# 误差函数
def Cost(error, label_data):
    m = np.shape(error)[0]
    sum_cost = 0.0
    for i in range(m):
        if error[i, label_data[i, 0]] / np.sum(error[i, :]) > 0:     # 如果事件发生的概率大于 0。I{ y = j } = 1
            sum_cost -= np.log(error[i, label_data[i, 0]] / np.sum(error[i, :]))
        else:     # I{ y = j } = 0
            sum_cost -= 0
    return sum_cost / m


def load_data(filename):
    data = open(filename)
    feature_data = []
    label_data = []
    for line in data.readlines():
        feature_tmp = []
        feature_tmp.append(1)
        lines = line.strip().split('\t')
        a = len(lines)-1
        for x in range(a):
            feature_tmp.append(float(lines[a]))
        label_data.append(int(lines[-1]))
        feature_data.append(feature_tmp)
    data.close()
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))


def save_model(weights_name, weights):
    f_w = open(weights_name, 'w')
    m, n = np.shape(weights)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(weights[i, j]))
        f_w.write('\t'.join(w_tmp)+'\n')
    f_w.close()
    return print('写入数据成功')


def main():
    # 1.导入训练数据
    feature, label, k = load_data('./train_data.txt')
    print(feature.shape)
    # 2.训练模型
    weights = GradientAscent(feature, label, k, 1, 0.9)
    # 3.保存模型
    save_model('weights_Softmax_Regression', weights)

if __name__ == '__main__':
    main()

