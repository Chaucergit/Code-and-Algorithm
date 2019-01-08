# -*- coding: utf-8 -*-
# @Time    : 2019-1-8 19:30
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Test_main.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np


def load_weights(filename):
    weights_pre = open(filename)
    w = []
    for line in weights_pre:
        lines = line.strip().split('\t')
        w_temp = []
        for x in lines:
            w_temp.append(float(x))
        w.append(w_temp)
    weights_pre.close()
    return np.mat(w)


def load_data(filename, n):
    data = open(filename)
    data_test = []
    for line in data.readlines():
        feature_tmp = []
        lines = line.strip().split('\t')
        if len(lines) != n-1:
            continue
        feature_tmp.append(1)
        for x in lines:
            feature_tmp.append(float(x))
        data_test.append(feature_tmp)
    data.close()
    return np.mat(data_test)


# 定义 Sigmoid 函数
def Sigmoid(matrix):
    return 1.0 / (1+np.exp(-matrix))


def predict(data, w):
    h = Sigmoid(data * w.T)     # 获取预测值
    m = np.shape(h)[0]
    for i in range(m):
        if h[i, 0] < 0.5:
            h[i, 0] = 0.0
        else:
            h[i, 0] = 1.0
    return h


def save_test(filename, result):
    m = np.shape(result)[0]
    tmp = []
    for i in range(m):
        tmp.append(str(result[i, 0]))
    f_result = open(filename, 'w')
    f_result.write('\t'.join(tmp))     # tmp list 中每个元素用空格隔开，在写入文件时。
    f_result.close()


def main():
    # 1、导入模型
    print('***************** 导入模型 *****************')
    w = load_weights('weights')
    n = np.shape(w)[1]
    # 2、导入测试数据
    print('***************** 导入数据 *****************')
    data_test = load_data('./test_data.txt', n)
    # 3、对测试数据进行预测
    print('***************** 测试数据 *****************')
    h = predict(data_test, w)
    # 4、保存最终预测结果
    print('***************** 保存模型 *****************')
    save_test('test_model', h)


if __name__ == '__main__':
    main()
    dataset = load_data('test_data.txt', 3)
    print(dataset.shape)
