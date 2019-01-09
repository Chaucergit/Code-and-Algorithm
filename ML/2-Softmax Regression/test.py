# -*- coding: utf-8 -*-
# @Time    : 2019-1-9 14:42
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Test.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np


def load_data(num, m):
    testDataSet = np.mat(np.ones((num, m)))
    for i in range(num):
        testDataSet[i, 1] = np.random.random() * 6 - 3
        testDataSet[i, 2] = np.random.random() * 15
    return testDataSet


def load_weights(weights_path):
    f = open(weights_path)
    w = []
    for line in f.readlines():
        w_tmp = []
        lines = line.strip().split('\t')
        for x in range(len(lines) - 1):
            w_tmp.append(float(lines[x]))
        w.append(w_tmp)
    f.close()
    weights = np.mat(w)
    m, n = np.shape(weights)
    return weights, m, n


def predict(test_data, w):
    h = test_data * w
    return h.argmax(axis=1)


def save_result(filename, result):
    f_reslt = open(filename, 'w')
    m = np.shape(result)[0]
    for i in range(m):
        f_reslt.write(str(result[i, 0])+'\n')
    f_reslt.close()
    return print('写入数据成功')


def main():
    w, m, n = load_weights('../Softmax Regression Train/weights_Softmax_Regression')
    test_data = load_data(4000, m)
    result = predict(test_data, w)
    save_result('test_result', result)


if __name__ == '__main__':
    main()

