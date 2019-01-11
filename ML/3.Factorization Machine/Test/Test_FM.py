# -*- coding: utf-8 -*-
# @Time    : 2019-1-11 17:16
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Test_FM.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm

import numpy as np
# from FM_train import getPrediction


# 定义 Sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def loadDataSet(data):
    '''导入测试数据集
    input:  data(string)测试数据
    output: dataMat(list)特征
    '''
    dataMat = []
    fr = open(data)  # 打开文件
    for line in fr.readlines():
        lines = line.strip().split("\t")
        lineArr = []
        for i in range(len(lines)):
            lineArr.append(float(lines[i]))
        dataMat.append(lineArr)
    fr.close()
    return dataMat


def loadModel(model_file):
    '''导入FM模型
    input:  model_file(string)FM模型
    output: w0, np.mat(w).T, np.mat(v)FM模型的参数
    '''
    f = open(model_file)
    line_index = 0
    w0 = 0.0
    w = []
    v = []
    for line in f.readlines():
        lines = line.strip().split("\t")
        if line_index == 0:  # w0
            w0 = float(lines[0].strip())
        elif line_index == 1:  # w
            for x in lines:
                w.append(float(x.strip()))
        else:
            v_tmp = []
            for x in lines:
                v_tmp.append(float(x.strip()))
            v.append(v_tmp)
        line_index += 1
    f.close()
    return w0, np.mat(w).T, np.mat(v)


def save_result(file_name, result):
    '''保存最终的预测结果
    input:  file_name(string)需要保存的文件名
            result(mat):对测试数据的预测结果
    '''
    f = open(file_name, "w")
    f.write("\n".join(str(x) for x in result))
    f.close()


def main():
    # 1、导入测试数据
    print("---------- 1.load data ------------")
    dataTest = loadDataSet("test_data.txt")
    # 2.Softmax Regression Train、导入FM模型
    print("---------- 2.Softmax Regression Train.load model ------------")
    w0, w, v = loadModel("../Train/weights_FM")
    # 3、预测
    print("---------- 3.get Prediction ------------")
    result = prediction(dataTest, w0, w, v)
    # 4、保存最终的预测结果
    print("---------- 4.save prediction ------------")
    save_result("predict_result", result)


if __name__ == '__main__':
    main()
