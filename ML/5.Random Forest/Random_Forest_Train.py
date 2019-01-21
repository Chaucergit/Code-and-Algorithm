# -*- coding: utf-8 -*-
# @Time    : 2019-1-20 20:16
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Random_Forest_Train.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np
from math import log
import _pickle as pickle
from Tree import build_tree, predict, cal_gini_index, label_uniq_cnt


# 随机选择样本及特征---》从样本中随机选择样本及其特征
def choose_samples(data, k):
    m, n = np.shape(data)
    # 1.选择出 k 个特征的 index
    feature = []
    for j in range(k):
        feature.append(np.random.randint(0, n - 2))
    # 2.选择出 m 个样本的 index
    index = []
    for i in range(m):
        index.append(np.random.randint(0, m - 1))
    # 3.从 data 中选择出 m 个样本的 k 个特征，组成数据集 data_samples
    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in feature:
            data_tmp.append(data[i][fea])
        data_tmp.append(data[i][-1])
        data_samples.append(data_tmp)
    return data_samples, feature


def random_forest_training(data_train, trees_num):
    trees_result = []
    trees_feature = []
    n = np.shape(data_train)[1]
    if n > 2:
        k = int(log(n-1, 2)) + 1
    else:
        k = 1
    for i in range(trees_num):
        # 1.随机选择 m 个样本，k 个特征
        data_samples, feature = choose_samples(data_train, k)
        # 2.构建每一颗分类树
        tree = build_tree(data_samples)
        # 3.保存训练好的分类树模型
        trees_result.append(tree)
        # 4.保存好该分类树使用到的特征
        trees_feature.append(feature)
    return trees_result, trees_feature


# 导入数据集
def load_data(file_name):
    data_train = []
    f = open(file_name)
    for line in f.readlines():
        lines = line.strip().split('\t')
        data_tmp = []
        for x in lines:
            data_tmp.append(float(x))
        data_train.append(data_tmp)
    f.close()
    return data_train


def get_predict(trees_result, trees_feature, data_train):
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]
    result = []
    for i in range(m_tree):
        clf = trees_result[i]
        feature = trees_feature[i]
        data = split_data(data_train, feature)
        result_i = []
        for i in range(m):
            # result_i.append(list((predict(data[i][0:, -1], clf).keys()))[0])
            result_i.append(list((predict(data[i][0:-1], clf).keys()))[0])
        result.append(result_i)
    final_predict = np.sum(result, axis=0)
    return final_predict


# 计算准确度
def cal_correct_rate(data_train, final_predict):
    m = len(final_predict)
    corr = 0.0
    for i in range(m):
        if data_train[i][-1]*final_predict[i] > 0:
            corr += 1
    return corr/m


def split_data(data_train, feature):
    m = np.shape(data_train)[0]
    data = []
    for i in range(m):
        data_x_tmp = []
        for x in feature:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[i][-1])
        data.append(data_x_tmp)
    return data


def save_model(trees_result, trees_feature, result_file, feature_file):
    m = len(trees_feature)
    f_fea = open(feature_file, 'w')
    for i in range(m):
        fea_tmp = []
        for x in trees_feature[i]:
            fea_tmp.append(str(x))
        f_fea.writelines('\t'.join(fea_tmp)+'\n')
    f_fea.close()
    with open(result_file, 'wb') as f:
        pickle.dump(trees_result, f)


def main():
    # 1.导入数据集
    data_train = load_data('data.txt')
    print('********** Start **********')
    print(np.shape(data_train))
    # 2.训练模型
    trees_result, trees_feature = random_forest_training(data_train, 50)
    print('********** Train **********')
    # 3.得到训练准确性
    result = get_predict(trees_result, trees_feature, data_train)
    corr_rate = cal_correct_rate(data_train, result)
    print('正确率为：', corr_rate)
    # 4.保存最终的随机森林模型
    save_model(trees_result, trees_feature, 'result_file', 'feature_file')


if __name__ == '__main__':
    main()
