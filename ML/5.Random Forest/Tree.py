# -*- coding: utf-8 -*-
# @Time    : 2019-1-20 20:58
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Tree.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np
# from Random_Forest_Train import cal_gini_index, label_uniq_cnt


class node:
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea          # 用于切分数据集的属性的列索引值
        self.value = value      # 设置划分的值
        self.results = results  # 存储叶节点所属的类别
        self.right = right      # 右子树
        self.left = left        # 左子树


def split_tree(data, fea, value):
    set_1 = []
    set_2 = []
    # 划分左右子树
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    return set_1, set_2


# 计算数据集中不同标签的个数
def label_uniq_cnt(data):
    label_uniq_cnt = {}
    for x in data:
        label = x[len(x) - 1]
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        label_uniq_cnt[label] = label_uniq_cnt[label] + 1
    return label_uniq_cnt


# 计算基尼系数
def cal_gini_index(data):
    total_sample = len(data)    # 得到样本个数
    if total_sample == 0:
        return 0
    label_counts = label_uniq_cnt(data)     # 统计数据集中不同标签的个数
    gini = 0
    for label in label_counts:
        gini += pow(label_counts[label], 2)
    gini = 1 - float(gini) / pow(total_sample, 2)
    return gini



def build_tree(data):
    # 构建决策树，函数返回该决策树的根节点
    if len(data) == 0:
        return node()
    currentGini = cal_gini_index(data)
    bestCriteria = None
    bestGain = 0.0
    bestSets = None

    feature_num = len(data[0]) - 1
    for fea in range(0, feature_num):
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1
        for value in feature_values.keys():
            (set_1, set_2) = split_tree(data, fea, value)
            nowGini = float(len(set_1)*cal_gini_index(set_1)+len(set_2)*cal_gini_index(set_2))/len(data)
            gain = currentGini - nowGini
            if gain > bestGain and len(set_1) > 0 and len(set_2) > 0:
                bestGain = gain
                bestCriteria = (fea, value)
                bestSets = (set_1, set_2)
    if bestGain > 0:
        right = build_tree(bestSets[0])
        left = build_tree(bestSets[1])
        return node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
    else:
        return node(results=label_uniq_cnt(data))


# 利用训练好的分类树对新样本进行预测
def predict(sample, tree):
    if tree.results != None:
        return tree.results
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)
