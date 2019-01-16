# -*- coding: utf-8 -*-
# @Time    : 2019-1-15 18:58
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : SVM_Train.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import numpy as np
import _pickle as pickle


class SVM:
    def __init__(self, dataSet, labels, C, toler, kernel_option):
        self.train_x = dataSet
        self.train_y = labels
        self.C = C
        self.toler = toler
        self.n_samples = np.shape(dataSet)[0]
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))
        self.b = 0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))
        self.kernel_opt = kernel_option
        self.kernel_mat = calc_kernel(self.train_x, self.kernel_opt)


# 核函数矩阵
def calc_kernel(train_x, kernel_option):
    m = np.shape(train_x)[0]
    kernel_matrix = np.mat(np.zeros((m, m)))
    for i in range(m):
        kernel_matrix[:, i] = cal_kernel_value(train_x, train_x[i, :], kernel_option)
    return kernel_matrix


# 定义样本之间的核函数的值
def cal_kernel_value(train_x, train_x_i, kernel_option):
    kernel_type = kernel_option[0]
    m = np.shape(train_x)[0]
    kernel_value = np.mat(np.zeros((m, 1)))
    if kernel_type == 'rbf':
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(m):
            diff = train_x[i, :] - train_x_i
            kernel_value[i] = np.exp(diff * diff.T/(-2.0 * sigma**2))
    else:
        kernel_value = train_x * train_x_i.T
    return kernel_value


def SVM_training(train_x, train_y, C, toler, max_iter, kernel_option=('rbf', 0.431029)):
    # 1.初始化 SVM 分类器
    svm = SVM(train_x, train_y, C, toler, kernel_option)
    # 2.开始训练 SVM 分类器
    entireSet = True
    alpha_pairs_changed = 0
    iteration = 0
    while (iteration < max_iter) and((alpha_pairs_changed > 0) or entireSet):
        print('\t 迭代次数为：', iteration)
        alpha_pairs_changed = 0
        if entireSet:
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:
            bound_samples = []
            for i in range(svm.n_samples):
                if 0 < svm.alphas[i, 0] < svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        if entireSet:
            entireSet = False
        elif alpha_pairs_changed == 0:
            entireSet = True
    return svm


# 选择并更新参数
def choose_and_update(svm, alpha_i):
    # 计算第一个样本的误差error_i
    error_i = cal_error(svm, alpha_i)

    # 判断选择出的第一个变量是否违反了 KKT 条件
    if (svm.train_y[alpha_i]*error_i < -svm.toler) and (svm.alphas[alpha_i] < svm.C) or (svm.train_y[alpha_i]*error_i >
        svm.toler) and (svm.alphas[alpha_i] > 0):
        # 1.选择第二个变量
        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        # 2.计算上下界
        if svm.train_y[alpha_i] != svm.train_y[alpha_j]:
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = min(svm.C, svm.C + svm.alphas[alpha_j] - svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_j] + svm.alphas[alpha_i] - svm.C)
            H = min(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        if L == H:
            return 0

        # 3.计算 eta
        eta = 2.0 * svm.kernel_mat[alpha_i, alpha_j] - svm.kernel_mat[alpha_i, alpha_i] - svm.kernel_mat[alpha_j, alpha_j]
        if eta > 0:
            return 0

        # 4.更新 alpha_j
        svm.alphas[alpha_j] -= svm.train_y[alpha_j] * (error_i - error_j)/eta

        # 5.确定最终的 alpha_j
        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        # 6. 判断是否结束
        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error_tmp(svm, alpha_j)
            return 0

        # 7.更新 alpha_i
        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])

        # 8.更新 b
        b1 = svm.b - error_i - svm.train_y[alpha_i]*(svm.alphas[alpha_i]-alpha_i_old)*svm.kernel_mat[alpha_i, alpha_i]\
             -svm.train_y[alpha_j]*(svm.alphas[alpha_j]-alpha_j_old)*svm.kernel_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i]*(svm.alphas[alpha_i]-alpha_i_old)*svm.kernel_mat[alpha_i, alpha_j]\
             -svm.train_y[alpha_j]*(svm.alphas[alpha_j]-alpha_j_old)*svm.kernel_mat[alpha_j, alpha_j]
        if 0 < svm.alphas[alpha_i] < svm.C:
            svm.b = b1
        elif 0 < svm.alphas[alpha_j] < svm.C:
            svm.b = b2
        else:
            svm.b = (b1 + b2)/2.0

        # 9.更新 error
        update_error_tmp(svm, alpha_j)
        update_error_tmp(svm, alpha_i)
        return 1
    else:
        return 0


# 计算误差
def cal_error(svm, alpha_k):
    output_k = float(np.multiply(svm.alphas, svm.train_y).T * svm.kernel_mat[:, alpha_k]+svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k


# 选择第二个变量
def select_second_sample_j(svm, alpha_i, error_i):
    svm.error_tmp[alpha_i] = [1, error_i]
    candidateAlphaList = np.nonzero(svm.error_tmp[:, 0].A)[0]
    maxStep = 0
    alpha_j = 0
    error_j = 0

    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = cal_error(svm, alpha_k)
            if abs(error_k-error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k
    else:
        alpha_j = alpha_i
        while alpha_j == alpha_i:
            alpha_j = int(np.random.uniform(0, svm.n_samples))
        error_j = cal_error(svm, alpha_j)
    return alpha_j, error_j


# 重新计算误差值
def update_error_tmp(svm, alpha_k):
    error = cal_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]


def load_data_libsvm(filename):
    df = open(filename)
    features = []
    labels = []
    for line in df.readlines():
        lines = line.strip().split(' ')    # lines = ['+1', '1:0.708333', '2:1', '3:1', '4:-0.320755', '5:-0.105023', '6:-1', '7:1', '8:-0.419847', '9:-1', '10:-0.225806', '12:1', '13:-1']
        # print(lines[1][1])
        labels.append(float(lines[0]))
        index = 0
        tmp = []
        for i in range(1, len(lines)):
            x = lines[i].strip().split(':')
            # print(x[1])
            if int(x[0])-1 == index:
                tmp.append(float(x[1]))
            else:
                while int(x[0])-1 > index:
                    tmp.append(0)
                    index += 1
                tmp.append(float(x[1]))
            index += 1
        while len(tmp) > 13:
            tmp.append(0)
        features.append(tmp)
    df.close()
    return np.mat(features), np.mat(labels).T


def cal_accuracy(svm, test_x, test_y):
    n_samples = np.shape(test_x)[0]
    correct = 0.0
    for i in range(n_samples):
        predict = svm_predict(svm, test_x[i, :])
        if np.sign(predict) == np.sign(test_y[i]):
            correct += 1
    accuracy = correct / n_samples
    return accuracy


def svm_predict(svm, test_sample_x):
    kernel_value = cal_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
    predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
    return predict


# 保存模型
def save_model(svm_model, model_file):
    with open(model_file, 'wb') as file:
        pickle.dump(svm_model, file)


def main():
    # 1.导入数据集
    print('********* 导入数据集 **********')
    train_data, label_data = load_data_libsvm('heart_scale')
    print(train_data.shape, label_data.shape)
    # 2.训练 SVM 模型
    print('********* 训练 SVM 模型 **********')
    C = 0.6
    toler = 0.001
    maxIter = 500
    svm_model = SVM_training(train_data, label_data, C, toler, maxIter)
    # 3.计算模型的准确性
    print('********* 计算模型的准确性 **********')
    accuracy = cal_accuracy(svm_model, train_data, label_data)
    print('训练精度为：%.3f%%' % (accuracy*100))
    # 4.保存最终的模型
    print('********* 保存最终的模型 **********')
    save_model(svm_model, "model_file")

if __name__ == '__main__':
    main()
