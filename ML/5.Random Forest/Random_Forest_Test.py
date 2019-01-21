# -*- coding: utf-8 -*-
# @Time    : 2019-1-20 20:16
# @Author  : Chaucer_Gxm
# @Email   : gxm4167235@163.com
# @File    : Random_Forest_Test.py
# @GitHub  : https://github.com/Chaucergit/Code-and-Algorithm
# @blog    : https://blog.csdn.net/qq_24819773
# @Software: PyCharm
import _pickle as pickle
from random_forests_train import get_predict


def load_data(file_name):
    f = open(file_name)
    test_data = []
    for line in f.readlines():
        test_data_tmp = []
        lines = line.strip().split('\t')
        for x in lines:
            test_data_tmp.append(float(x))
        test_data_tmp.append(0)
        test_data.append(test_data_tmp)
    f.close()
    return test_data


def load_model(result_file, feature_file):
    trees_fiture = []
    f_tea = open(feature_file)
    for line in f_tea.readlines():
        tmp = []
        lines = line.strip().split('\t')
        for x in lines:
            tmp.append(int(x))
        trees_fiture.append(tmp)
    f_tea.close()

    with open(result_file, 'rb') as f:
        tree_result = pickle.load(f)
    return tree_result, trees_fiture


def save_result(data_test, prediction, result_file):
    '''保存最终的预测结果
    input:  data_test(list):待预测的数据
            prediction(list):预测的结果
            result_file(string):存储最终预测结果的文件名
    '''
    m = len(prediction)
    n = len(data_test[0])

    f_result = open(result_file, "w")
    for i in range(m):
        tmp = []
        for j in range(n - 1):
            tmp.append(str(data_test[i][j]))
        tmp.append(str(prediction[i]))
        f_result.writelines("\t".join(tmp) + "\n")
    f_result.close()

def main():
    data_test = load_data('test_data.txt')
    trees_result, trees_feature = load_model('result_file', 'feature_file')
    prediction = get_predict(trees_result, trees_feature, data_test)
    print(prediction)
    save_result(data_test, prediction, "final_result")


if __name__ == '__main__':
    main()
