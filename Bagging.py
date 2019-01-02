from random import randrange, seed, random
import numpy as np


# 导入数据集ving进行出路形成list，方便下面的函数进行操作计算
def loadData(filename):
    dataset = []
    df = open(filename, 'r')
    for line in df.readlines():
        if not line:
            continue
        lineArr = []
        for feature in line.split(','):
            str_line = feature.strip()
            if str_line.isdigit():
                lineArr.append(float(str_line))
            else:
                lineArr.append(str_line)
        dataset.append(lineArr)
    return dataset


# 分割数据集
def Cross_Validation_split(dataset, n_folds):
    """
    param  --> dataset: 输入的数据集或者说待处理的数据集
    param  --> n_folds: 数据集被分割后的块数
    return --> dataset_split: split后的数据集
    """


    dataset_split = []
    dataset_copy = dataset.copy()
    dataset_num = len(dataset_copy)    # 获取数据集的长度，此处为208
    fold_size = int(dataset_num) / n_folds    # 得到每个split后fold的大小
    for i in range(n_folds):
        fold = []
        while len(fold) < fold_size:
            index = np.random.randint(dataset_num)    # 随机生成一个范围内的随机数,作为寻找数据的索引值
            # fold.append(dataset_copy.pop(index))    # 无放回的方式
            fold.append(dataset_copy[index])    # 有放回的方式
        dataset_split.append(fold)
    return dataset_split


# 根据特征和特征值分割数据集
def test_split(index, value, dataset):
    """
    :param index: 待决策数据（特征）的索引值
    :param value: 特征值---row[index]
    :param dataset: 数据集
    :return:  
    """
    left, right = list(), list()
    for row in dataset:
        if float(row[index]) < float(value):
            # left.append(row[index])
            left.append(row)
        else:
            right.append(row)
            # right.append(row[index])
    return left, right


# 计算 group 的基尼系数，判断算法的优良
def gini_index(groups, class_values):
    """
    
    :param groups: 分组后的数据集
    :param class_values: 
    :return: 
    """
    gini = 0.0
    D = len(groups[0]) + len(groups[1])
    for class_value in class_values:
        for group in groups:
            size = len(group)
            if size == 0:
                continue
            # count()方法用于统计某个元素在列表中出现的次数,row[-1]--->[M,R],
            # class_value为dataset中的[M,R]
            # 故proportion为 class_value=[M,R] 在 group=float(size) 中的占比，即为:p(k)
            # proportion = [row[-1] for row in group].count(class_value) / float(size)
            proportion = [row[-1] for row in group].count(class_value) / float(size)
            gini += float(size)/D * (proportion * (1.0 - proportion))    # 基尼系数：Gini(p) = ∑p(k)*[1-p(k)]    k=1--->K
    return gini


# 通过计算 gini 系数来寻找分割数据集的最优特征：
# 最优的特征 index、特征值 row[index]、以及分割完的数据 groups(left, right)
def get_split(dataset, n_features):
    """
    :param dataset: 数据集
    :param n_features: 找取的特征的个数
    :return: {'index': b_index, 'value': b_value, 'groups': b_groups}字典
    b_index =》为最优特征的索引
    b_value =》最优特征的特征值
    b_groups =》最优的分组
    """
    # 获取数据集所属类别[M,R]
    class_value = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    features = list()    # 特征索引 list
    while len(features) < n_features:
        index = np.random.randint(len(dataset[0]) - 1)     # 随机获取数据集中的一个特征的索引
        if index not in features:
            features.append(index)
    # 根据给出的n_features的个数进行 split 数据集 group。
    for index in features:
        for row in dataset:
            groups = test_split(index, row[index], dataset)    # 根据不同特征索引，特征值划分数据集 dataset，得到 group
            gini = gini_index(groups, class_value)    # 计算分割后的数据集 group 的 Gini 值
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index': b_index, 'value': b_value, 'groups': b_groups}


# 输出group中出现次数较多的标签
def to_terminal(group):
    """
    :param group: 分组后的数据集
    :return: group中出现次数较多的标签 M or R
    """
    outcomes = [row[-1] for row in group]
    # max() 方法返回给定参数的最大值，参数可以为序列。
    # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
    return max(set(outcomes), key=outcomes.count)     # key=outcomes.count表示用count()函数对outcomes中的两个元素计数


# 创建随机森林中的子分割器进行递归分类，直到分类结束。
def split(node, max_depth, min_size, n_features, depth):
    left, right = node['groups']
    del (node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # max_depth 表示递归次数，若分类还未结束，则选取数据中分类标签较多的作为结果，使分类提前结束，防止过拟合
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        # get_split(left, n_features) 的 return 值 ===》{'index': b_index, 'value': b_value, 'groups': b_groups}
        node['left'] = get_split(left, n_features)     # left 为左支数数据集
        split(node['left'], max_depth, min_size, n_features, depth+1)   # 递归，depth+1计算递归层数
    if len(right) < min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, n_features)
        split(node['right'], max_depth, min_size, n_features, depth+1)


# 构建数
def build_tree(dataset, max_depth, min_size, n_features):
    root = get_split(dataset, n_features)     # 根据初始数据集，划分 root
    split(root, max_depth, min_size, n_features, 1)     # 持续跟新划分树 root
    return root


# 预测模型分类结果
def predict(node, row):
    """
    :param node: 决策树
    :param row: 遍历的dataset内容
    :return: True or False（bool类型数据，布尔数据）
    """
    # 返回 True or False
    if float(row[node['index']]) < float(node['value']):
        if isinstance(node['left'],  dict):    # 判断node['left']是否为字典
            return predict(node['left'], row)    # 形成递归
        else:
            return node['left']
    # 返回 True or False
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']


# 随机森林的分类预测
def bagginig_predict(trees, row):
    """
    bagging 预测
    :param trees: 决策树集合
    :param row: 测试数据集的每一行数据
    :return
        返回随机森林中,决策树结果出现次数最多的那个树
    """
    predictions = [predict(tree, row) for tree in trees]
    return max(predictions, key=predictions.count)


# 训练数据随机化
def subsample(dataset, ratio):
    sample = []
    n_sample = round(len(dataset) * ratio)     # round -->四舍五入
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample     # sample: 随机抽样的训练样本组合


# 随机森林的建立
def random_forest(train, test, max_depth, min_size, sample_size, n_features, n_trees):
    """
    :param train: 训练数据集
    :param test: 测试数据集
    :param max_depth: 决策树深度 不能太深 容易过拟合
    :param min_size: 叶子节点的大小
    :param sample_size: 训练数据集的样本比例
    :param n_features: 选取的特征的个数
    :param n_trees: 决策树的个数
    :return: 每一行的预测结果 bagging 预测最后的分类结果
    """
    trees = []
    for i in range(n_trees):
        sample = subsample(train, sample_size)  # 获取随机的训练数据样本
        tree = build_tree(sample, max_depth, min_size, n_features)  # 根据随机数据集样本构建不同的决策树
        trees.append(tree)    # 生成决策树集合 list
    prediction = [bagginig_predict(trees, row) for row in test]     # 计算决策树中的预测结果，输出最好的那棵决策树
    return prediction


# 计算精确度：导入实际值和预测值
def accuracy_metric(actual, predicted):  # 导入实际值和预测值，计算精确度
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# 评估算法性能，返回模型得分
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    """evaluate_algorithm(评估算法性能，返回模型得分)

    Args:
        dataset     原始数据集
        algorithm   使用的算法
        n_folds     数据的份数
        *args       其他的参数
    Returns:
        scores      模型得分
    """

    # 将数据集进行抽重抽样 n_folds 份，得到 folds 数据集合，数据可以重复重复抽取，每一次 list 的元素是无重复的
    folds = Cross_Validation_split(dataset, n_folds)
    scores = list()
    # 每次循环从 folds 从取出一个 fold 作为测试集，其余作为训练集，遍历整个 folds ，实现交叉验证
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)    # remove() 函数用于移除列表中某个值的第一个匹配项
        # 将多个 fold 列表组合成一个 train_set 列表, 类似 union all
        """
        In [20]: l1=[[1, 2, 'a'], [11, 22, 'b']]
        In [21]: l2=[[3, 4, 'c'], [33, 44, 'd']]
        In [22]: l=[]
        In [23]: l.append(l1)
        In [24]: l.append(l2)
        In [25]: l
        Out[25]: [[[1, 2, 'a'], [11, 22, 'b']], [[3, 4, 'c'], [33, 44, 'd']]]
        In [26]: sum(l, [])
        Out[26]: [[1, 2, 'a'], [11, 22, 'b'], [3, 4, 'c'], [33, 44, 'd']]
        """
        train_set = sum(train_set, [])      # 就是在train_set上再加一个[],即[train_set]
        test_set = list()
        # fold 表示从原始数据集 dataset 提取出来的测试集(去除row[-1]的值)
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            test_set.append(row_copy)
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]

        # 计算随机森林的预测结果的正确率
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def main():
    dataset = loadData('./sonar-all-data.txt')
    n_folds = 5        # 分成5份数据，进行交叉验证
    max_depth = 20     # 调参（自己修改） #决策树深度不能太深，不然容易导致过拟合
    min_size = 1       # 决策树的叶子节点最少的元素数量
    sample_size = 1.0  # 做决策树时候的样本的比例
    n_features = 15    # 调参（自己修改） #准确性与多样性之间的权衡
    for n_trees in [1, 10, 20, 30, 40, 50]:
        scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees,
                                    n_features)
        # 每一次执行本文件时都能产生同一个随机数
        # import numpy as np
        seed(1)
        print('random=', random())
        print('Trees: %d' % n_trees)
        print('Scores: %s' % scores)
        print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))


if __name__ == '__main__':
    main()

    # import numpy as np
    # dataset = loadData('./sonar-all-data.txt')
    # outcomes = [row[-1] for row in dataset]
    # print(max(set(outcomes), key=outcomes.count))
    # # class_values = list(set(row[-1] for row in dataset))
    # # print(class_values)
    # for row in dataset:
    #     # print(row[-1])
    #     for i in range(10):
    #         pass
    #         # print(row[i])

    # print(type(dataset))

    # print(np.mat(dataset).shape)
    # print(len(dataset))
    # dataset_split = Cross_Validation_split(dataset, 6)    # 分成6块
    # print(np.array(dataset_split).T.shape)
    # print(np.array(dataset_split)[0].shape)
    # print(np.array(dataset_split).shape)

    # for row in dataset:
    #     print(row[randrange(len(dataset[0])-1)])
    # # print(randrange(len(dataset[0])-1))
    # for n_trees in [1, 10, 20]:  # 理论上树是越多越好
    #     scores = evaluate_algorithm(dataset, random_forest, n_folds, max_depth, min_size, sample_size, n_trees, n_features)
    #     # 每一次执行本文件时都能产生同一个随机数
    #     seed(1)
    #     print('random=', random())
    #     print('Trees: %d' % n_trees)
    #     print('Scores: %s' % scores)
    #     print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
