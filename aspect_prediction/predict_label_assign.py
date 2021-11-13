#encoding:utf-8

'''
    该py文件的作用在确定了类别的情况下，为每个类别人工分配标签
'''

#case 1：
#当选用768+150维，且类别结果如下所示：
# B-VAV:5
# B-VAT:5
# B-VR:10

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

content = pd.read_csv('predict.csv')

cvdids = content['ID']
descriptions = content['description']
tokens = [eval(token) for token in content['tokens']]
labels = [eval(label) for label in content['predict_label']]
vectors = [np.array(eval(vector)) for vector in content['embedding_vector']]  # 最外层是一个列表，列表中的每一个位置上都是一个二维矩阵
length = len(cvdids)

B_VAV_train_test_index = []  # 存储用于训练B-VAV的数据集，每一个存储的是索引值
B_VAT_train_test_index = []
B_VR_train_test_index = []
B_VAV_predict_index = []  # 存储需要预测B-VAV的数据集
B_VAT_predict_index = []
B_VR_predict_index = []
# 全部过程都用索引值来记就ok

for i in range(length):
    if 'B-VAV' in labels[i] and 'B-VN' in labels[i] and 'B-VAT' in labels[i] and 'B-VR' in labels[i]:
        B_VAV_train_test_index.append(i)
        B_VAT_train_test_index.append(i)
        B_VR_train_test_index.append(i)
    elif 'B-VAV' not in labels[i] and 'B-VN' in labels[i] and 'B-VAT' in labels[i] and 'B-VR' in labels[i]:
        B_VAV_predict_index.append(i)
    elif 'B-VAV' in labels[i] and 'B-VN' in labels[i] and 'B-VAT' not in labels[i] and 'B-VR' in labels[i]:
        B_VAT_predict_index.append(i)
    elif 'B-VAV' in labels[i] and 'B-VN' in labels[i] and 'B-VAT' in labels[i] and 'B-VR' not in labels[i]:
        B_VR_predict_index.append(i)

# 先从训练集中把每个标签的类别确定出来，那么目前就是根据当前的向量将每一个训练集中该属性的向量表示出来，先以B_VAT为例
def function(start_label, end_label, train_test_set, label_set, vector_set):
    result_vector = []  # 用于存储所有节点的向量值，外层为列表，每个列表位置上为一个918维的array，列表长度与train_test_set一致
    for item in train_test_set:
        temp_vector = np.zeros(vector_set[item].shape[1])  # 生成与token维度等长的全0数组
        start_index = []
        end_index = []
        for j in range(len(label_set[item]) - 1):
            if labels[item][j] == start_label:  # 起始位置
                start_index.append(j)
            elif labels[item][j] == end_label and labels[item][j + 1] != end_label:  # 末尾位置
                end_index.append(j)
        # 还需要将最后一个位置考虑进去
        if labels[item][-1] == start_label:
            start_index.append(len(label_set[item]) - 1)
        if labels[item][-1] == end_label:
            end_index.append(len(label_set[item]) - 1)

        if len(start_index) == 1 and 1 == len(end_index):  # 如果只有一个VAT，且属性长度大于1时
            for value in range(start_index[0], end_index[0] + 1):
                temp_vector += vector_set[item][value]
            temp_vector /= end_index[0] - start_index[0] + 1
        elif len(start_index) == 1 and len(end_index) == 0:  # 如果只有一个VAT，且属性长度刚好等于1时
            temp_vector = vector_set[item][start_index[0]]
        elif len(start_index) > 1 and len(end_index) == 0:  # 如果有多个VAT，且每个的长度均等于1时
            for value in start_index:
                temp_vector += vector_set[item][value]
            temp_vector /= len(start_index)
        elif len(start_index) == len(end_index) and len(start_index) > 1:  # 如果有多个VAT，且每一个的长度都大于1，则选择长度更短的那个VAT作为最终结果
            min_gap = end_index[0] - start_index[0]
            start = start_index[0]
            end = end_index[0]
            for i in range(1, len(start_index)):
                if min_gap > end_index[i] - start_index[i]:
                    min_gap = end_index[i] - start_index[i]
                    start = start_index[i]
                    end = end_index[i]
            for value in range(start, end + 1):
                temp_vector += vector_set[item][value]
            temp_vector /= min_gap + 1
        result_vector.append(temp_vector)
    return result_vector


B_VAT_cluster_vectors=np.array(function('B-VAT','I-VAT',B_VAT_train_test_index,labels,vectors))[:,:-150]
# B_VAV_cluster_vectors = np.array(function('B-VAV', 'I-VAV', B_VAV_train_test_index, labels, vectors))[:,:-150]    #只取前768的word embedding
# B_VR_cluster_vectors=np.array(function('B-VR','I-VR',B_VR_train_test_index,labels,vectors))[:,:-150]

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(B_VAT_cluster_vectors)
predict_labels = kmeans.labels_

label_index=[[] for i in range(len(predict_labels))]    #用于存放所有
label_value=[[] for i in range(len(predict_labels))]

for i in range(len(predict_labels)):
    label_index[i].extend([j for j,x in enumerate(predict_labels) if x==i+1])

for item in label_index:
    for j in item:


# for item in label_index:
#     print(item)
#     print(len(item))
#case 2：
#当选用768维，且类别结果如下所示：
# B-VAV:6
# B-VAT:5
# B-VR:9